"""HTTP server usage queries contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from fastapi.testclient import TestClient
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    UsageProvider,
    _aggregate_usage_json,
    _collect_run,
    _price_book_payload,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.usage import CacheUsageMetrics, UsageMetrics
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def test_server_usage_aggregation_preserves_cache_ttl_buckets() -> None:
    from cayu.runtime._usage_accounting import UsageAccountingReducer

    metrics = (
        UsageMetrics(
            cache=CacheUsageMetrics(
                write_tokens=5,
                write_5m_tokens=2,
                write_1h_tokens=3,
            )
        ),
        UsageMetrics(
            cache=CacheUsageMetrics(
                write_tokens=7,
                write_5m_tokens=7,
                write_unknown_ttl_tokens=1,
            )
        ),
    )

    reducer = UsageAccountingReducer(EventQuery(), by_session=False, by_identity=True)
    reducer.add_page(
        [
            EventRecord(
                sequence=index,
                event=Event(
                    session_id="cache-summary",
                    type=EventType.MODEL_COMPLETED,
                    payload={"usage_metrics": item.model_dump(mode="json")},
                ),
            )
            for index, item in enumerate(metrics, 1)
        ]
    )
    combined = reducer.snapshot().provider_summaries[0].usage

    assert combined.cache.write_tokens == 12
    assert combined.cache.write_5m_tokens == 9
    assert combined.cache.write_1h_tokens == 3
    assert combined.cache.write_unknown_ttl_tokens == 1


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/usage_1/usage")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "usage_1",
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
    }


def test_server_usage_summaries_serialize_counters_beyond_int64() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    maximum = MAX_DURABLE_JSON_INTEGER
    expected = str(maximum * 2)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="usage_overflow",
                causal_budget_id="budget_overflow",
                messages=[Message.text("user", "seed")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "usage_overflow",
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="usage_overflow",
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": maximum,
                            "output_tokens": maximum,
                            "total_tokens": maximum,
                        }
                    },
                )
                for _ in range(2)
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_response = client.get("/api/sessions/usage_overflow/usage")
    causal_response = client.get("/api/causal-budgets/budget_overflow/usage")
    aggregate_response = client.post("/api/sessions/summary", json={})

    assert session_response.status_code == 200
    assert session_response.json()["usage"]["total_tokens"] == expected
    assert causal_response.status_code == 200
    assert causal_response.json()["usage"]["total_tokens"] == expected
    assert aggregate_response.status_code == 200
    aggregate = aggregate_response.json()
    assert aggregate["usage"]["usage"]["total_tokens"] == expected
    assert aggregate["provider_breakdown"][0]["usage"]["total_tokens"] == expected
    assert aggregate["model_breakdown"][0]["usage"]["total_tokens"] == expected


def test_server_usage_summaries_tolerate_malformed_optional_usage_fields() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="usage_malformed_optional",
                causal_budget_id="usage_malformed_budget",
                messages=[Message.text("user", "seed")],
            ),
            identity=SessionIdentity(provider_name="fake", model="valid-model"),
        )
        await store.append_events(
            "usage_malformed_optional",
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="usage_malformed_optional",
                    payload={
                        "usage_metrics": {
                            "provider_name": " fake ",
                            "model": "valid-model",
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                            "reasoning_output_tokens": "not-an-integer",
                            "custom_counter": 1,
                            "cache": {
                                "read_tokens": 4,
                                "write_tokens": -1,
                            },
                        }
                    },
                )
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {
        "pricing": _price_book_payload(
            provider_name="fake",
            model="valid-model",
        )
    }

    session_response = client.get("/api/sessions/usage_malformed_optional/usage")
    aggregate_response = client.post("/api/sessions/summary", json=pricing_body)
    session_cost_response = client.post(
        "/api/sessions/usage_malformed_optional/cost",
        json=pricing_body,
    )
    causal_cost_response = client.post(
        "/api/causal-budgets/usage_malformed_budget/cost",
        json=pricing_body,
    )

    assert session_response.status_code == 200
    session_usage = session_response.json()
    assert session_usage["provider_names"] == []
    assert session_usage["models"] == ["valid-model"]
    assert session_usage["usage"]["input_tokens"] == "7"
    assert session_usage["usage"]["output_tokens"] == "3"
    assert session_usage["usage"]["total_tokens"] == "10"
    assert session_usage["usage"]["reasoning_output_tokens"] == "0"
    assert session_usage["usage"]["cache"]["read_tokens"] == "4"
    assert session_usage["usage"]["cache"]["write_tokens"] == "0"

    assert aggregate_response.status_code == 200
    aggregate = aggregate_response.json()
    assert aggregate["usage"]["model_steps"] == 1
    assert aggregate["usage"]["usage"]["total_tokens"] == "10"
    assert aggregate["provider_breakdown"][0]["provider_name"] is None
    assert aggregate["provider_breakdown"][0]["usage"]["total_tokens"] == "10"
    assert aggregate["model_breakdown"][0]["provider_name"] is None
    assert aggregate["cost"]["model_steps"] == 1
    assert aggregate["cost"]["priced_model_steps"] == 0
    assert aggregate["cost"]["unpriced_model_steps"] == 1
    assert aggregate["cost"]["total_cost"] == "0"
    assert aggregate["cost"]["line_items"][0]["missing_pricing_reason"] == (
        "model.completed event has no valid normalized usage metrics"
    )

    for response in (session_cost_response, causal_cost_response):
        assert response.status_code == 200
        cost = response.json()
        assert cost["model_steps"] == 1
        assert cost["priced_model_steps"] == 0
        assert cost["unpriced_model_steps"] == 1
        assert cost["total_cost"] == "0"
        assert cost["line_items"][0]["missing_pricing_reason"] == (
            "model.completed event has no valid normalized usage metrics"
        )
    assert aggregate["model_breakdown"][0]["model"] == "valid-model"
    assert aggregate["model_breakdown"][0]["usage"]["total_tokens"] == "10"


def test_server_exposes_separate_store_local_operational_snapshots() -> None:
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(session_store=session_store, task_store=task_store)

    async def seed() -> None:
        await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="aggregate-running",
                labels={"team": "red"},
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await session_store.update_status("aggregate-running", SessionStatus.RUNNING)
        await task_store.create_task(
            TaskCreate(
                task_id="aggregate-task",
                type="deploy",
                assigned_agent_name="assistant",
            )
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/operations/snapshot",
        json={
            "session_filter": {"labels": {"team": "red"}},
            "task_filter": {"assigned_agent_name": "assistant"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "configured_stores"
    assert body["cross_store_atomic"] is False
    assert body["sessions"]["total_count"] == "1"
    assert body["sessions"]["counts_by_status"]["running"] == "1"
    assert body["sessions"]["accuracy"] == {
        "kind": "exact",
        "reason": None,
        "limit": None,
    }
    assert body["task_snapshot_status"] == "available"
    assert body["tasks"]["counts_by_status"]["pending"] == "1"
    assert body["tasks"]["claimable_pending_count"] == "1"
    assert body["tasks"]["scheduled_pending_count"] == "0"

    without_tasks = client.post(
        "/api/operations/snapshot",
        json={"include_tasks": False},
    ).json()
    assert without_tasks["task_snapshot_status"] == "not_requested"
    assert without_tasks["tasks"] is None

    not_configured = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG)).post(
        "/api/operations/snapshot",
        json={},
    )
    assert not_configured.status_code == 200
    assert not_configured.json()["task_snapshot_status"] == "not_configured"

    class UnsupportedTaskAggregateStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def aggregate_operational_snapshot(self, filters=None):
            raise NotImplementedError

    unsupported = TestClient(
        create_server(
            CayuApp(task_store=UnsupportedTaskAggregateStore()), config=_LOCAL_SERVER_CONFIG
        )
    ).post("/api/operations/snapshot", json={})
    assert unsupported.status_code == 200
    assert unsupported.json()["task_snapshot_status"] == "unsupported"
    assert unsupported.json()["tasks"] is None


def test_server_exposes_bounded_event_time_usage_rollup_and_cost() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="aggregate-usage",
                labels={"team": "red"},
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "aggregate-usage",
            [
                Event(
                    id="aggregate-model",
                    type=EventType.MODEL_COMPLETED,
                    session_id="aggregate-usage",
                    timestamp=start,
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
                Event(
                    id="aggregate-tool",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="aggregate-usage",
                    timestamp=start + timedelta(minutes=1),
                ),
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
            "session_filter": {"labels": {"team": "red"}},
            "group_limit": 10,
            "session_group_limit": 10,
            "pricing": _price_book_payload(),
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["scope"] == "configured_session_store"
    assert body["time_basis"] == "event.timestamp"
    assert body["session_filter_basis"] == "current_session_attributes"
    assert body["matching_session_count"] == "1"
    assert body["active_session_count"] == "1"
    assert body["totals"]["model_steps"] == "1"
    assert body["totals"]["tool_calls"] == "1"
    assert body["provider_breakdown"]["groups"][0]["provider_name"] == "fake"
    assert body["cost"]["accuracy"]["kind"] == "exact"
    assert body["cost"]["currencies"] == [
        {"currency": "USD", "model_steps": "1", "auxiliary_attempts": "0", "total_cost": "1"}
    ]
    assert body["cost"]["billing_breakdown"] == {
        "identified_model_steps": "0",
        "identified_auxiliary_attempts": "0",
        "groups": [],
        "remainder": None,
        "accuracy": {"kind": "exact", "limit": None, "reason": None},
    }
    assert body["session_breakdown"] == {
        "order": "tokens",
        "groups": [
            {
                "session_id": "aggregate-usage",
                "status": "pending",
                "agent_name": "assistant",
                "title": None,
                "active": True,
                "totals": body["totals"],
            }
        ],
        "remainder": None,
        "accuracy": {"kind": "exact", "limit": None, "reason": None},
    }
    assert body["session_cost_breakdown"] == {
        "price_book_version": "test",
        "price_book_generated_at": "2026-07-13",
        "groups": [
            {
                "session_id": "aggregate-usage",
                "cost": {
                    "accuracy": {"kind": "exact", "limit": None, "reason": None},
                    "evaluated_model_steps": "1",
                    "priced_model_steps": "1",
                    "unpriced_model_steps": "0",
                    "unevaluated_model_steps": "0",
                    "evaluated_auxiliary_attempts": "0",
                    "priced_auxiliary_attempts": "0",
                    "unpriced_auxiliary_attempts": "0",
                    "currencies": [
                        {
                            "currency": "USD",
                            "model_steps": "1",
                            "auxiliary_attempts": "0",
                            "total_cost": "1",
                        }
                    ],
                    "unpriced_reasons": [],
                },
            }
        ],
        "remainder": None,
        "accuracy": {"kind": "exact", "limit": None, "reason": None},
    }
    assert "pricing_inputs" not in body

    shared_only = client.post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
            "session_filter": {"labels": {"team": "red"}},
            "group_limit": 10,
            "pricing": _price_book_payload(),
        },
    ).json()
    assert shared_only["totals"] == body["totals"]
    assert shared_only["provider_breakdown"] == body["provider_breakdown"]
    assert shared_only["model_breakdown"] == body["model_breakdown"]
    assert shared_only["cost"] == body["cost"]
    assert shared_only["session_breakdown"] is None
    assert shared_only["session_cost_breakdown"] is None

    usage_only = client.post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
            "session_filter": {"labels": {"team": "red"}},
            "group_limit": 10,
            "session_group_limit": 10,
        },
    ).json()
    assert usage_only["totals"] == body["totals"]
    assert usage_only["session_breakdown"] == body["session_breakdown"]
    assert usage_only["cost"] is None
    assert usage_only["session_cost_breakdown"] is None


def test_server_rejects_response_amplifying_usage_currency_before_store_work() -> None:
    class StoreWorkMustNotStart(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        aggregate_called = False

        async def aggregate_usage(self, query):
            self.aggregate_called = True
            return await super().aggregate_usage(query)

    store = StoreWorkMustNotStart()
    app = CayuApp(session_store=store)
    pricing = cast("dict[str, Any]", _price_book_payload())
    pricing["prices"][0]["schedules"][0]["pricing"]["currency"] = "X" * 65
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
            "group_limit": 1,
            "session_group_limit": 100,
            "pricing": pricing,
        },
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body"],
                "msg": "Invalid usage rollup request.",
            }
        ]
    }
    assert len(response.content) < 1024
    assert store.aggregate_called is False

    accepted_pricing = cast("dict[str, Any]", _price_book_payload())
    accepted_pricing["prices"][0]["schedules"][0]["pricing"]["currency"] = ("€" * 21) + "X"
    accepted = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
            "session_group_limit": 100,
            "pricing": accepted_pricing,
        },
    )
    assert accepted.status_code == 200
    assert store.aggregate_called is True


@pytest.mark.parametrize("with_pricing", [False, True])
def test_server_rejects_secret_bearing_usage_session_authority(
    with_pricing: bool,
) -> None:
    secret = "usage-session-secret-canary"
    session_id = f"customer-{secret}"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            session_id,
            Event(
                id="secret-session-model",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                timestamp=start,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            ),
        )

    asyncio.run(seed())
    request: dict[str, Any] = {
        "start_at": start.isoformat(),
        "end_at": (start + timedelta(days=1)).isoformat(),
        "session_group_limit": 1,
    }
    if with_pricing:
        request["pricing"] = _price_book_payload()
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).post(
        "/api/usage/rollup",
        json=request,
    )

    assert response.status_code == 409
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "detail": "Usage session identity cannot cross the configured redaction boundary."
    }
    assert secret not in response.text
    assert REDACTED_SECRET not in response.text


def test_server_rejects_oversized_usage_session_identity_without_reflecting_it() -> None:
    store = InMemorySessionStore()
    session_id = "private-" + ("s" * 1025)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())
    response = TestClient(
        create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG)
    ).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
            "session_group_limit": 1,
        },
    )

    assert response.status_code == 413
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Usage rollup result exceeds the server byte limit."}
    assert session_id not in response.text


def test_server_aggregates_oversized_omitted_session_identity_without_reflecting_it() -> None:
    store = InMemorySessionStore()
    retained_session_id = "retained-usage-session"
    omitted_session_id = "private-" + ("s" * 1025)
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        for session_id in (retained_session_id, omitted_session_id):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        await store.append_event(
            retained_session_id,
            Event(
                id="retained-usage-model",
                type=EventType.MODEL_COMPLETED,
                session_id=retained_session_id,
                timestamp=start,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            ),
        )

    asyncio.run(seed())
    response = TestClient(
        create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG)
    ).post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
            "session_group_limit": 1,
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert [group["session_id"] for group in body["session_breakdown"]["groups"]] == [
        retained_session_id
    ]
    assert body["session_breakdown"]["remainder"]["group_count"] == "1"
    assert body["matching_session_count"] == "2"
    assert omitted_session_id not in response.text


def test_server_serializes_aggregate_counters_without_javascript_rounding() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    maximum = 2**63 - 1

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="aggregate-large-json-counter",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "aggregate-large-json-counter",
            [
                Event(
                    id=f"aggregate-large-json-counter-{index}",
                    type=EventType.MODEL_COMPLETED,
                    session_id="aggregate-large-json-counter",
                    timestamp=start + timedelta(minutes=index),
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": maximum,
                            "output_tokens": 0,
                            "total_tokens": maximum,
                        }
                    },
                )
                for index in range(2)
            ],
        )

    asyncio.run(seed())
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["model_steps"] == "2"
    assert body["totals"]["usage"]["input_tokens"] == str(2 * maximum)
    assert body["provider_breakdown"]["groups"][0]["totals"]["usage"]["input_tokens"] == str(
        2 * maximum
    )


def test_server_rejects_price_books_and_resolution_work_without_reflecting_input() -> None:
    client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    request: dict[str, Any] = {
        "start_at": "2026-07-01T00:00:00Z",
        "end_at": "2026-07-02T00:00:00Z",
        "pricing": _price_book_payload(),
    }
    price = cast("list[dict[str, object]]", request["pricing"]["prices"])[0]

    def assert_sanitized_validation_error(response) -> None:
        assert response.status_code == 422
        assert response.headers["cache-control"] == "private, no-store"
        assert response.json() == {
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body"],
                    "msg": "Invalid usage rollup request.",
                }
            ]
        }
        assert len(response.content) < 1024

    too_many_prices = json.loads(json.dumps(request))
    too_many_prices["pricing"]["prices"] = [price] * 501
    response = client.post("/api/usage/rollup", json=too_many_prices)
    assert_sanitized_validation_error(response)

    oversized_price_book = json.loads(json.dumps(request))
    oversized_price_book["pricing"]["price_book_version"] = "x" * (2 * 1024 * 1024)
    response = client.post("/api/usage/rollup", json=oversized_price_book)
    assert_sanitized_validation_error(response)

    excessive_resolution_work = json.loads(json.dumps(request))
    excessive_resolution_work["pricing_input_limit"] = 1000
    excessive_resolution_work["pricing"]["prices"][0]["aliases"] = [
        f"alias-{index}" for index in range(500)
    ]
    response = client.post("/api/usage/rollup", json=excessive_resolution_work)
    assert_sanitized_validation_error(response)

    excessive_context_values = json.loads(json.dumps(request))
    context_price = excessive_context_values["pricing"]["prices"][0]
    context_price["match"] = "exact"
    context_price["pricing_context"] = {
        "dimensions": {
            "region": [f"region-{index}" for index in range(1001)],
            "tier": [f"tier-{index}" for index in range(1001)],
        }
    }
    response = client.post("/api/usage/rollup", json=excessive_context_values)
    assert_sanitized_validation_error(response)

    excessive_context_work = json.loads(json.dumps(request))
    excessive_context_work["pricing_input_limit"] = 1000
    context_price = excessive_context_work["pricing"]["prices"][0]
    context_price["match"] = "exact"
    context_price["pricing_context"] = {
        "dimensions": {"tier": [f"tier-{index}" for index in range(500)]}
    }
    response = client.post("/api/usage/rollup", json=excessive_context_work)
    assert_sanitized_validation_error(response)

    bounded_shared_work = json.loads(json.dumps(request))
    bounded_shared_work["pricing_input_limit"] = 5000
    bounded_shared_work["pricing"]["prices"][0]["aliases"] = [
        f"alias-{index}" for index in range(47)
    ]
    assert client.post("/api/usage/rollup", json=bounded_shared_work).status_code == 200
    bounded_shared_work["session_group_limit"] = 10
    reflected_secret = "WORKLOAD-SECRET-SESSION-PROJECTION"
    bounded_shared_work["pricing"]["prices"][0]["schedules"][0]["provenance"]["url"] = (
        f"https://private.invalid/{reflected_secret}"
    )
    response = client.post("/api/usage/rollup", json=bounded_shared_work)
    assert_sanitized_validation_error(response)
    assert reflected_secret not in response.text

    oversized_body = b'{"ignored":"' + (b"x" * (3 * 1024 * 1024)) + b'"}'
    response = client.post(
        "/api/usage/rollup",
        content=oversized_body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Usage rollup request exceeds the server byte limit."}


def test_server_aggregate_routes_reject_invalid_windows_and_unsupported_stores() -> None:
    class UnsupportedAggregateStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_usage_aggregates = False

        async def aggregate_operational_snapshot(self, filters=None):
            raise NotImplementedError

        async def aggregate_usage(self, query):
            raise NotImplementedError

    client = TestClient(
        create_server(
            CayuApp(session_store=UnsupportedAggregateStore()),
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(runtime_config={"priceBook": _price_book_payload()})
            ),
        )
    )
    assert client.post("/api/operations/snapshot", json={}).status_code == 501
    surfaces = client.get("/api/contract").json()["capabilities"]["surfaces"]
    assert surfaces["usage"] == {
        "configured": True,
        "read": {"enabled": False, "unavailable_reason": "unsupported"},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    assert surfaces["pricing"] == {
        "configured": True,
        "read": {"enabled": False, "unavailable_reason": "unsupported"},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    unsupported_usage = client.post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
        },
    )
    assert unsupported_usage.status_code == 501
    openapi = client.get("/openapi.json").json()
    for path in ("/api/operations/snapshot", "/api/usage/rollup"):
        error_schema = openapi["paths"][path]["post"]["responses"]["501"]["content"][
            "application/json"
        ]["schema"]
        assert error_schema == {"$ref": "#/components/schemas/ApiErrorResponse"}
    oversized_schema = openapi["paths"]["/api/usage/rollup"]["post"]["responses"]["413"]["content"][
        "application/json"
    ]["schema"]
    assert oversized_schema == {"$ref": "#/components/schemas/ApiErrorResponse"}
    validation_schema = openapi["paths"]["/api/usage/rollup"]["post"]["responses"]["422"][
        "content"
    ]["application/json"]["schema"]
    assert validation_schema == {"$ref": "#/components/schemas/HTTPValidationError"}
    for status_code in ("409", "500"):
        error_schema = openapi["paths"]["/api/usage/rollup"]["post"]["responses"][status_code][
            "content"
        ]["application/json"]["schema"]
        assert error_schema == {"$ref": "#/components/schemas/ApiErrorResponse"}

    valid_client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    invalid_window = valid_client.post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-02T00:00:00Z",
            "end_at": "2026-07-01T00:00:00Z",
        },
    )
    assert invalid_window.status_code == 422
    assert invalid_window.headers["cache-control"] == "private, no-store"
    assert invalid_window.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body"],
                "msg": "Invalid usage rollup request.",
            }
        ]
    }

    class InvalidAggregateResultStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            from cayu.budgets.aggregates import UsageRollupStoreResult

            return UsageRollupStoreResult.model_validate({})

    invalid_store_client = TestClient(
        create_server(
            CayuApp(session_store=InvalidAggregateResultStore()), config=_LOCAL_SERVER_CONFIG
        ),
        raise_server_exceptions=False,
    )
    invalid_store_response = invalid_store_client.post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
        },
    )
    assert invalid_store_response.status_code == 500
    assert invalid_store_response.headers["cache-control"] == "private, no-store"
    assert invalid_store_response.json() == {
        "detail": "The configured session store returned an inconsistent usage projection."
    }


def test_server_rejects_custom_usage_store_results_that_ignore_query_bounds() -> None:
    class IgnoringUsageQueryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            return await super().aggregate_usage(
                query.model_copy(update={"session_group_limit": 2})
            )

    store = IgnoringUsageQueryStore()

    async def seed() -> None:
        for session_id in ("custom-store-one", "custom-store-two"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )

    asyncio.run(seed())
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))
    base_request = {
        "start_at": "2026-07-01T00:00:00Z",
        "end_at": "2026-07-02T00:00:00Z",
    }
    for body in (base_request, {**base_request, "session_group_limit": 1}):
        response = client.post("/api/usage/rollup", json=body)
        assert response.status_code == 500
        assert response.headers["cache-control"] == "private, no-store"
        assert response.json() == {
            "detail": "The configured session store returned an inconsistent usage projection."
        }


def test_server_rejects_custom_usage_store_with_inconsistent_session_totals() -> None:
    class InconsistentSessionTotalsStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            assert result.session_breakdown is not None
            group = result.session_breakdown.groups[0]
            inconsistent_group = group.model_copy(
                update={
                    "totals": group.totals.model_copy(
                        update={"usage": group.totals.usage.model_copy(update={"total_tokens": 1})}
                    )
                }
            )
            return result.model_copy(
                update={
                    "session_breakdown": result.session_breakdown.model_copy(
                        update={"groups": (inconsistent_group,)}
                    )
                }
            )

    store = InconsistentSessionTotalsStore()
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="custom-inconsistent-totals",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            "custom-inconsistent-totals",
            Event(
                id="custom-inconsistent-totals-model",
                type=EventType.MODEL_COMPLETED,
                session_id="custom-inconsistent-totals",
                timestamp=start,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 100,
                        "output_tokens": 0,
                        "total_tokens": 100,
                    }
                },
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
            "session_group_limit": 1,
        },
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "detail": "The configured session store returned an inconsistent usage projection."
    }


def test_server_canonically_revalidates_custom_usage_store_results() -> None:
    class InvalidNestedUsageStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            provider_group = result.provider_breakdown.groups[0]
            invalid_usage = provider_group.totals.usage.model_copy(update={"total_tokens": -1})
            invalid_group = provider_group.model_copy(
                update={"totals": provider_group.totals.model_copy(update={"usage": invalid_usage})}
            )
            return result.model_copy(
                update={
                    "provider_breakdown": result.provider_breakdown.model_copy(
                        update={"groups": (invalid_group,)}
                    )
                }
            )

    store = InvalidNestedUsageStore()
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="custom-invalid-nested-usage",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            "custom-invalid-nested-usage",
            Event(
                id="custom-invalid-nested-model",
                type=EventType.MODEL_COMPLETED,
                session_id="custom-invalid-nested-usage",
                timestamp=start,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 100,
                        "output_tokens": 0,
                        "total_tokens": 100,
                    }
                },
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/usage/rollup",
        json={
            "start_at": start.isoformat(),
            "end_at": (start + timedelta(days=1)).isoformat(),
        },
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "detail": "The configured session store returned an inconsistent usage projection."
    }


def test_server_classifies_nonserializable_custom_usage_result_as_inconsistent() -> None:
    class NonserializableUsageStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            return result.model_copy(update={"provider_breakdown": object()})

    response = TestClient(
        create_server(
            CayuApp(session_store=NonserializableUsageStore()),
            config=_LOCAL_SERVER_CONFIG,
        ),
        raise_server_exceptions=False,
    ).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
        },
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "detail": "The configured session store returned an inconsistent usage projection."
    }


def test_server_classifies_custom_retained_usage_identity_byte_overflow_as_413() -> None:
    oversized_session_id = "s" * 1025

    class OversizedRetainedIdentityStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        include_invalid_counter = False

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            assert result.session_breakdown is not None
            group = result.session_breakdown.groups[0]
            updates: dict[str, Any] = {"session_id": oversized_session_id}
            if self.include_invalid_counter:
                updates["totals"] = group.totals.model_copy(
                    update={"usage": group.totals.usage.model_copy(update={"total_tokens": -1})}
                )
            oversized_group = group.model_copy(update=updates)
            return result.model_copy(
                update={
                    "session_breakdown": result.session_breakdown.model_copy(
                        update={"groups": (oversized_group,)}
                    )
                }
            )

    store = OversizedRetainedIdentityStore()

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="custom-retained-identity",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())
    response = TestClient(
        create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG),
        raise_server_exceptions=False,
    ).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
            "session_group_limit": 1,
        },
    )

    assert response.status_code == 413
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Usage rollup result exceeds the server byte limit."}
    assert oversized_session_id not in response.text

    store.include_invalid_counter = True
    mixed_failure = TestClient(
        create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG),
        raise_server_exceptions=False,
    ).post(
        "/api/usage/rollup",
        json={
            "start_at": "2026-07-01T00:00:00Z",
            "end_at": "2026-07-02T00:00:00Z",
            "session_group_limit": 1,
        },
    )
    assert mixed_failure.status_code == 500
    assert mixed_failure.headers["cache-control"] == "private, no-store"
    assert mixed_failure.json() == {
        "detail": "The configured session store returned an inconsistent usage projection."
    }
    assert oversized_session_id not in mixed_failure.text


def test_server_rejects_custom_usage_store_with_misattributed_session_pricing() -> None:
    class MisattributedSessionPricingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            session_items = result.session_pricing_inputs
            assert session_items
            if len(session_items) == 1:
                session_item = session_items[0]
                assert session_item.metrics is not None
                replacement = next(
                    item.metrics
                    for item in result.pricing_inputs
                    if item.metrics is not None and item.metrics.model != session_item.metrics.model
                )
                misattributed = (session_item.model_copy(update={"metrics": replacement}),)
            else:
                first, second = session_items
                misattributed = (
                    first.model_copy(update={"metrics": second.metrics}),
                    second.model_copy(update={"metrics": first.metrics}),
                )
            return result.model_copy(update={"session_pricing_inputs": misattributed})

    store = MisattributedSessionPricingStore()
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        for session_id, model in (
            ("custom-cheap", "cheap-model"),
            ("custom-expensive", "expensive-model"),
        ):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model=model),
            )
            await store.append_event(
                session_id,
                Event(
                    id=f"{session_id}-pricing",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=start,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": model,
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        }
                    },
                ),
            )

    asyncio.run(seed())
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))
    request = {
        "start_at": start.isoformat(),
        "end_at": (start + timedelta(days=1)).isoformat(),
        "pricing": _price_book_payload(),
    }
    for session_group_limit in (1, 2):
        response = client.post(
            "/api/usage/rollup",
            json={**request, "session_group_limit": session_group_limit},
        )
        assert response.status_code == 500
        assert response.headers["cache-control"] == "private, no-store"
        assert response.json() == {
            "detail": "The configured session store returned an inconsistent usage projection."
        }


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/summary",
        params=[
            ("label", "organization=org_123"),
            ("label_selector", "project in (ap_q2,research)"),
            ("order_by", "created_at_asc"),
        ],
        json={
            "pricing": _price_book_payload(
                standard=[
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
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_count"] == 2
    assert body["total_count"] == 2
    assert body["next_cursor"] is None
    assert [item["session"]["id"] for item in body["sessions"]] == [
        "summary_filter_invoice",
        "summary_filter_research",
    ]
    assert body["usage"]["session_count"] == 2
    assert body["usage"]["usage"]["total_tokens"] == "24"
    assert body["provider_breakdown"] == [
        {
            "provider_name": "fake",
            "model": None,
            "session_count": 2,
            "model_steps": 2,
            "usage": _aggregate_usage_json(
                input_tokens=20,
                output_tokens=4,
                total_tokens=24,
                cache_read_tokens=8,
                cached_input_tokens=8,
                uncached_input_tokens=12,
            ),
        }
    ]
    assert body["model_breakdown"] == [
        {
            "provider_name": "fake",
            "model": "fake-model",
            "session_count": 2,
            "model_steps": 2,
            "usage": _aggregate_usage_json(
                input_tokens=20,
                output_tokens=4,
                total_tokens=24,
                cache_read_tokens=8,
                cached_input_tokens=8,
                uncached_input_tokens=12,
            ),
        }
    ]
    assert body["cost"]["session_count"] == 2
    assert body["cost"]["total_cost"] == "0.000216"
    assert body["cost"]["line_items"][0]["pricing_tier_max_input_tokens"] is None
    assert body["cost"]["line_items"][0]["pricing_provenance"] == {
        "source": "official",
        "url": "https://example.com/pricing",
        "as_of": "2026-07-13",
    }
    assert [item["session_id"] for item in body["cost"]["session_costs"]] == [
        "summary_filter_invoice",
        "summary_filter_research",
    ]


def test_server_filtered_sessions_summary_uses_one_native_accounting_read() -> None:
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
    original_accounting = app.session_store.read_usage_accounting

    async def read_usage_accounting(query, *, by_session=False, by_identity=False):
        queries.append(query)
        return await original_accounting(query, by_session=by_session, by_identity=by_identity)

    app.session_store.read_usage_accounting = read_usage_accounting

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/sessions/summary",
        params={"label": "organization=org_123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_count"] == 1
    assert body["total_count"] == 1
    assert body["next_cursor"] is None
    assert body["sessions"][0]["session"]["id"] == "summary_no_body"
    assert body["usage"]["usage"]["total_tokens"] == "12"
    assert body["cost"] is None


def test_server_sessions_summary_filters_debug_states_before_pagination() -> None:
    app = CayuApp()

    async def create(session_id: str, status: SessionStatus, events: list[Event]) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, status)
        await app.session_store.append_events(session_id, events)

    async def seed() -> None:
        await create(
            "debug_normal_completed",
            SessionStatus.COMPLETED,
            [
                Event(
                    id="debug_normal_completed_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id="debug_normal_completed",
                )
            ],
        )
        await create(
            "debug_tool_failed_completed",
            SessionStatus.COMPLETED,
            [
                Event(
                    id="debug_tool_failed_event",
                    type=EventType.TOOL_CALL_FAILED,
                    session_id="debug_tool_failed_completed",
                    tool_name="deploy_service",
                    payload={"error": "deploy failed"},
                ),
                Event(
                    id="debug_tool_failed_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id="debug_tool_failed_completed",
                ),
            ],
        )
        await create(
            "debug_tool_blocked_completed",
            SessionStatus.COMPLETED,
            [
                Event(
                    id="debug_tool_blocked_event",
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id="debug_tool_blocked_completed",
                    tool_name="deploy_service",
                    payload={"reason": "policy denied"},
                ),
                Event(
                    id="debug_tool_blocked_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id="debug_tool_blocked_completed",
                ),
            ],
        )
        await create(
            "debug_failed_session",
            SessionStatus.FAILED,
            [
                Event(
                    id="debug_failed_terminal",
                    type=EventType.SESSION_FAILED,
                    session_id="debug_failed_session",
                    payload={"error": "provider failed", "error_type": "RuntimeError"},
                )
            ],
        )
        await create(
            "debug_interrupted_session",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="debug_interrupted_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="debug_interrupted_session",
                    payload={"interruption_type": "tool_approval_required"},
                )
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    tool_response = client.post(
        "/api/sessions/summary",
        params={
            "debug_state": "tool_issue",
            "order_by": "created_at_asc",
        },
    )
    assert tool_response.status_code == 200
    tool_body = tool_response.json()
    assert tool_body["session_count"] == 2
    assert tool_body["total_count"] == 2
    assert tool_body["next_cursor"] is None
    assert [item["session"]["id"] for item in tool_body["sessions"]] == [
        "debug_tool_failed_completed",
        "debug_tool_blocked_completed",
    ]
    assert tool_body["sessions"][0]["events"]["counts_by_type"]["tool.call.failed"] == 1
    assert tool_body["sessions"][1]["events"]["counts_by_type"]["tool.call.blocked"] == 1

    list_response = client.get(
        "/api/sessions",
        params={
            "debug_state": "tool_issue",
            "order_by": "created_at_asc",
        },
    )
    assert list_response.status_code == 200
    list_body = list_response.json()
    assert list_body["total_count"] == 2
    assert list_body["next_cursor"] is None
    assert [session["id"] for session in list_body["sessions"]] == [
        "debug_tool_failed_completed",
        "debug_tool_blocked_completed",
    ]

    failure_response = client.post(
        "/api/sessions/summary",
        params={"debug_state": "session_failure", "order_by": "created_at_asc"},
    )
    assert failure_response.status_code == 200
    assert [item["session"]["id"] for item in failure_response.json()["sessions"]] == [
        "debug_failed_session"
    ]

    interruption_response = client.post(
        "/api/sessions/summary",
        params={"debug_state": "interruption", "order_by": "created_at_asc"},
    )
    assert interruption_response.status_code == 200
    assert [item["session"]["id"] for item in interruption_response.json()["sessions"]] == [
        "debug_interrupted_session"
    ]

    attention_response = client.post(
        "/api/sessions/summary",
        params={
            "debug_state": "needs_attention",
            "limit": 3,
            "order_by": "created_at_asc",
        },
    )
    assert attention_response.status_code == 200
    attention_body = attention_response.json()
    assert attention_body["session_count"] == 3
    assert attention_body["total_count"] == 4
    assert attention_body["next_cursor"] is not None
    assert [item["session"]["id"] for item in attention_body["sessions"]] == [
        "debug_tool_failed_completed",
        "debug_tool_blocked_completed",
        "debug_failed_session",
    ]

    next_attention_response = client.post(
        "/api/sessions/summary",
        params={
            "cursor": attention_body["next_cursor"],
            "debug_state": "needs_attention",
            "limit": 3,
            "order_by": "created_at_asc",
        },
    )
    assert next_attention_response.status_code == 200
    next_attention_body = next_attention_response.json()
    assert next_attention_body["session_count"] == 1
    assert next_attention_body["total_count"] == 4
    assert next_attention_body["next_cursor"] is None
    assert [item["session"]["id"] for item in next_attention_body["sessions"]] == [
        "debug_interrupted_session",
    ]

"""HTTP server session queries contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    OneShotProvider,
    UsageProvider,
    _aggregate_usage_json,
    _collect_run,
    _create_session,
    _lifecycle_store_and_client,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message, MessageRole, ProviderStatePart, TextPart, ThinkingPart
from cayu.runtime._event_projection import public_event_id
from cayu.server import create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest


def test_server_lists_sessions_with_label_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    assert {session["id"] for session in org_response.json()["sessions"]} == {
        "sess_invoice",
        "sess_research",
    }
    assert exact_response.status_code == 200
    assert [session["id"] for session in exact_response.json()["sessions"]] == ["sess_invoice"]
    assert exact_response.json()["sessions"][0]["labels"] == {
        "organization": "org_123",
        "project": "ap_q2",
    }
    assert missing_response.status_code == 200
    assert missing_response.json()["sessions"] == []


def test_server_lists_sessions_with_typed_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    assert [session["id"] for session in builder_response.json()["sessions"]] == [
        "sess_builder_local",
        "sess_builder_prod",
    ]
    assert completed_response.status_code == 200
    assert [session["id"] for session in completed_response.json()["sessions"]] == [
        "sess_builder_prod"
    ]
    assert env_response.status_code == 200
    assert [session["id"] for session in env_response.json()["sessions"]] == ["sess_builder_prod"]
    assert causal_response.status_code == 200
    assert [session["id"] for session in causal_response.json()["sessions"]] == [
        "sess_builder_prod"
    ]


def test_server_lists_sessions_with_typed_and_label_filters_together() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    assert [session["id"] for session in response.json()["sessions"]] == ["sess_builder_invoice"]


def test_server_lists_sessions_with_label_selectors() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    assert [session["id"] for session in exists_response.json()["sessions"]] == [
        "sess_selector_invoice"
    ]
    assert in_response.status_code == 200
    assert {session["id"] for session in in_response.json()["sessions"]} == {
        "sess_selector_invoice",
        "sess_selector_research",
        "sess_selector_unowned",
    }
    assert equals_response.status_code == 200
    assert {session["id"] for session in equals_response.json()["sessions"]} == {
        "sess_selector_invoice",
        "sess_selector_unowned",
    }
    assert not_in_response.status_code == 200
    assert [session["id"] for session in not_in_response.json()["sessions"]] == [
        "sess_selector_invoice"
    ]
    assert not_exists_response.status_code == 200
    assert [session["id"] for session in not_exists_response.json()["sessions"]] == [
        "sess_selector_unowned"
    ]


def test_server_session_label_filters_allow_reserved_query_keys() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/sessions?label=cayu:agent=builder")

    assert response.status_code == 200
    assert response.json()["sessions"] == []


def test_server_rejects_invalid_session_label_filters() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    assert client.get("/api/sessions?label=missing_separator").status_code == 422
    assert client.get("/api/sessions?label=%20=org_123").status_code == 422
    assert client.get("/api/sessions?label=owner=org_123&label=owner=org_456").status_code == 422
    assert client.get("/api/sessions?agent_name=%20").status_code == 422
    assert client.get("/api/sessions?status=not-a-status").status_code == 422
    assert client.get("/api/sessions?label_selector=project%20in%20ap_q2").status_code == 422


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/summary_1/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["session"]["id"] == "summary_1"
    assert body["session"]["status"] == "completed"
    assert body["session"]["agent_name"] == "assistant"
    assert body["session"]["provider_name"] == "fake"
    assert body["session"]["model"] == "fake-model"
    assert body["session"]["environment_name"] is None
    assert "interruption_cascade" not in body
    assert body["events"]["total_events"] == 10
    assert body["events"]["counts_by_type"] == {
        "interaction.completed": 1,
        "interaction.started": 1,
        "model.completed": 1,
        "model.started": 1,
        "model.text.delta": 1,
        "request.footprint.recorded": 1,
        "session.completed": 1,
        "session.started": 1,
        "tool.exposure.recorded": 1,
        "turn.completed": 1,
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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
    assert outcome["terminal_event"]["id"] == public_event_id(2)
    assert outcome["latest_retry_event"]["id"] == public_event_id(1)
    assert body["events"]["latest_event"]["id"] == public_event_id(3)


def test_server_session_summary_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/missing/summary")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_summary_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/%20/summary")

    assert response.status_code == 422


def test_server_exposes_bounded_session_state_without_heavy_loaders() -> None:
    app = CayuApp()

    async def seed() -> None:
        session = await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="state_1",
                messages=[Message.text("user", "hello")],
                metadata={"unbounded": "must not be loaded"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session.id, SessionStatus.RUNNING)
        await app.session_store.checkpoint(
            session.id,
            {"unrelated": {"large": "must not be loaded"}},
        )

    asyncio.run(seed())

    async def fail_heavy_read(*_args, **_kwargs):
        raise AssertionError("bounded state route must not use heavyweight loaders")

    app.session_store.load = fail_heavy_read  # type: ignore[method-assign]
    app.session_store.load_checkpoint = fail_heavy_read  # type: ignore[method-assign]
    app.get_session_usage = fail_heavy_read  # type: ignore[method-assign]

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/state_1/state")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "state_1"
    assert body["status"] == "running"
    assert body["interruption_cascade"] == "none"
    assert body["provider_operation"] == {
        "status": "synchronous",
        "stage_id": None,
        "run_epoch": None,
        "provider": None,
        "operation_id": None,
        "stream_protocol": None,
        "recovery_reason": None,
        "duplicate_request_risk": False,
        "allowed_resolutions": [],
        "resolution_action": None,
        "resolution_id": None,
        "cancellation_status": "not_requested",
        "accounting_status": "not_applicable",
        "reservation_count": 0,
    }
    assert body["updated_at"]
    assert body["last_activity_at"]
    assert set(body) == {
        "execution",
        "execution_snapshots",
        "session_id",
        "status",
        "updated_at",
        "last_activity_at",
        "interruption_cascade",
        "provider_operation",
    }


def test_server_session_state_exposes_provider_reconnect_in_progress() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="state_provider_operation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
        )
        model_identity = {
            "provider": "reconnectable",
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "max_attempts": 1,
            "model_step_id": "mstep_" + "a" * 32,
            "model_attempt_id": "matt_" + "b" * 32,
            "source_run_epoch": 1,
        }
        await app.session_store.append_events(
            "state_provider_operation",
            [
                Event(
                    type=EventType.MODEL_STARTED,
                    session_id="state_provider_operation",
                    interaction_id="interaction-a",
                    payload=model_identity,
                ),
                Event(
                    type=EventType.PROVIDER_OPERATION_STARTED,
                    session_id="state_provider_operation",
                    interaction_id="interaction-a",
                    payload={
                        **model_identity,
                        "provider": "reconnectable",
                        "start_id": "provider-operation:" + model_identity["model_attempt_id"],
                        "state_version": 1,
                        "operation_id": "response_123",
                        "stream_protocol": "responses-v1",
                        "status": "in_progress",
                        "recovery_metadata": {"cursor": 0},
                    },
                ),
                Event(
                    type=EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
                    session_id="state_provider_operation",
                    interaction_id="interaction-a",
                    payload={
                        **model_identity,
                        "provider": "reconnectable",
                        "operation_id": "response_123",
                        "stream_protocol": "responses-v1",
                        "status": "in_progress",
                    },
                ),
            ],
        )

    asyncio.run(seed())

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/state_provider_operation/state")

    assert response.status_code == 200
    assert response.json()["provider_operation"] == {
        "status": "reconnect_in_progress",
        "stage_id": None,
        "run_epoch": None,
        "provider": "reconnectable",
        "operation_id": "response_123",
        "stream_protocol": "responses-v1",
        "recovery_reason": None,
        "duplicate_request_risk": False,
        "allowed_resolutions": [],
        "resolution_action": None,
        "resolution_id": None,
        "cancellation_status": "not_requested",
        "accounting_status": "not_applicable",
        "reservation_count": 0,
    }


def test_server_session_state_keeps_ambiguous_start_and_model_error_visible() -> None:
    app = CayuApp()

    async def seed() -> None:
        for session_id, include_started in (
            ("state_provider_start_ambiguous", False),
            ("state_provider_error", True),
        ):
            await app.session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="reconnectable", model="fake-model"),
            )
            model_identity = {
                "provider": "reconnectable",
                "model": "fake-model",
                "step": 1,
                "attempt": 1,
                "max_attempts": 1,
                "model_step_id": "mstep_" + "a" * 32,
                "model_attempt_id": "matt_" + "b" * 32,
                "source_run_epoch": 1,
            }
            events = [
                Event(
                    type=EventType.MODEL_STARTED,
                    session_id=session_id,
                    interaction_id="interaction-a",
                    payload=model_identity,
                ),
                Event(
                    type=EventType.PROVIDER_OPERATION_STARTING,
                    session_id=session_id,
                    interaction_id="interaction-a",
                    payload={
                        **model_identity,
                        "provider": "reconnectable",
                        "start_id": "provider-operation:" + model_identity["model_attempt_id"],
                    },
                ),
            ]
            if include_started:
                events.extend(
                    [
                        Event(
                            type=EventType.PROVIDER_OPERATION_STARTED,
                            session_id=session_id,
                            interaction_id="interaction-a",
                            payload={
                                **model_identity,
                                "provider": "reconnectable",
                                "start_id": (
                                    "provider-operation:" + model_identity["model_attempt_id"]
                                ),
                                "state_version": 1,
                                "operation_id": "response_123",
                                "stream_protocol": "responses-v1",
                                "status": "in_progress",
                                "recovery_metadata": {"cursor": 0},
                            },
                        ),
                        Event(
                            type=EventType.MODEL_ERROR,
                            session_id=session_id,
                            interaction_id="interaction-a",
                            payload=model_identity,
                        ),
                    ]
                )
            await app.session_store.append_events(session_id, events)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    ambiguous = client.get("/api/sessions/state_provider_start_ambiguous/state")
    model_error = client.get("/api/sessions/state_provider_error/state")

    assert ambiguous.status_code == 200
    assert ambiguous.json()["provider_operation"] == {
        "status": "ambiguous_submission",
        "stage_id": None,
        "run_epoch": None,
        "provider": "reconnectable",
        "operation_id": None,
        "stream_protocol": None,
        "recovery_reason": "ambiguous_submission",
        "duplicate_request_risk": True,
        "allowed_resolutions": ["fallback_retry", "fail"],
        "resolution_action": None,
        "resolution_id": None,
        "cancellation_status": "not_requested",
        "accounting_status": "not_applicable",
        "reservation_count": 0,
    }
    assert model_error.status_code == 200
    assert model_error.json()["provider_operation"] == {
        "status": "provider_operation_in_progress",
        "stage_id": None,
        "run_epoch": None,
        "provider": "reconnectable",
        "operation_id": "response_123",
        "stream_protocol": "responses-v1",
        "recovery_reason": None,
        "duplicate_request_risk": False,
        "allowed_resolutions": [],
        "resolution_action": None,
        "resolution_id": None,
        "cancellation_status": "not_requested",
        "accounting_status": "not_applicable",
        "reservation_count": 0,
    }


def test_server_session_state_returns_404_and_validates_id() -> None:
    client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    assert client.get("/api/sessions/missing/state").status_code == 404
    assert client.get("/api/sessions/%20/state").status_code == 422


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

    async def fail_unbounded_session_load(*_args, **_kwargs):
        raise AssertionError("event pagination must use the bounded state projection")

    app.session_store.load = fail_unbounded_session_load  # type: ignore[method-assign]

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    first_page = client.get("/api/sessions/events_1/events?limit=2")
    assert first_page.status_code == 200
    first_body = first_page.json()
    assert first_body["session_id"] == "events_1"
    assert first_body["order_by"] == "sequence_asc"
    assert first_body["has_more"] is True
    assert first_body["next_sequence"] == 2
    assert first_body["scan_through_sequence"] == 2
    assert [event["id"] for event in first_body["events"]] == [
        public_event_id(1),
        public_event_id(2),
    ]
    assert first_body["events"][1] == {
        "sequence": 2,
        "id": public_event_id(2),
        "type": "tool.call.completed",
        "session_id": "events_1",
        "interaction_id": None,
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
    assert second_body["order_by"] == "sequence_asc"
    assert second_body["next_sequence"] == 3
    assert second_body["scan_through_sequence"] == 3
    assert [event["id"] for event in second_body["events"]] == [public_event_id(3)]

    latest_page = client.get("/api/sessions/events_1/events?order_by=sequence_desc&limit=2")
    assert latest_page.status_code == 200
    latest_body = latest_page.json()
    assert latest_body["order_by"] == "sequence_desc"
    assert latest_body["has_more"] is True
    assert latest_body["next_sequence"] == 2
    assert latest_body["scan_through_sequence"] == 3
    assert [event["id"] for event in latest_body["events"]] == [
        public_event_id(3),
        public_event_id(2),
    ]

    older_page = client.get(
        "/api/sessions/events_1/events?order_by=sequence_desc&before_sequence=2&limit=2"
    )
    assert older_page.status_code == 200
    older_body = older_page.json()
    assert older_body["order_by"] == "sequence_desc"
    assert older_body["has_more"] is False
    assert older_body["next_sequence"] == 1
    assert older_body["scan_through_sequence"] is None
    assert [event["id"] for event in older_body["events"]] == [public_event_id(1)]

    exhausted_page = client.get(
        "/api/sessions/events_1/events?order_by=sequence_desc&before_sequence=1&limit=2"
    )
    assert exhausted_page.status_code == 200
    assert exhausted_page.json() == {
        "session_id": "events_1",
        "events": [],
        "order_by": "sequence_desc",
        "next_sequence": 1,
        "scan_through_sequence": None,
        "has_more": False,
    }


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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
    assert body["scan_through_sequence"] == 3
    assert [event["id"] for event in body["events"]] == [public_event_id(1)]

    bounded_response = client.get(
        "/api/sessions/events_filters/events",
        params={"event_type": "tool.call.completed", "limit": 1},
    )
    assert bounded_response.status_code == 200
    bounded_body = bounded_response.json()
    assert bounded_body["has_more"] is True
    assert bounded_body["scan_through_sequence"] == 1
    assert [event["id"] for event in bounded_body["events"]] == [public_event_id(1)]


def test_server_finds_exact_session_scoped_event_id() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    event_id = "shared/event?id=1"

    async def seed_events() -> None:
        for session_id, source in (
            ("event_lookup", "selected"),
            ("event_lookup_other", "other"),
        ):
            await app.session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await app.session_store.append_events(
                session_id,
                [
                    Event(
                        id=event_id,
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        payload={"source": source},
                    )
                ],
            )

    asyncio.run(seed_events())

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get(
        "/api/sessions/event_lookup/events",
        params={"event_id": event_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["next_sequence"] == 1
    assert body["scan_through_sequence"] == 1
    assert [(event["id"], event["session_id"], event["payload"]) for event in body["events"]] == [
        (public_event_id(1), "event_lookup", {"source": "selected"})
    ]

    missing_response = client.get(
        "/api/sessions/event_lookup/events",
        params={"event_id": "missing"},
    )
    assert missing_response.status_code == 200
    assert missing_response.json() == {
        "session_id": "event_lookup",
        "events": [],
        "order_by": "sequence_asc",
        "next_sequence": None,
        "scan_through_sequence": 1,
        "has_more": False,
    }


def test_server_excludes_event_type_before_pagination() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_events() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="events_exclusion",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "events_exclusion",
            [
                Event(
                    id="useful_old",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="events_exclusion",
                ),
                *[
                    Event(
                        id=f"delta_{index}",
                        type=EventType.MODEL_TEXT_DELTA,
                        session_id="events_exclusion",
                        payload={"delta": "x"},
                    )
                    for index in range(20)
                ],
                Event(
                    id="useful_new",
                    type=EventType.MODEL_COMPLETED,
                    session_id="events_exclusion",
                ),
                *[
                    Event(
                        id=f"trailing_delta_{index}",
                        type=EventType.MODEL_TEXT_DELTA,
                        session_id="events_exclusion",
                        payload={"delta": "y"},
                    )
                    for index in range(10)
                ],
            ],
        )

    asyncio.run(seed_events())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get(
        "/api/sessions/events_exclusion/events",
        params={
            "exclude_event_type": "model.text.delta",
            "order_by": "sequence_desc",
            "limit": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert [event["id"] for event in body["events"]] == [
        public_event_id(22),
        public_event_id(1),
    ]
    assert body["scan_through_sequence"] == 32

    useful_new_sequence = body["events"][0]["sequence"]
    forward_response = client.get(
        "/api/sessions/events_exclusion/events",
        params={
            "exclude_event_type": "model.text.delta",
            "after_sequence": useful_new_sequence,
            "order_by": "sequence_asc",
            "limit": 2,
        },
    )
    assert forward_response.status_code == 200
    forward_body = forward_response.json()
    assert forward_body["events"] == []
    assert forward_body["next_sequence"] == useful_new_sequence
    assert forward_body["scan_through_sequence"] == 32

    caught_up_response = client.get(
        "/api/sessions/events_exclusion/events",
        params={
            "exclude_event_type": "model.text.delta",
            "after_sequence": forward_body["scan_through_sequence"],
            "order_by": "sequence_asc",
            "limit": 2,
        },
    )
    assert caught_up_response.status_code == 200
    caught_up_body = caught_up_response.json()
    assert caught_up_body["events"] == []
    assert caught_up_body["next_sequence"] == 32
    assert caught_up_body["scan_through_sequence"] == 32


def test_server_session_events_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    assert client.get("/api/sessions/events_validation/events?limit=0").status_code == 422
    assert (
        client.get(
            "/api/sessions/events_validation/events?after_sequence=2&before_sequence=2"
        ).status_code
        == 422
    )
    assert (
        client.get("/api/sessions/events_validation/events?order_by=not_valid").status_code == 422
    )
    assert (
        client.get("/api/sessions/events_validation/events?event_type=not.valid").status_code == 422
    )
    assert client.get("/api/sessions/events_validation/events?event_id=%20").status_code == 422
    for malformed_alias in (
        f"cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}",
        f"cayu_event_{'9' * 5000}",
    ):
        response = client.get(
            "/api/sessions/events_validation/events",
            params={"event_id": malformed_alias},
        )
        assert response.status_code == 422
        assert "malformed Cayu public event alias" in response.json()["detail"]
    assert (
        client.get(
            "/api/sessions/events_validation/events?exclude_event_type=not.valid"
        ).status_code
        == 422
    )
    assert client.get("/api/sessions/%20/events").status_code == 422


@pytest.mark.parametrize(
    ("arguments_state", "arguments"),
    [("finalized", {"path": "notes/result.txt"}), ("unavailable", {})],
)
def test_server_exposes_paginated_session_transcript(arguments_state, arguments) -> None:
    from cayu.messages import ToolCallPart

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
                    calls=[
                        ToolCallPart(
                            tool_call_id="call_1",
                            tool_name="read_file",
                            arguments=arguments,
                            arguments_state=arguments_state,
                        )
                    ]
                ),
            ],
            interaction_id="interaction_1",
        )
        await app.session_store.append_transcript_messages(
            "transcript_1",
            [
                Message.tool_result(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    content="file contents",
                    model_step_id=f"mstep_{'1' * 32}",
                    model_attempt_id=f"matt_{'2' * 32}",
                    tool_round_id=f"tround_{'3' * 32}",
                ),
                Message.text("assistant", "done"),
            ],
            interaction_id="interaction_2",
        )

    asyncio.run(seed_transcript())

    async def fail_unbounded_session_load(*_args, **_kwargs):
        raise AssertionError("transcript pagination must use the bounded state projection")

    app.session_store.load = fail_unbounded_session_load  # type: ignore[method-assign]

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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
    assert [message["interaction_id"] for message in first_body["messages"]] == [
        "interaction_1",
        "interaction_1",
    ]
    assert first_body["messages"][1]["content"] == [
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "read_file",
            "arguments": arguments,
            "arguments_state": arguments_state,
        }
    ]

    second_page = client.get("/api/sessions/transcript_1/transcript?offset=2&limit=2")
    assert second_page.status_code == 200
    second_body = second_page.json()
    assert second_body["next_offset"] == 4
    assert second_body["has_more"] is False
    assert [message["role"] for message in second_body["messages"]] == ["tool", "assistant"]
    assert [message["interaction_id"] for message in second_body["messages"]] == [
        "interaction_2",
        "interaction_2",
    ]
    result_content = second_body["messages"][0]["content"][0]
    assert result_content["model_step_id"] == f"mstep_{'1' * 32}"
    assert result_content["model_attempt_id"] == f"matt_{'2' * 32}"
    assert result_content["tool_round_id"] == f"tround_{'3' * 32}"

    interaction_page = client.get(
        "/api/sessions/transcript_1/transcript?interaction_id=interaction_2"
    )
    assert interaction_page.status_code == 200
    assert [message["index"] for message in interaction_page.json()["messages"]] == [2, 3]


def test_server_exposes_response_scoped_interaction_summaries() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed_interactions() -> None:
        await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="interaction_summary_1",
                messages=[Message.text("user", "first")],
            ),
        )
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="interaction_summary_1",
                    messages=[Message.text("user", "second")],
                )
            )
        ]
        await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="interaction_summary_other",
                messages=[Message.text("user", "other")],
            ),
        )

    asyncio.run(seed_interactions())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/sessions/interaction_summary_1/interactions?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "interaction_summary_1"
    assert body["has_more"] is False
    assert len(body["interactions"]) == 2
    assert [item["status"] for item in body["interactions"]] == [
        "completed",
        "completed",
    ]
    first = body["interactions"][1]
    second = body["interactions"][0]
    assert first["interaction_id"] != second["interaction_id"]
    assert first["source_transcript_start"] == 0
    assert first["result_transcript_end"] == 1
    assert second["source_transcript_start"] == 2
    assert second["result_transcript_end"] == 3
    assert first["model_step_count"] == 1
    assert first["tool_call_count"] == 0
    assert first["active_duration_ms"] >= 0

    newest_page = client.get("/api/sessions/interaction_summary_1/interactions?limit=1").json()
    assert newest_page["has_more"] is True
    older_page = client.get(
        "/api/sessions/interaction_summary_1/interactions",
        params={"limit": 1, "before_sequence": newest_page["next_sequence"]},
    ).json()
    assert older_page["has_more"] is False
    assert [
        newest_page["interactions"][0]["interaction_id"],
        older_page["interactions"][0]["interaction_id"],
    ] == [second["interaction_id"], first["interaction_id"]]

    detail = client.get(
        f"/api/sessions/interaction_summary_1/interactions/{first['interaction_id']}"
    )
    assert detail.status_code == 200
    assert detail.json() == first
    events = client.get(
        f"/api/sessions/interaction_summary_1/events?interaction_id={first['interaction_id']}"
    ).json()["events"]
    assert events
    assert {event["interaction_id"] for event in events} == {first["interaction_id"]}
    transcript = client.get(
        f"/api/sessions/interaction_summary_1/transcript?interaction_id={first['interaction_id']}"
    ).json()["messages"]
    assert [message["index"] for message in transcript] == [0, 1]
    assert (
        client.get(
            f"/api/sessions/interaction_summary_other/interactions/{first['interaction_id']}"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/sessions/interaction_summary_1/events",
            params={"interaction_id": " "},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/sessions/interaction_summary_1/transcript",
            params={"interaction_id": " "},
        ).status_code
        == 422
    )


def test_server_rejects_interaction_completion_before_start() -> None:
    started_at = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_interaction_time",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await store.append_event(
            "sess_invalid_interaction_time",
            Event(
                id="interaction-invalid-completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id="sess_invalid_interaction_time",
                interaction_id="interaction-invalid-time",
                timestamp=started_at,
                payload={
                    "status": "completed",
                    "start_event_id": "interaction-invalid-started",
                    "started_at": started_at.isoformat(),
                    "completed_at": (started_at - timedelta(microseconds=1)).isoformat(),
                },
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with pytest.raises(ValidationError, match="completed_at must not precede started_at"):
        client.get("/api/sessions/sess_invalid_interaction_time/interactions?limit=10")


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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    assert client.get("/api/sessions/transcript_validation/transcript?limit=0").status_code == 422
    assert (
        client.get("/api/sessions/transcript_validation/transcript?role=invalid").status_code == 422
    )
    assert client.get("/api/sessions/%20/transcript").status_code == 422


def test_server_lists_sessions_with_cursor_pagination() -> None:
    async def seed(store):
        for index in range(3):
            await _create_session(store, f"sess_{index}")

    _, client = _lifecycle_store_and_client(seed)

    page1 = client.get("/api/sessions?limit=2&order_by=created_at_asc").json()
    assert [session["id"] for session in page1["sessions"]] == ["sess_0", "sess_1"]
    assert page1["total_count"] == 3
    assert page1["next_cursor"] is not None

    page2 = client.get(
        "/api/sessions",
        params={"limit": 2, "order_by": "created_at_asc", "cursor": page1["next_cursor"]},
    ).json()
    assert [session["id"] for session in page2["sessions"]] == ["sess_2"]
    assert page2["total_count"] == 3
    assert page2["next_cursor"] is None


def test_server_lists_sessions_rejects_invalid_cursor() -> None:
    async def seed(store):
        await _create_session(store, "sess_only")

    _, client = _lifecycle_store_and_client(seed)

    response = client.get("/api/sessions", params={"cursor": "!!!not-a-cursor"})
    assert response.status_code == 422


def test_server_list_omits_metadata_but_detail_includes_it() -> None:
    async def seed(store):
        await _create_session(store, "sess_m", metadata={"secret": "value"})

    _, client = _lifecycle_store_and_client(seed)

    # The list view omits the (unbounded) per-session metadata...
    listed = client.get("/api/sessions").json()["sessions"]
    assert [row["id"] for row in listed] == ["sess_m"]
    assert "metadata" not in listed[0]
    assert "labels" in listed[0]  # base fields still present
    # ...but the single-session detail view includes it.
    detail = client.get("/api/sessions/sess_m").json()
    assert detail["metadata"] == {"secret": "value"}
    assert "events" not in detail
    assert "transcript" not in detail
    assert "interruption_cascade" not in detail


def test_server_session_detail_does_not_read_history_or_checkpoint_state() -> None:
    async def seed(store):
        session = await _create_session(store, "sess_bounded_detail", metadata={"kind": "demo"})
        await store.append_event(
            session.id,
            Event(
                type=EventType.SESSION_STARTED,
                session_id=session.id,
                agent_name=session.agent_name,
                payload={},
            ),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("assistant", "response")],
        )

    store, client = _lifecycle_store_and_client(seed)

    with (
        patch.object(store, "load_events", wraps=store.load_events) as load_events,
        patch.object(store, "load_transcript", wraps=store.load_transcript) as load_transcript,
        patch.object(store, "query_events", wraps=store.query_events) as query_events,
        patch.object(store, "query_transcript", wraps=store.query_transcript) as query_transcript,
        patch.object(
            store,
            "load_interruption_cascade_marker",
            wraps=store.load_interruption_cascade_marker,
        ) as load_interruption_cascade_marker,
    ):
        response = client.get("/api/sessions/sess_bounded_detail")

    assert response.status_code == 200
    assert response.json()["metadata"] == {"kind": "demo"}
    load_events.assert_not_awaited()
    load_transcript.assert_not_awaited()
    query_events.assert_not_awaited()
    query_transcript.assert_not_awaited()
    load_interruption_cascade_marker.assert_not_awaited()


def test_server_session_state_exposes_typed_interruption_cascade_state() -> None:
    async def seed(store):
        await _create_session(store, "sess_cascade_state")

    store, client = _lifecycle_store_and_client(seed)

    def assert_cascade_state(expected: str) -> None:
        assert (
            client.get("/api/sessions/sess_cascade_state/state").json()["interruption_cascade"]
            == expected
        )

    assert_cascade_state("none")

    async def set_marker(*, failed: bool) -> None:
        marker = {
            "attempt_id": "cascade-attempt",
            "interrupt_payload": {"interruption_type": "operator_requested"},
            "created_at": datetime.now(UTC).isoformat(),
        }
        if failed:
            marker["failure_recorded"] = True
        else:
            marker.update(
                {
                    "generation": 1,
                    "claim_id": "cascade-claim",
                    "claim_expires_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                }
            )
        await store.checkpoint(
            "sess_cascade_state",
            {"pending_interruption_cascade": marker},
        )

    asyncio.run(set_marker(failed=False))
    assert_cascade_state("pending")

    asyncio.run(set_marker(failed=True))
    assert_cascade_state("failed")

    async def set_malformed_active_marker() -> None:
        await store.checkpoint(
            "sess_cascade_state",
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt",
                    "interrupt_payload": {"interruption_type": "operator_requested"},
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation": "invalid",
                    "claim_id": "cascade-claim",
                    "claim_expires_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                }
            },
        )

    asyncio.run(set_malformed_active_marker())
    assert_cascade_state("failed")


def test_transcript_pagination_terminates_when_excluding_thinking() -> None:
    # Regression: with include_thinking=false the store drops thinking-only records from a
    # page, so the route must advance next_offset by the window size (not the returned
    # record count) or pagination stalls on an empty page (reviewer's repro: thinking-only
    # first record + limit=1).
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    async def seed() -> None:
        session = await store.create(
            RunRequest(
                agent_name="a",
                session_id="sess_think",
                messages=[Message.text("user", "q")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session.id,
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[ThinkingPart(text="reasoning", provider_state={"signature": "S"})],
                ),
                Message(role=MessageRole.ASSISTANT, content=[TextPart(text="answer")]),
            ],
        )

    asyncio.run(seed())

    offset = 0
    seen_offsets: list[int] = []
    collected: list[dict] = []
    for _ in range(10):  # cap guards against the infinite loop the bug caused
        assert offset not in seen_offsets, "pagination revisited an offset (loop)"
        seen_offsets.append(offset)
        body = client.get(
            "/api/sessions/sess_think/transcript",
            params={"include_thinking": "false", "limit": 1, "offset": offset},
        ).json()
        collected.extend(body["messages"])
        if not body["has_more"]:
            break
        assert body["next_offset"] > offset  # must advance even on an empty (filtered) page
        offset = body["next_offset"]
    else:
        raise AssertionError("pagination did not terminate")

    parts = [part for message in collected for part in message["content"]]
    assert any(part["type"] == "text" for part in parts)  # the answer survives
    assert all(part["type"] != "thinking" for part in parts)  # thinking excluded


def test_transcript_api_omits_opaque_provider_state_payload() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="opaque_provider_state",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            "opaque_provider_state",
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        ProviderStatePart(
                            provider="chat_completions",
                            state={
                                "type": "reasoning_details",
                                "details": [
                                    {
                                        "data": "encrypted-provider-canary",
                                        "signature": "signed-provider-canary",
                                    }
                                ],
                            },
                        )
                    ],
                )
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/sessions/opaque_provider_state/transcript")

    assert response.status_code == 200
    [message] = response.json()["messages"]
    assert message["content"] == [{"type": "provider_state", "provider": "chat_completions"}]
    assert "encrypted-provider-canary" not in response.text
    assert "signed-provider-canary" not in response.text

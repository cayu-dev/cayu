"""HTTP server contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio
import base64
import json
import logging
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    OneShotProvider,
    UsageProvider,
    _create_session,
    _lifecycle_store_and_client,
    _price_book_payload,
    _sse_events,
    _sse_frames,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import InMemoryBudgetStore
from cayu.configuration import DEFAULT_MAX_STEPS, MAX_STEPS
from cayu.context.base import CheckpointCompactionContextPolicy, TranscriptDigestCompactor
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import (
    Event,
    EventType,
    event_with_durable_sequence,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import TerminalEventPublicationUncertain
from cayu.messages import Message
from cayu.observability.events import InMemoryEventSink
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._continuation_task_failure import runtime_task_failure_identity_from_task
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    REDACTED_CUSTOM_EVENT_TYPE,
    public_event_envelope_alias_field,
    public_event_id,
    public_event_linkage_id,
    public_event_sequence,
)
from cayu.runtime.provider_operations import PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import (
    DashboardConfig,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    ServerLifecycleConfig,
    create_server,
    mount_cayu,
    mount_dashboard,
)
from cayu.server.routes import (
    _accepted_event_stream_response,
    _detached_event_stream_response,
    _log_mutation_acceptance_failure,
    _next_replay_poll_interval,
    _start_detached_event_stream_response,
)
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SSE_OBSERVER_MAX_BYTES,
    SSE_OBSERVER_MAX_FRAMES,
    SSE_REPLAY_PAGE_EVENTS,
    SSE_SEND_TIMEOUT_SECONDS,
)
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, SessionIdentity, SessionStatus
from cayu.sessions.recovery import RecoveryExecutionRequest, RecoveryPlan
from cayu.sessions.requests import (
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    run_request_with_runtime_generated_authority,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import Task, TaskStatus
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

_SHORT_REPLAY_SERVER_CONFIG = ServerConfig.local_development(
    lifecycle=ServerLifecycleConfig(replay_idle_timeout_s=0.01)
)


class UnavailableTargetProvider(OneShotProvider):
    def __init__(self) -> None:
        self.dispatches = 0

    def preflight_model_target(self, *, model: str) -> None:
        raise RuntimeError(f"model target {model!r} is not configured")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.dispatches += 1
        async for event in super().stream(request):
            yield event


class FailOnceEventSink(InMemoryEventSink):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    async def emit(self, event: Event) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("sink temporarily unavailable")
        await super().emit(event)


class NeverReturningEventSink(InMemoryEventSink):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def emit(self, event: Event) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled = True


def test_server_uses_explicit_non_assistant_agent_for_runs_and_task_list() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    client = TestClient(
        create_server(
            app,
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(
                    runtime_config={
                        "apiBaseUrl": "/ignored",
                        "priceBook": _price_book_payload(output_per_million="3"),
                    }
                )
            ),
        )
    )

    assert client.get("/").status_code == 404

    dashboard = client.get("/cayu/")
    assert dashboard.status_code == 200
    assert "root" in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/api"' in dashboard.text
    assert '"priceBook":{"price_book_version":"test"' in dashboard.text

    with client.stream(
        "POST",
        "/api/run",
        json={"agent": "reviewer", "prompt": "hello"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 1
    assert tasks[0]["type"] == "run"
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["assigned_agent_name"] == "reviewer"
    assert tasks[0]["worker_id"] is None
    assert tasks[0]["lease_expires_at"] is None


@pytest.mark.parametrize("secret_start", [62, 63, 64, 79, 80])
def test_server_redacts_complete_prompt_before_task_title_bound(secret_start: int) -> None:
    secret = "task-title-boundary-secret"
    prompt = "p" * secret_start + secret + ":tail"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream("POST", "/api/run", json={"prompt": prompt}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    task = asyncio.run(task_store.list_tasks())[0]
    expected_prompt = "p" * secret_start + REDACTED_SECRET + ":tail"
    assert task.input == {"prompt": expected_prompt}
    assert task.title is not None
    assert len(task.title) <= 80
    assert secret not in task.title
    assert secret[:10] not in task.title
    assert not any(
        task.title.endswith(REDACTED_SECRET[:length]) for length in range(1, len(REDACTED_SECRET))
    )


def test_server_run_rejection_before_session_creates_no_task() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    # The runtime is advanced through its atomic session claim before the route
    # creates a task. A rejected command therefore has neither resource to clean up.
    response = client.post(
        "/api/run",
        json={
            "prompt": "hello",
            "structured_output": {
                "json_schema": {"type": "object"},
                "strategy": "native",
            },
        },
    )
    assert response.status_code == 409
    assert "Native structured output" in response.json()["detail"]

    assert client.get("/api/tasks").json() == []
    assert client.get("/api/sessions").json()["sessions"] == []

    response = client.post("/api/run", json={"prompt": "x", "agent": "ghost"})
    assert response.status_code == 404

    assert client.get("/api/tasks").json() == []
    assert client.get("/api/sessions").json()["sessions"] == []


def test_server_run_provider_target_preflight_precedes_session_and_task_mutation() -> None:
    provider = UnavailableTargetProvider()
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="missing-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/run", json={"prompt": "must not dispatch"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Mutation failed before streaming began."}
    assert provider.dispatches == 0
    assert client.get("/api/tasks").json() == []
    assert client.get("/api/sessions").json()["sessions"] == []


def test_server_run_failure_before_acceptance_is_generic_and_creates_no_task(caplog) -> None:
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor("secret-token"),
    )

    async def broken_run(request):
        raise OSError("storage failed with secret-token")
        yield  # pragma: no cover

    app.run = broken_run
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with caplog.at_level(logging.ERROR, logger="cayu.server.routes"):
        response = client.post("/api/run", json={"prompt": "hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Mutation failed before streaming began."}
    assert "secret-token" not in response.text
    assert "secret-token" not in caplog.text
    assert REDACTED_SECRET in caplog.text
    assert "stage=before_first_event" in caplog.text
    assert "error_type=OSError" in caplog.text
    assert client.get("/api/tasks").json() == []
    assert client.get("/api/sessions").json()["sessions"] == []


def test_mutation_acceptance_log_projects_private_session_identity(caplog) -> None:
    private_session_id = "private-session-secret"
    app = CayuApp(
        secret_redactor=SecretRedactor(private_session_id),
        enable_logging=False,
    )

    with caplog.at_level(logging.ERROR, logger="cayu.server.routes"):
        _log_mutation_acceptance_failure(
            app,
            RuntimeError("acceptance failed"),
            session_id=private_session_id,
            stage="after_first_event",
        )

    assert private_session_id not in caplog.text
    assert app.project_session_id_for_exposure(private_session_id) in caplog.text


def test_server_run_task_setup_failure_finalizes_claimed_session(caplog) -> None:
    class FailingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_running_task(self, request, *, session_invocation):
            raise OSError("task store unavailable with secret-token")

    app = CayuApp(
        task_store=FailingTaskStore(),
        secret_redactor=SecretRedactor("secret-token"),
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session_task_setup_failure"

    with caplog.at_level(logging.ERROR, logger="cayu.server.routes"):
        response = client.post(
            "/api/run",
            json={"prompt": "hello", "session_id": session_id},
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Mutation setup failed after its durable acceptance event."
    }
    assert "secret-token" not in caplog.text
    assert REDACTED_SECRET in caplog.text
    assert "stage=after_first_event" in caplog.text
    assert "error_type=OSError" in caplog.text
    state = asyncio.run(app.session_store.load_state(session_id))
    assert state is not None
    assert state.status is SessionStatus.INTERRUPTED


def test_server_run_terminal_prefix_does_not_create_a_running_task() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def terminal_run(request):
        session = await app.session_store.create(
            request,
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session.id, SessionStatus.FAILED)
        turn_completed = Event(
            id="event_terminal_prefix",
            type=EventType.TURN_COMPLETED,
            session_id=session.id,
            agent_name=request.agent_name,
        )
        await app.session_store.append_event(session.id, turn_completed)
        yield turn_completed
        session_failed = Event(
            id="event_terminal_failure",
            type=EventType.SESSION_FAILED,
            session_id=session.id,
            agent_name=request.agent_name,
        )
        await app.session_store.append_event(session.id, session_failed)
        yield session_failed

    app.run = terminal_run
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session_terminal_prefix"},
    )

    assert response.status_code == 200
    assert client.get("/api/tasks").json() == []


def test_server_run_environment_factory_failure_terminalizes_linked_task() -> None:
    class FailingEnvironmentFactory(EnvironmentFactory):
        async def create(
            self,
            _request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            raise RuntimeError("factory failed")

    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        FailingEnvironmentFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session_factory_failure"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        events = [frame["data"] for frame in _sse_frames(response) if "data" in frame]

    assert [event["type"] for event in events] == [
        EventType.INTERACTION_STARTED,
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.TASK_STARTED,
        EventType.ENVIRONMENT_LIFECYCLE_TRANSITION,
        EventType.ENVIRONMENT_LIFECYCLE_TRANSITION,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.TASK_FAILED,
        EventType.INTERACTION_FAILED,
        EventType.SESSION_FAILED,
    ]
    tasks = asyncio.run(task_store.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.FAILED
    assert tasks[0].session_id == session_id
    assert tasks[0].error == {
        "message": "factory failed",
        "type": "RuntimeError",
        "session_id": session_id,
    }


def test_server_run_binding_failure_terminalizes_prestarted_task() -> None:
    class FailingWorkspaceBinding(WorkspaceBinding):
        async def bind(
            self,
            workspace,
            runner,
            *,
            session_id: str,
            agent_name: str | None = None,
            environment_name: str | None = None,
            metadata: dict[str, Any] | None = None,
        ):
            raise RuntimeError("binding failed")

        async def finalize(
            self,
            bound,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ):
            raise AssertionError("A failed binding must not be finalized.")

    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="bound"),
            binding=FailingWorkspaceBinding(),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session_binding_failure"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        events = [frame["data"] for frame in _sse_frames(response) if "data" in frame]

    assert [event["type"] for event in events] == [
        EventType.INTERACTION_STARTED,
        EventType.ENVIRONMENT_LIFECYCLE_TRANSITION,
        EventType.ENVIRONMENT_LIFECYCLE_TRANSITION,
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.TASK_STARTED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.TASK_FAILED,
        EventType.INTERACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    tasks = asyncio.run(task_store.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.FAILED
    assert tasks[0].session_id == session_id
    assert tasks[0].error is not None
    task_error = dict(tasks[0].error)
    task_error.pop("runtime_task_failure")
    assert task_error == {
        "message": "binding failed",
        "type": "RuntimeError",
        "session_id": session_id,
    }
    session = asyncio.run(app.session_store.load(session_id))
    assert session is not None
    failure_identity = runtime_task_failure_identity_from_task(
        tasks[0],
        session_id=session.id,
        session_instance_id=session.instance_id,
    )
    assert failure_identity is not None
    assert events[-1]["payload"]["runtime_task_failure_id"] == failure_identity.failure_id


def test_run_threads_inbound_traceparent_into_session_metadata() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    traceparent = "00-11111111111111111111111111111111-2222222222222222-01"
    started = None
    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello"},
        headers={"traceparent": traceparent},
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :].strip())
            if event["type"] == "session.started":
                started = event

    assert started is not None
    assert started["payload"]["traceparent"] == traceparent


def _session_started_event(client: TestClient, path: str, body: dict, headers: dict) -> dict:
    started = None
    with client.stream("POST", path, json=body, headers=headers) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :].strip())
            if event["type"] in ("session.started", "session.resumed"):
                started = event
    assert started is not None
    return started


def test_resume_threads_inbound_traceparent_into_session_metadata() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    started = _session_started_event(client, "/api/run", {"prompt": "hello"}, {})
    session_id = started["session_id"]

    traceparent = "00-44444444444444444444444444444444-5555555555555555-01"
    resumed = _session_started_event(
        client,
        "/api/resume",
        {"session_id": session_id, "prompt": "again"},
        {"traceparent": traceparent},
    )
    assert resumed["type"] == "session.resumed"
    assert resumed["payload"]["traceparent"] == traceparent


def test_open_server_resume_constructs_explicit_profile_adoption_intent() -> None:
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    started = _session_started_event(client, "/api/run", {"prompt": "hello"}, {})

    captured: list[ResumeRequest] = []
    original_resume = app.resume

    def capture(request: ResumeRequest):
        captured.append(request)
        return original_resume(request)

    app.resume = capture  # type: ignore[method-assign]
    adoption_body = {
        "session_id": started["session_id"],
        "prompt": "again",
        "profile_adoption": {
            "idempotency_key": "server-adoption-v1",
            "reason": "Explicit operator adoption.",
            "requested_by": {
                "subject": "operator",
                "source": "request",
                "claims": {"role": "maintainer"},
            },
        },
    }
    first_traceparent = "00-11111111111111111111111111111111-2222222222222222-01"
    with client.stream(
        "POST",
        "/api/resume",
        json=adoption_body,
        headers={"traceparent": first_traceparent},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert len(captured) == 1
    adoption = captured[0].profile_adoption
    assert adoption is not None
    assert adoption.idempotency_key == "server-adoption-v1"
    assert adoption.reason == "Explicit operator adoption."
    assert adoption.requested_by.subject == "operator"
    assert adoption.requested_by.source.value == "request"
    assert captured[0].metadata == {}

    retry_traceparent = "00-33333333333333333333333333333333-4444444444444444-01"
    with client.stream(
        "POST",
        "/api/resume",
        json=adoption_body,
        headers={"traceparent": retry_traceparent},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    assert len(captured) == 2
    assert captured[1].metadata == {}

    response = client.post(
        "/api/resume",
        json={
            "session_id": started["session_id"],
            "prompt": "invalid adoption",
            "profile_adoption": {
                "idempotency_key": "server-adoption-without-actor",
                "reason": "Missing open-server provenance.",
            },
        },
    )
    assert response.status_code == 400
    assert "requested_by is required" in response.json()["detail"]
    assert len(captured) == 2


def test_server_run_accepts_budget_limits() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={
            "prompt": "hello",
            "budget_limits": [
                {
                    "scope": "session",
                    "max_estimated_cost": "0.000001",
                    "pricing": _price_book_payload(),
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["status"] == "interrupted"
    provenance = sessions[0]["runtime_build_provenance"]
    assert provenance["availability"] == "available"
    assert provenance["origin"] == "development_source_tree"
    assert len(provenance["fingerprint"]) == 64


def test_server_run_defaults_and_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    captured: list[int] = []
    captured_explicitness: list[bool] = []
    captured_targets: list[ModelTarget | None] = []
    original_run = app.run

    def spy_run(request: RunRequest):
        captured.append(request.max_steps)
        captured_explicitness.append("max_steps" in request.model_fields_set)
        captured_targets.append(request.target)
        return original_run(request)

    app.run = spy_run  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    with client.stream(
        "POST",
        "/api/run",
        json={
            "prompt": "hello",
            "max_steps": 7,
            "target": {
                "provider_name": "fake",
                "model": "request-model",
            },
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured == [DEFAULT_MAX_STEPS, 7]
    assert captured_explicitness == [False, True]
    assert captured_targets == [
        None,
        ModelTarget(provider_name="fake", model="request-model"),
    ]


@pytest.mark.parametrize("max_steps", [42, 257, 10000, MAX_STEPS])
def test_server_resume_overrides_max_steps(max_steps) -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    started = _session_started_event(
        client,
        "/api/run",
        {"prompt": "hello", "max_steps": max_steps},
        {},
    )
    session_id = started["session_id"]

    captured: list[int] = []
    original_resume = app.resume

    def spy_resume(request):
        captured.append(request.max_steps)
        return original_resume(request)

    app.resume = spy_resume  # type: ignore[method-assign]

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "again", "max_steps": max_steps},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured == [max_steps]


@pytest.mark.parametrize("path", ["/api/run", "/api/resume"])
@pytest.mark.parametrize("bad_value", [0, MAX_STEPS + 1, -1])
def test_server_rejects_out_of_range_max_steps(path: str, bad_value: int) -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    body: dict = {"prompt": "hello", "max_steps": bad_value}
    if path == "/api/resume":
        body["session_id"] = "session-does-not-matter"
    response = client.post(path, json=body)
    assert response.status_code == 422


def test_server_run_rejects_request_budget_reservations() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={
            "prompt": "hello",
            "budget_limits": [
                {
                    "scope": "session",
                    "max_estimated_cost": "0.01",
                    "pricing": _price_book_payload(),
                    "reservation": {
                        "max_input_tokens": 1,
                        "max_output_tokens": 0,
                    },
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid control-plane request."}
    assert response.headers["cache-control"] == "private, no-store"


def test_dashboard_routes_fall_back_to_index_without_masking_api_or_assets() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    for path in ["/cayu/sessions", "/cayu/run", "/cayu/sessions/session-abc"]:
        response = client.get(path)
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        assert '<base href="/cayu/" />' in response.text
        assert '"basePath":"/cayu"' in response.text

    assert client.get("/sessions").status_code == 404
    assert client.get("/api/missing").status_code == 404
    assert client.get("/cayu/assets/missing.js").status_code == 404


def test_dashboard_uses_effective_paths_when_server_is_nested_under_asgi_mount() -> None:
    parent = FastAPI()
    parent.mount("/product", create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    client = TestClient(parent)
    response = client.get("/product/cayu/sessions/deep-link")

    assert response.status_code == 200
    assert '<base href="/product/cayu/" />' in response.text
    assert '"basePath":"/product/cayu"' in response.text
    assert '"apiBaseUrl":"/product/api"' in response.text

    asset_paths = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', response.text)
    assert asset_paths
    for asset_path in asset_paths:
        assert client.get(f"/product/cayu/{asset_path}").status_code == 200
        assert client.get(f"/cayu/{asset_path}").status_code == 404


def test_dashboard_serves_lazy_route_chunks_under_nested_asgi_mount() -> None:
    parent = FastAPI()
    parent.mount("/product", create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    client = TestClient(parent)
    shell = client.get("/product/cayu/sessions/deep-link")

    assert shell.status_code == 200
    entry_scripts = re.findall(r'src="\./(assets/[^"]+\.js)"', shell.text)
    assert len(entry_scripts) == 1

    pending_assets = list(entry_scripts)
    visited_assets: set[str] = set()
    entry_chunks: set[str] | None = None

    while pending_assets:
        asset_path = pending_assets.pop()
        if asset_path in visited_assets:
            continue
        visited_assets.add(asset_path)

        response = client.get(f"/product/cayu/{asset_path}")
        assert response.status_code == 200, asset_path
        assert "javascript" in response.headers["content-type"], asset_path

        chunks = set(re.findall(r'["`]\./([^"`]+\.js)["`]', response.text))
        if asset_path == entry_scripts[0]:
            entry_chunks = chunks
        pending_assets.extend(f"assets/{chunk}" for chunk in chunks)

    assert entry_chunks is not None
    assert any(chunk.startswith("session-detail-") for chunk in entry_chunks)
    assert any(chunk.startswith("artifacts-") for chunk in entry_chunks)


def test_dashboard_path_can_be_disabled_or_customized() -> None:
    disabled = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False)),
        )
    )
    assert disabled.get("/cayu/").status_code == 404

    custom = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(dashboard=DashboardConfig(path="/inspector")),
        )
    )
    response = custom.get("/inspector/sessions")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert '<base href="/inspector/" />' in response.text
    assert '"basePath":"/inspector"' in response.text
    assert custom.get("/cayu/").status_code == 404


def test_create_server_can_embed_api_under_dashboard_path() -> None:
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                api=ServerApiConfig(path="/cayu/api"),
                dashboard=DashboardConfig(path="/cayu"),
            ),
        )
    )

    dashboard = client.get("/cayu/sessions")
    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '<base href="/cayu/" />' in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard.text

    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/api/health").status_code == 404
    assert client.get("/cayu/api/missing").status_code == 404


def test_mount_dashboard_helper_supports_composed_apps() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/inspector") is True

    client = TestClient(app)
    response = client.get("/inspector/knowledge")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert '<base href="/inspector/" />' in response.text
    assert '"basePath":"/inspector"' in response.text


def test_mount_dashboard_owns_slashless_path_before_host_fallback() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/internal/agents") is True

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(app)
    redirect = client.get(
        "/internal/agents?source=embed",
        follow_redirects=False,
    )

    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://testserver/internal/agents/?source=embed"

    head_redirect = client.head(
        "/internal/agents?source=probe",
        follow_redirects=False,
    )
    assert head_redirect.status_code == 307
    assert head_redirect.headers["location"] == ("http://testserver/internal/agents/?source=probe")

    dashboard = client.get(redirect.headers["location"])
    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '"basePath":"/internal/agents"' in dashboard.text
    assert '"apiBaseUrl":"/api"' in dashboard.text
    assert client.get("/elsewhere").json() == {"app": "host", "path": "elsewhere"}


@pytest.mark.parametrize("dashboard_path", [None, "/missing"])
def test_mount_dashboard_does_not_claim_unavailable_slashless_path(
    tmp_path, dashboard_path
) -> None:
    app = FastAPI()
    dashboard_dir = tmp_path / "missing" if dashboard_path is not None else None

    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path=dashboard_path,
        )
        is False
    )

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    requested_path = "/missing" if dashboard_path is not None else "/cayu"
    response = TestClient(app).get(requested_path, follow_redirects=False)
    assert response.status_code == 200
    assert response.json()["app"] == "host"


def test_mount_dashboard_does_not_claim_directory_without_entrypoint(tmp_path) -> None:
    app = FastAPI()
    dashboard_dir = tmp_path / "empty-dashboard"
    dashboard_dir.mkdir()

    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path="/inspector",
        )
        is False
    )

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    response = TestClient(app).get("/inspector", follow_redirects=False)
    assert response.status_code == 200
    assert response.json() == {"app": "host", "path": "inspector"}


def test_mount_dashboard_non_directory_does_not_claim_slashless_path(
    tmp_path,
) -> None:
    app = FastAPI()
    invalid_dashboard = tmp_path / "dashboard.html"
    invalid_dashboard.write_text("not a directory")

    routes_before = list(app.routes)
    assert (
        mount_dashboard(
            app,
            dashboard_dir=invalid_dashboard,
            dashboard_path="/inspector",
        )
        is False
    )
    assert app.routes == routes_before

    @app.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    response = TestClient(app).get("/inspector", follow_redirects=False)
    assert response.status_code == 200
    assert response.json() == {"app": "host", "path": "inspector"}


def test_root_dashboard_mount_does_not_register_redirect() -> None:
    app = FastAPI()

    assert mount_dashboard(app, dashboard_path="/") is True

    response = TestClient(app).get("/", follow_redirects=False)
    assert response.status_code == 200
    assert '"basePath":"/"' in response.text


def test_slashless_dashboard_head_redirects_to_canonical_path() -> None:
    app = FastAPI()
    assert mount_dashboard(app, dashboard_path="/inspector") is True

    response = TestClient(app).head("/inspector", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://testserver/inspector/"


@pytest.mark.parametrize(
    "method",
    ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
def test_slashless_dashboard_redirect_is_limited_to_get_and_head(method: str) -> None:
    app = FastAPI()
    assert mount_dashboard(app, dashboard_path="/inspector") is True

    response = TestClient(app).request(
        method,
        "/inspector",
        follow_redirects=False,
    )

    assert response.status_code == 405


def test_mount_dashboard_injects_base_before_custom_shell_assets(tmp_path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text(
        '<!doctype html><!-- <head data-fake=">"> -->'
        '<html><HEAD data-theme="dark" data-label="a > b">'
        '<script src="./assets/app.js"></script></HEAD><body>custom</body></html>',
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("window.customDashboard = true", encoding="utf-8")

    app = FastAPI()
    assert (
        mount_dashboard(
            app,
            dashboard_dir=dashboard_dir,
            dashboard_path="/inspector",
        )
        is True
    )

    client = TestClient(app)
    response = client.get("/inspector/sessions/deep-link")

    assert response.status_code == 200
    assert '<base href="/inspector/" />' in response.text
    assert response.text.index("<base ") < response.text.index("./assets/app.js")
    assert client.get("/inspector/assets/app.js").status_code == 200


def test_mount_dashboard_preserves_doctype_for_custom_shell_without_head(tmp_path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><body>'
        '<script src="./assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("window.customDashboard = true", encoding="utf-8")

    app = FastAPI()
    assert mount_dashboard(app, dashboard_dir=dashboard_dir, dashboard_path="/inspector") is True

    response = TestClient(app).get("/inspector/sessions/deep-link")

    assert response.status_code == 200
    assert response.text.startswith("<!doctype html><html")
    assert '<head>\n    <base href="/inspector/" />' in response.text
    assert response.text.index("<!doctype html>") < response.text.index("<head>")
    assert response.text.index("<head>") < response.text.index("./assets/app.js")


def test_mount_cayu_mounts_api_and_dashboard_under_product_path() -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    mount_cayu(server, cayu_app, path="/cayu", access=OpenAccess())

    @server.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(server)
    redirect = client.get("/cayu", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://testserver/cayu/"

    head_redirect = client.head("/cayu", follow_redirects=False)
    assert head_redirect.status_code == 307
    assert head_redirect.headers["location"] == "http://testserver/cayu/"

    dashboard_root = client.get(redirect.headers["location"])
    assert dashboard_root.status_code == 200
    assert '"basePath":"/cayu"' in dashboard_root.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard_root.text

    dashboard = client.get("/cayu/knowledge")

    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '<base href="/cayu/" />' in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard.text
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/api/health").json() == {"app": "host", "path": "api/health"}
    assert client.get("/elsewhere").json() == {"app": "host", "path": "elsewhere"}
    assert client.get("/cayu/api/missing").status_code == 404


def test_mount_cayu_can_disable_dashboard_for_api_only_services() -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    @server.get("/{path:path}")
    async def host_fallback(path: str) -> dict[str, str]:
        return {"app": "host", "path": path}

    client = TestClient(server)
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/cayu", follow_redirects=False).json()["app"] == "host"


def test_mount_cayu_closes_the_app_under_one_shutdown_deadline() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    calls: list[tuple[str, float | None]] = []
    resume_calls = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        resume_calls.append(interrupting_inactive_for_seconds)
        return 0

    def drain(name, *, takes_s=0.0):
        async def record(*, timeout_s):
            calls.append((name, timeout_s))
            await asyncio.sleep(takes_s)
            return True

        return record

    async def close_runtime_timing():
        calls.append(("runtime_timing", None))

    # The first drain uses part of the deadline; later ones only get the rest.
    cayu_app.drain_background_interruptions = drain("drain_background_interruptions", takes_s=0.3)
    for name in (
        "drain_recovery_cleanups",
        "drain_provider_operation_cancellations",
        "drain_environment_cleanups",
        "close_idle_environment_resources",
        "drain_knowledge_publications",
        "drain_collaboration_requests",
        "drain_session_exports",
    ):
        setattr(cayu_app, name, drain(name))
    cayu_app.close_runtime_timing = close_runtime_timing
    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    mount_cayu(
        server,
        cayu_app,
        path="/cayu",
        dashboard=False,
        access=OpenAccess(),
        interruption_shutdown_grace_seconds=2.5,
        knowledge_publication_shutdown_grace_seconds=1.5,
    )

    with TestClient(server):
        pass

    names = [name for name, _ in calls]
    assert names[:5] == [
        "drain_background_interruptions",
        "drain_recovery_cleanups",
        "drain_provider_operation_cancellations",
        "drain_environment_cleanups",
        "close_idle_environment_resources",
    ]
    assert set(names[5:8]) == {
        "drain_knowledge_publications",
        "drain_collaboration_requests",
        "drain_session_exports",
    }
    # Timing delivery stops after the other shutdown owners have drained.
    assert names[8:] == ["runtime_timing"]
    # Every drain shares the one 4.0s deadline instead of a fresh grace each.
    assert all(0 < timeout <= 4.0 for _, timeout in calls if timeout is not None)
    assert all(timeout <= 3.7 for _, timeout in calls[1:] if timeout is not None)
    assert cayu_app.lifecycle_state == "closed"
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.settled and outcome.timeout_seconds == 4.0
    assert len(resume_calls) == 1
    assert type(resume_calls[0]) is int
    assert resume_calls[0] >= 0


@pytest.mark.parametrize("host", ["mount_cayu", "create_server"])
def test_a_stalled_model_policy_stop_is_bounded_by_the_shutdown_deadline(host, caplog) -> None:
    cayu_app = CayuApp()
    stops: list[str] = []

    async def stalled_stop_model_policy():
        stops.append("stop")
        await asyncio.sleep(15)

    cayu_app.stop_model_policy = stalled_stop_model_policy  # ty: ignore[invalid-assignment]
    lifecycle = ServerLifecycleConfig(
        interruption_shutdown_grace_seconds=0.2,
        knowledge_publication_shutdown_grace_seconds=0.2,
    )
    if host == "mount_cayu":
        server = FastAPI()
        mount_cayu(
            server,
            cayu_app,
            path="/cayu",
            dashboard=False,
            access=OpenAccess(),
            interruption_shutdown_grace_seconds=0.2,
            knowledge_publication_shutdown_grace_seconds=0.2,
        )
    else:
        server = create_server(
            cayu_app,
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(enabled=False), lifecycle=lifecycle
            ),
        )

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="cayu.server"), TestClient(server):
        pass
    # The stalled stop no longer holds the host past its deadline.
    assert time.monotonic() - started < 10
    assert stops == ["stop"]
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.status == "incomplete"
    step = outcome.step("model_policy")
    assert step is not None and step.reason == "overran_budget"
    assert any("Model policy workers did not settle" in message for message in caplog.messages)


def test_mounted_shutdown_stops_model_policy_when_work_outlasts_the_deadline() -> None:
    cayu_app = CayuApp()
    stops: list[str] = []

    async def stop_model_policy():
        stops.append("stop")

    cayu_app.stop_model_policy = stop_model_policy  # ty: ignore[invalid-assignment]

    @asynccontextmanager
    async def host_lifespan(_server):
        # The host keeps one counted operation running past Cayu's shutdown.
        release = asyncio.Event()
        holder = asyncio.create_task(cayu_app._run_worker_step(release.wait))
        try:
            yield
        finally:
            release.set()
            await holder

    server = FastAPI(lifespan=host_lifespan)
    mount_cayu(
        server,
        cayu_app,
        path="/cayu",
        dashboard=False,
        access=OpenAccess(),
        interruption_shutdown_grace_seconds=0.1,
        knowledge_publication_shutdown_grace_seconds=0.1,
    )

    with TestClient(server):
        pass

    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.status == "incomplete"
    assert outcome.open_operations == 1
    step = outcome.step("model_policy")
    assert step is not None and step.status == "settled"
    assert stops == ["stop"]


@pytest.mark.parametrize("host", ["mount_cayu", "create_server"])
def test_server_shutdown_stops_a_real_model_policy(host) -> None:
    from tests.core.test_model_policy_runtime import Channel, make_app

    from cayu.model_policy import InMemoryModelPolicyStore

    cayu_app, controller, _provider = make_app(InMemoryModelPolicyStore(), Channel())
    if host == "mount_cayu":
        server = FastAPI()
        mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())
    else:
        server = create_server(
            cayu_app,
            config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False)),
        )

    with TestClient(server):
        assert controller.status == "ready"

    # The host started the workers and aclose() stopped them.
    assert controller.status == "stopped"
    assert cayu_app.model_policy is not None and not cayu_app.model_policy._tasks
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.settled
    step = outcome.step("model_policy")
    assert step is not None and step.status == "settled"


def test_every_shutdown_step_has_a_server_log_subject() -> None:
    from cayu.server import _SHUTDOWN_STEP_LOG_SUBJECTS

    outcome = asyncio.run(CayuApp(owned_resources=(_ClosingResource(),)).aclose())
    assert {step.subsystem for step in outcome.steps} == set(_SHUTDOWN_STEP_LOG_SUBJECTS)


class _ClosingResource:
    async def close(self) -> None:
        return None


def test_a_closed_app_cannot_start_another_server_lifespan() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        pass

    with (
        pytest.raises(RuntimeError, match="already shut down"),
        TestClient(server),
    ):
        pass


def test_mount_cayu_reports_unresolved_provider_cancellation_owners(caplog) -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    async def unresolved_provider_cancellations(*, timeout_s):
        assert 0 < timeout_s <= 20.0
        return False

    cayu_app.drain_provider_operation_cancellations = unresolved_provider_cancellations
    cayu_app.provider_operation_cancellation_status = lambda: SimpleNamespace(
        active_owners=3, admission_rejections=0
    )
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with caplog.at_level(logging.WARNING, logger="cayu.server"), TestClient(server):
        pass

    assert (
        "3 provider-operation cancellation owner(s) remained unresolved after the "
        "20.000s shutdown grace period."
    ) in caplog.messages


def test_mount_cayu_drains_cascades_when_startup_recovery_fails() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        assert type(interrupting_inactive_for_seconds) is int
        assert interrupting_inactive_for_seconds >= 0
        calls.append("recover")
        raise RuntimeError("mounted recovery failed")

    async def drain_background_interruptions(*, timeout_s):
        assert 0 < timeout_s <= 20.0
        calls.append("drain")
        return True

    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    cayu_app.drain_background_interruptions = drain_background_interruptions
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with (
        pytest.raises(RuntimeError, match="mounted recovery failed"),
        TestClient(server),
    ):
        pass

    assert calls == ["recover", "drain"]


def test_mount_cayu_recovers_backlog_past_poison_delivery(monkeypatch) -> None:
    class FailingBudgetStore(InMemoryBudgetStore):
        async def append_event(self, event: Event) -> None:
            if event.type == EventType.MODEL_COMPLETED:
                raise RuntimeError("budget unavailable")
            await super().append_event(event)

    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="mounted_side_effect_backlog",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        poison = Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
        healthy = Event(type="custom.healthy", session_id=session.id)
        await store.append_events(session.id, [poison, healthy])
        return store, poison, healthy

    store, poison, healthy = asyncio.run(prepare())
    sink = InMemoryEventSink()
    cayu_app = CayuApp(
        session_store=store,
        budget_store=FailingBudgetStore(),
        event_sinks=[sink],
        enable_logging=False,
    )
    server = FastAPI()
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE",
        1,
    )
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        deliveries = asyncio.run(store.list_persisted_event_side_effect_deliveries())

    assert [event.id for event in sink.events] == [public_event_id(2)]
    assert [(delivery.event_id, delivery.status.value) for delivery in deliveries] == [
        (poison.id, "failed"),
        (healthy.id, "delivered"),
    ]


def test_run_rejects_blank_prompt_and_agent_before_runtime() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    assert client.post("/api/run", json={"prompt": " "}).status_code == 422
    assert client.post("/api/run", json={"prompt": "hello", "agent": " "}).status_code == 422
    assert (
        client.post("/api/run", json={"prompt": "hello", "model": "removed-model"}).status_code
        == 422
    )
    assert (
        client.post("/api/resume", json={"session_id": " ", "prompt": "hello"}).status_code == 422
    )
    assert (
        client.post(
            "/api/provider-operations/resolve",
            json={
                "session_id": "session_1",
                "stage_id": " ",
                "expected_run_epoch": 1,
                "action": "fallback_retry",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/tool-approvals/resolve",
            json={
                "session_id": " ",
                "approval_id": "approval_1",
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
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
                "tool_round_id": "round_1",
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
                "tool_round_id": "round_1",
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
                "tool_round_id": "round_1",
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
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
                "decision": "maybe",
            },
        ).status_code
        == 422
    )
    for bad_max_steps in (True, "7"):
        assert (
            client.post(
                "/api/tool-approvals/resolve",
                json={
                    "session_id": "session_1",
                    "approval_id": "approval_1",
                    "tool_round_id": "round_1",
                    "tool_call_id": "call_1",
                    "decision": "approve",
                    "max_steps": bad_max_steps,
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
                    "tool_round_id": "round_1",
                    "tool_call_id": "call_1",
                    "outcome": "completed",
                    "message": "confirmed externally",
                    "max_steps": bad_max_steps,
                },
            ).status_code
            == 422
        )


def test_server_resolves_provider_operation_with_bounded_audited_request() -> None:
    from cayu.approvals.tools import ResolutionActorSource

    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="provider_resolution",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "provider_resolution",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    captured = []

    async def resolve_provider_operation(request):
        captured.append(request)
        yield Event(
            type=EventType.PROVIDER_OPERATION_RESOLVED,
            session_id=request.session_id,
            payload={"resolution_action": request.action.value},
        )

    app.resolve_provider_operation = resolve_provider_operation  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/provider-operations/resolve",
        json={
            "session_id": "provider_resolution",
            "stage_id": "stage_1",
            "expected_run_epoch": 0,
            "action": "fail",
            "reason": "Operator confirmed the provider request cannot be recovered.",
            "metadata": {"ticket": "INC-757"},
            "resolved_by": {"subject": "operator-a"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert len(captured) == 1
    request = captured[0]
    assert request.session_id == "provider_resolution"
    assert request.stage_id == "stage_1"
    assert request.expected_run_epoch == 0
    assert request.action.value == "fail"
    assert request.metadata == {"ticket": "INC-757"}
    assert request.resolved_by is not None
    assert request.resolved_by.subject == "operator-a"
    assert request.resolved_by.source is ResolutionActorSource.REQUEST


def test_provider_operation_resolution_endpoint_enforces_runtime_bounds() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="provider_resolution_bounds",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "provider_resolution_bounds",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    captured = []

    async def resolve_provider_operation(request):
        captured.append(request)
        yield Event(
            type=EventType.PROVIDER_OPERATION_RESOLVED,
            session_id=request.session_id,
            payload={"resolution_action": request.action.value},
        )

    app.resolve_provider_operation = resolve_provider_operation  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    metadata_overhead = len(canonical_durable_json_bytes({"note": ""}, "metadata"))
    value = "x" * (PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES - metadata_overhead)
    at_limit_metadata = {"note": value}
    assert (
        len(canonical_durable_json_bytes(at_limit_metadata, "metadata"))
        == PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES
    )
    base_request = {
        "session_id": "provider_resolution_bounds",
        "stage_id": "stage_1",
        "expected_run_epoch": MAX_DURABLE_JSON_INTEGER,
        "action": "fail",
        "metadata": at_limit_metadata,
    }

    with client.stream(
        "POST",
        "/api/provider-operations/resolve",
        json=base_request,
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    assert len(captured) == 1

    canary = "provider-resolution-boundary-secret"
    oversized_value = value[: -len(canary)] + canary + "x"
    metadata_response = client.post(
        "/api/provider-operations/resolve",
        json={**base_request, "metadata": {"note": oversized_value}},
    )
    assert metadata_response.status_code == 422
    assert canary not in metadata_response.text

    epoch_response = client.post(
        "/api/provider-operations/resolve",
        json={**base_request, "expected_run_epoch": MAX_DURABLE_JSON_INTEGER + 1},
    )
    assert epoch_response.status_code == 422
    assert value[:128] not in epoch_response.text
    assert len(captured) == 1


@pytest.mark.parametrize("secret", ["metadata", "unavailable", "fail"])
def test_server_projects_provider_recovery_event_contract_with_schema_secret_collision(
    tmp_path,
    secret: str,
) -> None:
    session_id = "provider-recovery-event-projection"
    alias_codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
    )
    store = SQLiteSessionStore(
        tmp_path / "provider-recovery-event-projection.sqlite3",
        public_authority_alias_codec=alias_codec,
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        required = event_with_runtime_payload_authority(
            Event(
                type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                session_id=session_id,
                payload={
                    "provider": "fake",
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": "model-step-private",
                    "model_attempt_id": "model-attempt-private",
                    "source_run_epoch": 0,
                    "run_epoch": 1,
                    "operation_id": "response-public",
                    "stream_protocol": "responses-v1",
                    "status": "unavailable",
                    "recovery_reason": "unavailable",
                },
            ),
            "model_step_id",
            "model_attempt_id",
            "operation_id",
            "stream_protocol",
        )
        resolved = event_with_runtime_payload_authority(
            Event(
                type=EventType.PROVIDER_OPERATION_RESOLVED,
                session_id=session_id,
                payload={
                    "provider": "fake",
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": "model-step-private",
                    "model_attempt_id": "model-attempt-private",
                    "source_run_epoch": 0,
                    "run_epoch": 1,
                    "operation_id": "response-public",
                    "stream_protocol": "responses-v1",
                    "status": "unavailable",
                    "stage_id": "stage-private",
                    "resolution_id": "resolution-private",
                    "resolution_action": "fail",
                    "recovery_reason": "unavailable",
                    "duplicate_request_risk": True,
                    "reason": "operator decision",
                    "metadata": {"ticket": "INC-757"},
                    "resolved_by": {
                        "source": "request",
                        "subject": "operator",
                        "tenant": "tenant-a",
                    },
                },
            ),
            "model_step_id",
            "model_attempt_id",
            "operation_id",
            "resolution_id",
            "stage_id",
            "stream_protocol",
        )
        await app._event_writer.emit(required)
        await app._event_writer.emit(resolved)

    try:
        asyncio.run(seed())
        client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
        response = client.get(f"/api/sessions/{session_id}/events?limit=10")

        assert response.status_code == 200
        events = response.json()["events"]
        required_payload = events[0]["payload"]
        resolved_payload = events[1]["payload"]
        assert required_payload["operation_id"] == "response-public"
        assert required_payload["stream_protocol"] == "responses-v1"
        assert required_payload["model_step_id"] == PRIVATE_EVENT_AUTHORITY
        assert required_payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
        assert required_payload["status"] == "unavailable"
        assert required_payload["recovery_reason"] == "unavailable"
        assert resolved_payload["operation_id"] == "response-public"
        assert resolved_payload["stream_protocol"] == "responses-v1"
        assert "stage_id" not in resolved_payload
        assert "resolution_id" not in resolved_payload
        assert resolved_payload["status"] == "unavailable"
        assert resolved_payload["recovery_reason"] == "unavailable"
        assert resolved_payload["resolution_action"] == "fail"
        assert resolved_payload["duplicate_request_risk"] is True
        assert resolved_payload["metadata"] == {"ticket": "INC-757"}
        assert resolved_payload["resolved_by"] == {
            "source": "request",
            "subject": "operator",
            "tenant": "tenant-a",
        }
    finally:
        asyncio.run(store.close())


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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    async def recover_tool_approval(request):
        recovered_requests.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resolve_tool_approval = resolve_tool_approval
    app.recover_tool_approval = recover_tool_approval
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_resolve_metadata",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
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
            "tool_round_id": "round_2",
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

    mismatched_alias = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_resolve_metadata",
            "approval_id": public_event_linkage_id(1, "tool_call_id"),
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
    )
    assert mismatched_alias.status_code == 409
    assert "field-mismatched" in mismatched_alias.json()["detail"]
    assert len(resolved_requests) == 1


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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_endpoint/interrupt",
        json={
            "reason": "operator requested stop",
            "metadata": {"ticket": "incident-42"},
            "requested_by": {
                "subject": "dev-operator",
                "source": "http_auth",
                "claims": {"role": "operator"},
            },
        },
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "operator requested stop" in body
    data_line = next(line for line in lines if line.startswith("data: "))
    event = json.loads(data_line.removeprefix("data: "))
    assert event["payload"]["requested_by"] == {
        "subject": "dev-operator",
        "tenant": None,
        "source": "request",
    }
    assert "claims" not in event["payload"]["requested_by"]

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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/sessions/session_interrupt_completed/interrupt")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session cannot be interrupted from status: completed",
    }


def test_interrupt_session_endpoint_rejects_completion_race_before_streaming() -> None:
    class CompletingRaceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_idempotent/interrupt",
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "already interrupted" in body


def test_server_deletes_session_and_is_idempotent() -> None:
    async def seed(store):
        await _create_session(store, "sess_del")

    _, client = _lifecycle_store_and_client(seed)

    assert client.delete("/api/sessions/sess_del").status_code == 204
    assert client.get("/api/sessions/sess_del").status_code == 404
    # Idempotent: deleting a missing session is still 204.
    assert client.delete("/api/sessions/sess_del").status_code == 204


def test_server_delete_running_session_conflicts() -> None:
    async def seed(store):
        await _create_session(store, "sess_run")
        await store.update_status("sess_run", SessionStatus.RUNNING)

    _, client = _lifecycle_store_and_client(seed)

    response = client.delete("/api/sessions/sess_run")
    assert response.status_code == 409


def test_server_delete_active_durable_operation_conflicts() -> None:
    async def seed(store):
        await _create_session(store, "sess_active_operation")
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
        await store.checkpoint(
            "sess_active_operation",
            {
                "session_operations": {
                    "version": 1,
                    "active_operation_id": "operation-1",
                    "records": {
                        "request-1": {
                            "operation_id": "operation-1",
                            "status": "running",
                            "claim_expires_at": expires_at.isoformat(),
                        }
                    },
                }
            },
        )

    _, client = _lifecycle_store_and_client(seed)

    response = client.delete("/api/sessions/sess_active_operation")
    assert response.status_code == 409
    assert "durable operation operation-1 is active" in response.json()["detail"]


def test_server_updates_session_labels() -> None:
    async def seed(store):
        await _create_session(store, "sess_lab", labels={"team": "research"})

    _, client = _lifecycle_store_and_client(seed)

    response = client.patch("/api/sessions/sess_lab/labels", json={"labels": {"stage": "review"}})
    assert response.status_code == 200
    # Full replacement: the old "team" label is gone.
    assert response.json()["labels"] == {"stage": "review"}
    missing = client.patch("/api/sessions/sess_missing/labels", json={"labels": {}})
    assert missing.status_code == 404
    # An invalid label (blank value) is a client error (422), not an unhandled 500.
    invalid = client.patch("/api/sessions/sess_lab/labels", json={"labels": {"k": "   "}})
    assert invalid.status_code == 422
    non_durable = client.patch(
        "/api/sessions/sess_lab/labels", json={"labels": {"stage": "review\x00hidden"}}
    )
    assert non_durable.status_code == 422
    # A typo'd key must 422 (extra="forbid"), NOT silently replace all labels with {}.
    typo = client.patch("/api/sessions/sess_lab/labels", json={"lables": {"a": "b"}})
    assert typo.status_code == 422
    # A missing required field must 422, not default to an empty (wiping) replacement.
    empty_body = client.patch("/api/sessions/sess_lab/labels", json={})
    assert empty_body.status_code == 422
    # The labels were not wiped by any of the rejected requests.
    assert client.get("/api/sessions/sess_lab").json()["labels"] == {"stage": "review"}


def test_server_updates_session_metadata() -> None:
    async def seed(store):
        await _create_session(
            store,
            "sess_meta",
            metadata={
                "a": 1,
                "subagent": {"mode": "background"},
                "cayu:taint_labels": ["untrusted"],
            },
        )

    _, client = _lifecycle_store_and_client(seed)

    response = client.patch("/api/sessions/sess_meta/metadata", json={"metadata": {"b": [1, 2]}})
    assert response.status_code == 200
    expected = {
        "b": [1, 2],
        "subagent": {"mode": "background"},
        "cayu:taint_labels": ["untrusted"],
    }
    assert response.json()["metadata"] == expected
    missing = client.patch("/api/sessions/sess_missing/metadata", json={"metadata": {}})
    assert missing.status_code == 404
    # Typo'd key / missing field must 422, never silently wipe metadata.
    assert client.patch("/api/sessions/sess_meta/metadata", json={"metadat": {}}).status_code == 422
    assert client.patch("/api/sessions/sess_meta/metadata", json={}).status_code == 422
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            json={"metadata": {"subagent": {}}},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            json={"metadata": {"cayu:taint_labels": []}},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata", json={"metadata": {"nested": ["value\x00"]}}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            content=b'{"metadata":{"nested":"\\ud800"}}',
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )
    assert client.get("/api/sessions/sess_meta").json()["metadata"] == expected


async def _post_and_disconnect_before_first_body(
    server: FastAPI,
    path: str,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Send one ASGI POST and disconnect immediately after HTTP acceptance."""
    request_sent = False
    response_started = asyncio.Event()
    disconnect_delivered = asyncio.Event()
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(body).encode(),
                "more_body": False,
            }
        await response_started.wait()
        disconnect_delivered.set()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and disconnect_delivered.is_set():
            # The server may race one write already queued at its ASGI boundary;
            # model the socket loss by dropping it before the client receives it.
            return
        messages.append(message)
        if message["type"] == "http.response.start":
            response_started.set()
            # Do not let the response task emit its first body frame before the
            # disconnect listener has observed the injected network loss.
            await disconnect_delivered.wait()

    await server(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    return messages


def test_run_accepts_and_detaches_before_environment_factory_finishes() -> None:
    class BlockingEnvironmentFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def create(
            self,
            _request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.started.set()
            await self.release.wait()
            return EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="dynamic"))
            )

    factory = BlockingEnvironmentFactory()
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_factory_acceptance_boundary"

    async def exercise() -> tuple[bool, list[dict[str, Any]]]:
        request_task = asyncio.create_task(
            _post_and_disconnect_before_first_body(
                server,
                "/api/run",
                {"prompt": "hello", "session_id": session_id},
            )
        )
        await asyncio.wait_for(factory.started.wait(), timeout=10)
        done, _ = await asyncio.wait({request_task}, timeout=0.2)
        accepted_before_release = request_task in done
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert state.status is SessionStatus.RUNNING
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert active_runs[0].runtime_task is not request_task
        assert not active_runs[0].runtime_task.done()

        factory.release.set()
        messages = await asyncio.wait_for(request_task, timeout=5)
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                return accepted_before_release, messages
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("detached run did not complete after factory release")
            await asyncio.sleep(0.01)

    accepted_before_release, messages = asyncio.run(exercise())

    assert accepted_before_release
    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [200]


def test_interrupt_after_run_acceptance_cancels_detached_provider() -> None:
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.started is None or self.cancelled is None or self.never_complete is None:
                raise AssertionError("BlockingProvider test events were not initialized.")
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = BlockingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_detached_interrupt_ownership"

    async def exercise() -> tuple[list[dict[str, Any]], list[Event]]:
        provider.started = asyncio.Event()
        provider.cancelled = asyncio.Event()
        provider.never_complete = asyncio.Event()
        request_task = asyncio.create_task(
            _post_and_disconnect_before_first_body(
                server,
                "/api/run",
                {"prompt": "hello", "session_id": session_id},
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        messages = await asyncio.wait_for(request_task, timeout=1)

        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert active_runs[0].runtime_task is not request_task
        assert not active_runs[0].runtime_task.done()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        await asyncio.wait_for(provider.cancelled.wait(), timeout=1)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return messages, interrupt_events

    messages, interrupt_events = asyncio.run(exercise())

    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [200]
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]


def test_interrupt_after_acceptance_before_observer_start_reaches_runtime() -> None:
    class RecordingProvider(OneShotProvider):
        dispatches = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.dispatches += 1
            async for event in super().stream(request):
                yield event

    provider = RecordingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_before_observer_start"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        response = await _accepted_event_stream_response(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ),
            cayu_app=app,
            session_id=session_id,
        )
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert not active_runs[0].runtime_task.done()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        observed = [message async for message in response.body_iterator]
        runtime_task = active_runs[0].runtime_task
        await asyncio.wait_for(asyncio.shield(runtime_task), timeout=5)
        assert not runtime_task.cancelled() and runtime_task.cancelling() == 0
        assert provider.dispatches == 0
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[:2] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_STARTED,
    ]
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_interrupt_before_observer_start_cancels_environment_factory() -> None:
    class BlockingEnvironmentFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.never_complete = asyncio.Event()

        async def create(
            self,
            _request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            raise AssertionError("unreachable")

    factory = BlockingEnvironmentFactory()
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_factory_before_observer_start"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        response = await _accepted_event_stream_response(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ),
            cayu_app=app,
            session_id=session_id,
        )
        assert not factory.started.is_set()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        observed = [message async for message in response.body_iterator]
        await asyncio.wait_for(factory.started.wait(), timeout=10)
        await asyncio.wait_for(factory.cancelled.wait(), timeout=1)
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[:2] == [
        EventType.INTERACTION_STARTED,
        EventType.ENVIRONMENT_FACTORY_STARTED,
    ]
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_interrupt_during_run_acceptance_finishes_task_bookkeeping() -> None:
    class RecordingProvider(OneShotProvider):
        def __init__(self) -> None:
            self.dispatches = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.dispatches += 1
            async for event in super().stream(request):
                yield event

    class BlockingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.release_create = asyncio.Event()

        async def create_running_task(self, request, *, session_invocation):
            self.create_started.set()
            await self.release_create.wait()
            return await super().create_running_task(
                request,
                session_invocation=session_invocation,
            )

    task_store = BlockingTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    provider = RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_during_acceptance_bookkeeping"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        async def create_task_after_accept(_event: Event) -> None:
            snapshot = await app.session_store.load_invocation_snapshot(session_id)
            if snapshot is None:
                raise AssertionError("Accepted run session was not persisted.")
            await task_store.create_running_task(
                TaskCreate(
                    task_id="task_interrupt_during_acceptance_bookkeeping",
                    type="run",
                    session_id=session_id,
                ),
                session_invocation=snapshot,
            )

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="operator stop")
                )
            ]

        run_request = run_request_with_runtime_generated_authority(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                task_id="task_interrupt_during_acceptance_bookkeeping",
                messages=[Message.text("user", "hello")],
            ),
            "task_id",
        )
        response_task = asyncio.create_task(
            _accepted_event_stream_response(
                app.run(run_request),
                cayu_app=app,
                session_id=session_id,
                after_accept=create_task_after_accept,
            )
        )
        await asyncio.wait_for(task_store.create_started.wait(), timeout=10)
        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.sleep(0)
        assert not response_task.done()

        task_store.release_create.set()
        response, interrupt_events = await asyncio.gather(response_task, interrupt_task)
        observed = [message async for message in response.body_iterator]
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        tasks = await task_store.list_tasks()
        assert [task.id for task in tasks] == ["task_interrupt_during_acceptance_bookkeeping"]
        assert tasks[0].session_id == session_id
        assert tasks[0].status is TaskStatus.RUNNING
        assert provider.dispatches == 0
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        assert [record.event.type for record in records] == [
            json.loads(message["data"])["type"] for message in observed
        ]
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    # Acceptance is interaction.started, not session.started. Deferred
    # interruption can arrive at the group-membership read before startup or
    # at the next await after startup; neither ordering may lose bookkeeping.
    assert observed_types in (
        [
            EventType.INTERACTION_STARTED,
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        ],
        [
            EventType.INTERACTION_STARTED,
            EventType.SESSION_STARTED,
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        ],
    )
    assert status is SessionStatus.INTERRUPTED


def test_run_route_interrupt_during_acceptance_state_read_keeps_task_linked() -> None:
    class BlockingAcceptanceStateStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_next_state_read = True
            self.state_read_started = asyncio.Event()
            self.release_state_read = asyncio.Event()

        async def load_invocation_snapshot(self, session_id: str):
            if self.block_next_state_read:
                self.block_next_state_read = False
                self.state_read_started.set()
                await self.release_state_read.wait()
            return await super().load_invocation_snapshot(session_id)

    session_store = BlockingAcceptanceStateStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_interrupt_during_acceptance_state_read"

    async def exercise() -> tuple[httpx.Response, list[Event], SessionStatus, list[Task]]:
        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="operator stop")
                )
            ]

        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/api/run",
                    json={"prompt": "hello", "session_id": session_id},
                )
            )
            await asyncio.wait_for(session_store.state_read_started.wait(), timeout=10)
            interrupt_task = asyncio.create_task(interrupt())
            await asyncio.sleep(0)
            session_store.release_state_read.set()
            response, interrupt_events = await asyncio.wait_for(
                asyncio.gather(request_task, interrupt_task),
                timeout=2,
            )

        state = await session_store.load_state(session_id)
        assert state is not None
        return response, interrupt_events, state.status, await task_store.list_tasks()

    response, interrupt_events, status, tasks = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert frames[-1]["data"]["type"] == EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert status is SessionStatus.INTERRUPTED
    assert len(tasks) == 1
    assert tasks[0].session_id == session_id
    assert tasks[0].status is TaskStatus.RUNNING


def test_request_cancellation_during_acceptance_does_not_cancel_detached_run() -> None:
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.started.set()
            await self.release.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = BlockingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_cancelled_acceptance_owner"

    async def exercise() -> tuple[SessionStatus, list[EventType | str]]:
        callback_started = asyncio.Event()
        release_callback = asyncio.Event()

        async def after_accept(_event: Event) -> None:
            callback_started.set()
            await release_callback.wait()

        response_task = asyncio.create_task(
            _accepted_event_stream_response(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ),
                cayu_app=app,
                session_id=session_id,
                after_accept=after_accept,
            )
        )
        await asyncio.wait_for(callback_started.wait(), timeout=10)
        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task

        release_callback.set()
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert not active_runs[0].runtime_task.done()

        provider.release.set()
        deadline = asyncio.get_running_loop().time() + 1
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("detached run did not complete after request cancellation")
            await asyncio.sleep(0.01)

        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        # Durable completion precedes trailing cleanup and active-task removal.
        await asyncio.wait_for(asyncio.shield(active_runs[0].runtime_task), timeout=5)
        assert app._session_control.active_runs(session_id) == ()
        return state.status, [record.event.type for record in records]

    status, event_types = asyncio.run(exercise())

    assert status is SessionStatus.COMPLETED
    assert event_types[-1] == EventType.SESSION_COMPLETED


def test_event_source_cancellation_before_acceptance_does_not_hang_request() -> None:
    async def cancelled_stream() -> AsyncIterator[Event]:
        raise asyncio.CancelledError
        yield  # pragma: no cover - makes this an async generator

    async def exercise() -> None:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                _accepted_event_stream_response(
                    cancelled_stream(),
                    cayu_app=CayuApp(enable_logging=False),
                    session_id="session_cancelled_before_acceptance",
                ),
                timeout=1,
            )

    asyncio.run(exercise())


def test_runtime_failure_survives_cancelled_stream_cleanup() -> None:
    class CancelledCloseStream:
        def __init__(self) -> None:
            self.calls = 0

        def __aiter__(self) -> CancelledCloseStream:
            return self

        async def __anext__(self) -> Event:
            self.calls += 1
            if self.calls == 1:
                return event_with_durable_sequence(
                    Event(
                        id="event_before_runtime_failure",
                        type="custom.before_runtime_failure",
                        session_id="session_cancelled_stream_cleanup",
                    ),
                    1,
                )
            raise RuntimeError("runtime failed")

        async def aclose(self) -> None:
            raise asyncio.CancelledError

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            CancelledCloseStream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_cancelled_stream_cleanup",
        )

        async def collect() -> list[dict[str, str]]:
            return [message async for message in response.body_iterator]

        return await asyncio.wait_for(collect(), timeout=1)

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_cancelled_stream_cleanup:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "runtime_failed"


def test_terminal_publication_uncertainty_requests_durable_reconciliation() -> None:
    publication_failure = ConnectionError("terminal append acknowledgement lost")
    reconciliation_failure = TimeoutError("terminal reconciliation unavailable")

    async def uncertain_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_terminal_uncertainty",
                type="custom.before_terminal_uncertainty",
                session_id="session_terminal_uncertainty",
            ),
            1,
        )
        raise TerminalEventPublicationUncertain(
            event=Event(
                id="event_terminal",
                type=EventType.SESSION_COMPLETED,
                session_id="session_terminal_uncertainty",
            ),
            publication_failure=publication_failure,
            reconciliation_failure=reconciliation_failure,
        )

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            uncertain_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_terminal_uncertainty",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_terminal_uncertainty:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "terminal_event_publication_uncertain"
    assert error["retryable"] is True
    assert error["session_id"] == "session_terminal_uncertainty"
    assert error["error_type"] == "TerminalEventPublicationUncertain"


def test_preaccept_terminal_publication_uncertainty_is_not_reported_as_conflict() -> None:
    async def uncertain_stream() -> AsyncIterator[Event]:
        raise TerminalEventPublicationUncertain(
            event=Event(
                id="event_preaccept_terminal_uncertainty",
                type=EventType.SESSION_INTERRUPTED,
                session_id="session_preaccept_terminal_uncertainty",
            ),
            publication_failure=ConnectionError("terminal append acknowledgement lost"),
            reconciliation_failure=TimeoutError("terminal reconciliation unavailable"),
        )
        yield  # pragma: no cover

    async def exercise() -> None:
        with pytest.raises(fastapi.HTTPException) as raised:
            await _accepted_event_stream_response(
                uncertain_stream(),
                cayu_app=CayuApp(enable_logging=False),
                session_id="session_preaccept_terminal_uncertainty",
                conflict_error_types=(RuntimeError,),
            )
        assert raised.value.status_code == 500
        assert raised.value.detail == (
            "Terminal event publication outcome is uncertain; inspect durable session "
            "state before retrying the mutation."
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal_append_committed", [False, True])
def test_interrupt_accepts_before_first_frame_terminal_publication_uncertainty(
    terminal_append_committed: bool,
) -> None:
    class AmbiguousInterruptTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.attempted_terminal_event_id: str | None = None
            self.unreadable_event_id: str | None = None

        async def append_event(self, session_id: str, event: Event) -> None:
            if self.armed and event.type == EventType.SESSION_INTERRUPTED:
                self.armed = False
                self.attempted_terminal_event_id = event.id
                self.unreadable_event_id = event.id
                if terminal_append_committed:
                    await super().append_event(session_id, event)
                raise ConnectionError("terminal append acknowledgement lost")
            await super().append_event(session_id, event)

        async def query_events(self, query: EventQuery):
            if self.unreadable_event_id is not None and query.event_id == self.unreadable_event_id:
                self.unreadable_event_id = None
                raise TimeoutError("terminal reconciliation unavailable")
            return await super().query_events(query)

    store = AmbiguousInterruptTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session-interrupt-terminal-publication-uncertain"
    mutation_id = "mutation-interrupt-terminal-publication-uncertain"

    async def prepare() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create pending session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(prepare())
    store.armed = True
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/interrupt",
        json={"reason": "operator stop"},
        headers={"Cayu-Mutation-ID": mutation_id},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert len(frames) == 1
    assert frames[0]["event"] == "error"
    assert frames[0]["data"]["code"] == "terminal_event_publication_uncertain"
    assert frames[0]["data"]["retryable"] is True

    events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"order_by": "sequence_asc", "limit": 100},
    ).json()["events"]
    interrupted = [event for event in events if event["type"] == EventType.SESSION_INTERRUPTED]
    accepted = [event for event in events if event["type"] == EventType.SERVER_MUTATION_ACCEPTED]

    assert len(interrupted) == int(terminal_append_committed)
    assert len(accepted) == 1
    expected_accepted_payload = {
        "mutation_id": mutation_id,
        "mutation_kind": "interrupt",
        "accepted_event_id": (
            interrupted[0]["id"] if terminal_append_committed else PRIVATE_EVENT_AUTHORITY
        ),
        "accepted_event_type": EventType.SESSION_INTERRUPTED,
        "accepted_event_publication_uncertain": True,
    }
    if terminal_append_committed:
        expected_accepted_payload["accepted_event_sequence"] = interrupted[0]["sequence"]
    assert accepted[0]["payload"] == expected_accepted_payload

    durable_markers = asyncio.run(
        store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SERVER_MUTATION_ACCEPTED,
                limit=1,
            )
        )
    )
    assert len(durable_markers) == 1
    if terminal_append_committed:
        assert (
            durable_markers[0].event.payload["accepted_event_sequence"]
            == interrupted[0]["sequence"]
        )
        assert "accepted_event_id" not in durable_markers[0].event.payload
    else:
        assert (
            durable_markers[0].event.payload["accepted_event_id"]
            == store.attempted_terminal_event_id
        )
    if terminal_append_committed:
        assert events.index(interrupted[0]) < events.index(accepted[0])
    state = client.get(f"/api/sessions/{session_id}/state").json()
    assert state["status"] == SessionStatus.INTERRUPTED


def test_request_cancellation_does_not_cancel_uncertain_acceptance_marker() -> None:
    class BlockingUncertainAcceptanceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.unreadable_event_id: str | None = None
            self.marker_started = asyncio.Event()
            self.release_marker = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            if self.armed and event.type == EventType.SESSION_INTERRUPTED:
                self.armed = False
                self.unreadable_event_id = event.id
                raise ConnectionError("terminal append acknowledgement lost")
            if event.type == EventType.SERVER_MUTATION_ACCEPTED:
                self.marker_started.set()
                await self.release_marker.wait()
            await super().append_event(session_id, event)

        async def query_events(self, query: EventQuery):
            if self.unreadable_event_id is not None and query.event_id == self.unreadable_event_id:
                self.unreadable_event_id = None
                raise TimeoutError("terminal reconciliation unavailable")
            return await super().query_events(query)

    async def exercise() -> tuple[SessionStatus, list[Event]]:
        store = BlockingUncertainAcceptanceStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "session-cancelled-uncertain-acceptance"
        mutation_id = "mutation-cancelled-uncertain-acceptance"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create pending session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        store.armed = True
        server = create_server(app, config=_LOCAL_SERVER_CONFIG)
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            request_task = asyncio.create_task(
                client.post(
                    f"/api/sessions/{session_id}/interrupt",
                    json={"reason": "operator stop"},
                    headers={"Cayu-Mutation-ID": mutation_id},
                )
            )
            await asyncio.wait_for(store.marker_started.wait(), timeout=10)
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
            store.release_marker.set()

            deadline = asyncio.get_running_loop().time() + 1
            while True:
                records = await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SERVER_MUTATION_ACCEPTED,
                        limit=10,
                    )
                )
                if records:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "Detached uncertainty acceptance did not persist its mutation marker."
                    )
                await asyncio.sleep(0.01)

        state = await store.load_state(session_id)
        assert state is not None
        return state.status, [record.event for record in records]

    status, markers = asyncio.run(exercise())

    assert status is SessionStatus.INTERRUPTED
    assert len(markers) == 1
    assert markers[0].payload["mutation_id"] == "mutation-cancelled-uncertain-acceptance"
    assert markers[0].payload["accepted_event_publication_uncertain"] is True


def test_runtime_cancellation_group_becomes_structured_stream_error() -> None:
    async def grouped_failure_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_grouped_failure",
                type="custom.before_grouped_failure",
                session_id="session_grouped_stream_failure",
            ),
            1,
        )
        raise BaseExceptionGroup(
            "runtime cleanup cancelled and failed",
            [asyncio.CancelledError(), RuntimeError("cleanup failed")],
        )

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            grouped_failure_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_grouped_stream_failure",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_grouped_stream_failure:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "runtime_failed"


def test_runtime_cancellation_only_group_is_not_reported_as_runtime_failure() -> None:
    cancellation_group = BaseExceptionGroup(
        "runtime cleanup cancelled",
        [asyncio.CancelledError()],
    )

    async def grouped_cancellation_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_grouped_cancellation",
                type="custom.before_grouped_cancellation",
                session_id="session_grouped_stream_cancellation",
            ),
            1,
        )
        raise cancellation_group

    async def exercise() -> tuple[list[dict[str, str]], BaseExceptionGroup]:
        response, pump_task, _ = _start_detached_event_stream_response(
            grouped_cancellation_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_grouped_stream_cancellation",
        )
        messages = [message async for message in response.body_iterator]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await pump_task
        return messages, exc_info.value

    messages, failure = asyncio.run(exercise())

    assert failure is cancellation_group
    assert [message.get("event") for message in messages] == [None]


def test_live_projection_requires_a_unique_durable_event_record() -> None:
    app = CayuApp(enable_logging=False)
    queried_limits: list[int | None] = []

    async def duplicate_query(query: EventQuery | None = None):
        assert query is not None
        queried_limits.append(query.limit)
        return [
            EventRecord(
                sequence=sequence,
                event=Event(
                    id="duplicate-event",
                    type="custom.duplicate",
                    session_id="duplicate-session",
                ),
            )
            for sequence in (1, 2)
        ]

    app.session_store.query_events = duplicate_query

    async def event_stream() -> AsyncIterator[Event]:
        yield Event(
            id="duplicate-event",
            type="custom.duplicate",
            session_id="duplicate-session",
        )

    async def exercise() -> list[dict[str, str]]:
        response = _detached_event_stream_response(
            event_stream(),
            cayu_app=app,
            session_id="duplicate-session",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert queried_limits == [2]
    assert messages[-1]["event"] == "error"


def test_accepted_stream_driver_finishes_when_response_start_send_fails() -> None:
    async def exercise() -> bool:
        completed = asyncio.Event()

        async def event_stream() -> AsyncIterator[Event]:
            yield event_with_durable_sequence(
                Event(
                    id="event_response_start_failure",
                    type="custom.accepted",
                    session_id="session_response_start_failure",
                ),
                1,
            )
            completed.set()

        response = await _accepted_event_stream_response(
            event_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_response_start_failure",
        )

        async def receive() -> dict[str, Any]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(_message: dict[str, Any]) -> None:
            raise OSError("response send failed")

        with pytest.raises(OSError, match="response send failed"):
            await response(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/api/run",
                    "raw_path": b"/api/run",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [],
                    "client": ("127.0.0.1", 50000),
                    "server": ("testserver", 80),
                },
                receive,
                send,
            )
        await asyncio.wait_for(completed.wait(), timeout=1)
        return completed.is_set()

    assert asyncio.run(exercise()) is True


def _detached_observer_first_message(events: list[Event]) -> tuple[dict[str, str], float, bool]:
    async def scenario() -> tuple[dict[str, str], float, bool]:
        completed = False

        async def event_stream() -> AsyncIterator[Event]:
            nonlocal completed
            try:
                for sequence, event in enumerate(events, start=1):
                    yield event_with_durable_sequence(event, sequence)
            finally:
                completed = True

        response = _detached_event_stream_response(
            event_stream(),
            cayu_app=CayuApp(),
            session_id="session_observer_bound",
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if completed:
                break
        iterator = response.body_iterator.__aiter__()
        message = await anext(iterator)
        await iterator.aclose()
        await asyncio.sleep(0)
        return message, response.send_timeout, completed

    return asyncio.run(scenario())


def test_detached_observer_frame_count_is_bounded_without_stopping_pump() -> None:
    events = [
        Event(
            id=f"event_{index}",
            type="custom.observer",
            session_id="session_observer_bound",
        )
        for index in range(SSE_OBSERVER_MAX_FRAMES + 1)
    ]

    message, send_timeout, completed = _detached_observer_first_message(events)
    data = json.loads(message["data"])

    assert completed is True
    assert send_timeout == SSE_SEND_TIMEOUT_SECONDS
    assert message["event"] == "error"
    assert data["kind"] == "observer"
    assert data["code"] == "observer_lagged"
    assert data["retryable"] is True


def test_detached_observer_does_not_misclassify_a_healthy_synchronous_burst() -> None:
    app = CayuApp()
    completed = False

    async def synchronous_burst(request: RunRequest) -> AsyncIterator[Event]:
        nonlocal completed
        try:
            for index in range(SSE_OBSERVER_MAX_FRAMES + 1):
                yield event_with_durable_sequence(
                    Event(
                        id=f"event_{index}",
                        type="custom.observer",
                        session_id=request.session_id,
                    ),
                    index + 1,
                )
        finally:
            completed = True

    app.run = synchronous_burst  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert completed is True
    assert len(frames) == SSE_OBSERVER_MAX_FRAMES + 1
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(index + 1) for index in range(SSE_OBSERVER_MAX_FRAMES + 1)
    ]
    assert all(frame.get("event") != "error" for frame in frames)


def test_detached_observer_serialized_bytes_are_bounded() -> None:
    payload_chars = SSE_OBSERVER_MAX_BYTES // 2 + 1024
    events = [
        Event(
            id=f"event_{index}",
            type="custom.observer",
            session_id="session_observer_bound",
            payload={"value": "x" * payload_chars},
        )
        for index in range(2)
    ]

    message, _, completed = _detached_observer_first_message(events)
    data = json.loads(message["data"])

    assert completed is True
    assert data["kind"] == "observer"
    assert data["code"] == "observer_lagged"


def test_detached_oversized_frame_does_not_stop_runtime_pump() -> None:
    event = Event(
        id="event_large",
        type="custom.observer",
        session_id="session_observer_bound",
        payload={"value": "x" * SSE_EVENT_DATA_MAX_BYTES},
    )

    message, _, completed = _detached_observer_first_message([event])
    data = json.loads(message["data"])

    assert completed is True
    assert data["kind"] == "observer"
    assert data["code"] == "event_frame_too_large"
    assert data["retryable"] is False


def test_replay_polling_backs_off_and_resets_after_events() -> None:
    interval = 0.05
    observed = []
    for _ in range(7):
        observed.append(interval)
        interval = _next_replay_poll_interval(interval, received_events=False)

    assert observed == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0]
    assert _next_replay_poll_interval(interval, received_events=True) == 0.05


def test_run_stream_carries_resumable_event_ids_and_replays_on_last_event_id() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_id = "session-dashboard-run"
    run_body = {"prompt": "hello", "session_id": session_id}
    with client.stream("POST", "/api/run", json=run_body) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    assert frames
    assert frames[0]["data"]["session_id"] == session_id
    # Every frame carries a resumable id of the form `<session_id>:<event_id>`.
    for frame in frames:
        assert frame["id"] == f"{session_id}:{frame['data']['id']}"

    async def fail_unbounded_session_load(*_args, **_kwargs):
        raise AssertionError("SSE replay must use the bounded state projection")

    app.session_store.load = fail_unbounded_session_load  # type: ignore[method-assign]

    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_execution
    app.resume = unexpected_execution

    # A reconnect with Last-Event-ID replays the persisted events the client missed
    # instead of starting a new run.
    queries = []
    original_query_events = app.session_store.query_events

    async def query_events(query=None):
        queries.append(query)
        return await original_query_events(query)

    app.session_store.query_events = query_events
    with client.stream(
        "POST",
        "/api/run",
        json=run_body,
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        assert response.status_code == 200
        replayed = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in replayed] == [
        frame["data"]["id"] for frame in frames[1:]
    ]
    assert [frame["data"]["interaction_id"] for frame in replayed] == [
        frame["data"]["interaction_id"] for frame in frames[1:]
    ]
    assert replayed[-1]["data"]["type"] == "session.completed"
    alias_lookup = next(query for query in queries if query.after_sequence == 0)
    replay_page = next(query for query in queries if query.after_sequence == 1)
    assert alias_lookup.limit == 1
    assert replay_page.limit == SSE_REPLAY_PAGE_EVENTS

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        assert response.status_code == 200
        resume_replayed = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in resume_replayed] == [
        frame["data"]["id"] for frame in frames[1:]
    ]

    with client.stream(
        "POST",
        "/api/run",
        json=run_body,
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        replayed_from_start = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in replayed_from_start] == [
        frame["data"]["id"] for frame in frames
    ]
    assert [frame["data"]["interaction_id"] for frame in replayed_from_start] == [
        frame["data"]["interaction_id"] for frame in frames
    ]

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        resume_replayed_from_start = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in resume_replayed_from_start] == [
        frame["data"]["id"] for frame in frames
    ]
    assert executed == []
    # No new session was created by the replay request.
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [session["id"] for session in sessions] == [session_id]
    assert len(client.get("/api/tasks").json()) == 1


def test_legacy_custom_event_uses_one_safe_rest_sse_and_replay_projection() -> None:
    secret = "eventscope-canary"
    session_id = f"session-{secret}"
    private_event_id = f"private-{secret}-event"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "inspect legacy event")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id=private_event_id,
                    type=f"custom.{secret}",
                    session_id=session_id,
                    payload={
                        secret: secret,
                        "tool_call_id": f"tool-{secret}",
                    },
                ),
                Event(
                    id="private-terminal-event",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    history = client.get(f"/api/sessions/{session_id}/events").json()["events"]
    assert [event["id"] for event in history] == [
        public_event_id(1),
        public_event_id(2),
    ]
    assert history[0]["type"] == REDACTED_CUSTOM_EVENT_TYPE
    assert secret not in repr(history[0])

    for event_id in (public_event_id(1), private_event_id):
        selected = client.get(
            f"/api/sessions/{session_id}/events",
            params={"event_id": event_id},
        ).json()["events"]
        assert [event["id"] for event in selected] == [public_event_id(1)]
        assert secret not in repr(selected)

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(1),
        public_event_id(2),
    ]
    assert frames[0]["data"]["type"] == REDACTED_CUSTOM_EVENT_TYPE
    assert secret not in repr(frames)

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        projected_session_replay = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in projected_session_replay] == [public_event_id(2)]
    assert secret not in repr(projected_session_replay)

    for marker_id in (public_event_id(1), private_event_id):
        with client.stream(
            "POST",
            "/api/run",
            json={"prompt": "ignored", "session_id": session_id},
            headers={"Last-Event-ID": f"{session_id}:{marker_id}"},
        ) as response:
            replay = [frame for frame in _sse_frames(response) if "data" in frame]
        assert [frame["data"]["id"] for frame in replay] == [public_event_id(2)]
        assert secret not in repr(replay)


def test_replay_accepts_server_issued_redacted_colon_bearing_session_marker() -> None:
    secret = "colon-session-canary"
    session_id = f"legacy-{secret}:partition"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "inspect legacy session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                ),
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        initial = [frame for frame in _sse_frames(response) if "data" in frame]

    assert len(initial) == 2
    assert secret not in initial[0]["id"]
    assert public_event_envelope_alias_field(initial[0]["id"].split(":", 1)[0]) == "session_id"

    with client.stream(
        "POST",
        "/api/resume",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": initial[0]["id"]},
    ) as response:
        replay = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in replay] == [public_event_id(2)]
    assert secret not in repr(replay)


def test_unknown_public_event_alias_cannot_collide_with_a_private_event_id() -> None:
    private_sentinel = "__cayu_missing_private_event__"
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_alias_miss",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            "session_alias_miss",
            Event(
                id=private_sentinel,
                type=EventType.SESSION_STARTED,
                session_id="session_alias_miss",
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    missing = client.get(
        "/api/sessions/session_alias_miss/events",
        params={"event_id": public_event_id(999)},
    )
    assert missing.status_code == 200
    assert missing.json()["events"] == []

    legacy_exact = client.get(
        "/api/sessions/session_alias_miss/events",
        params={"event_id": private_sentinel},
    )
    assert [event["id"] for event in legacy_exact.json()["events"]] == [public_event_id(1)]


def test_public_event_aliases_disambiguate_legacy_raw_namespace_collisions() -> None:
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        for session_id, first_event_id in (
            ("session-alias-ambiguous", public_event_id(2)),
            ("session-alias-raw-only", public_event_id(9)),
            ("session-alias-same-record", public_event_id(5)),
            ("session-alias-malformed-raw", "cayu_event_legacy"),
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
                        id=first_event_id,
                        type=EventType.SESSION_STARTED,
                        session_id=session_id,
                    ),
                    Event(
                        id=f"{session_id}-completed",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                    ),
                ],
            )
            await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    ambiguous = client.get(
        "/api/sessions/session-alias-ambiguous/events",
        params={"event_id": public_event_id(2)},
    )
    assert ambiguous.status_code == 409
    assert "ambiguous" in ambiguous.json()["detail"]

    replay_ambiguous = client.post(
        "/api/resume",
        json={"prompt": "ignored", "session_id": "session-alias-ambiguous"},
        headers={"Last-Event-ID": (f"session-alias-ambiguous:{public_event_id(2)}")},
    )
    assert replay_ambiguous.status_code == 409

    raw_only = client.get(
        "/api/sessions/session-alias-raw-only/events",
        params={"event_id": public_event_id(9)},
    )
    assert [event["id"] for event in raw_only.json()["events"]] == [public_event_id(3)]

    same_record = client.get(
        "/api/sessions/session-alias-same-record/events",
        params={"event_id": public_event_id(5)},
    )
    assert [event["id"] for event in same_record.json()["events"]] == [public_event_id(5)]

    malformed_raw = client.get(
        "/api/sessions/session-alias-malformed-raw/events",
        params={"event_id": "cayu_event_legacy"},
    )
    assert [event["id"] for event in malformed_raw.json()["events"]] == [public_event_id(7)]


def test_streaming_mutation_id_creates_an_exact_durable_acceptance_event() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session-mutation-identity"
    mutation_id = "mutation-run-identity"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
        headers={"Cayu-Mutation-ID": mutation_id},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    markers = [
        frame["data"]
        for frame in frames
        if frame["data"]["type"] == EventType.SERVER_MUTATION_ACCEPTED
    ]
    assert len(markers) == 1
    marker = markers[0]
    assert marker["session_id"] == session_id
    assert marker["interaction_id"] == frames[0]["data"]["interaction_id"]
    assert marker["interaction_id"] is not None
    assert marker["payload"] == {
        "mutation_id": mutation_id,
        "mutation_kind": "run",
        "accepted_event_id": frames[0]["data"]["id"],
        "accepted_event_sequence": public_event_sequence(frames[0]["data"]["id"]),
        "accepted_event_type": frames[0]["data"]["type"],
    }
    assert frames.index(next(frame for frame in frames if frame["data"] == marker)) > 0

    events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"event_type": EventType.SERVER_MUTATION_ACCEPTED},
    ).json()["events"]
    assert [event["id"] for event in events] == [marker["id"]]
    assert events[0]["interaction_id"] == marker["interaction_id"]


def test_streaming_mutation_id_header_rejects_unsafe_values_before_execution() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session-invalid-mutation-id"},
        headers={"Cayu-Mutation-ID": "invalid mutation id"},
    )

    assert response.status_code == 422
    assert asyncio.run(app.session_store.load_state("session-invalid-mutation-id")) is None


def test_secret_bearing_mutation_id_is_rejected_before_execution() -> None:
    session_id = "sess-secret-id"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor("u"),
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
        headers={"Cayu-Mutation-ID": "mutation-identity"},
    )

    assert response.status_code == 422
    assert "workload secret" in response.json()["detail"]
    assert asyncio.run(app.session_store.load_state(session_id)) is None


def test_explicit_compaction_endpoint_uses_replayable_mutation_contract() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=TranscriptDigestCompactor(),
            max_user_turns=1,
        ),
    )

    async def prepare() -> tuple[int, int]:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-explicit-compact-endpoint",
                    messages=[Message.text("user", "create only")],
                )
            )
        ]
        session = await app.session_store.load("session-explicit-compact-endpoint")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await app.session_store.append_transcript_messages(session.id, transcript)
        persisted_transcript = await app.session_store.load_transcript(session.id)
        return session.run_epoch, len(persisted_transcript)

    run_epoch, transcript_cursor = asyncio.run(prepare())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session-explicit-compact-endpoint"
    body = {
        "idempotency_key": "compact-endpoint-1",
        "expected_run_epoch": run_epoch,
        "expected_transcript_cursor": transcript_cursor,
        "instructions": "Keep decisions.",
        "requested_by": {"subject": "operator@example.com"},
    }
    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/compact",
        json=body,
        headers={"Cayu-Mutation-ID": "mutation-compact-1"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    event_types = [frame["data"]["type"] for frame in frames]
    assert event_types[:3] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
    ]
    accepted_events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"event_type": EventType.SERVER_MUTATION_ACCEPTED},
    ).json()["events"]
    assert len(accepted_events) == 1
    accepted = accepted_events[0]
    assert accepted["payload"] == {
        "mutation_id": "mutation-compact-1",
        "mutation_kind": "session.compact",
        "accepted_event_id": frames[0]["data"]["id"],
        "accepted_event_sequence": public_event_sequence(frames[0]["data"]["id"]),
        "accepted_event_type": EventType.CONTEXT_COMPACTION_STARTED,
    }
    assert frames[0]["data"]["payload"]["actor"] == {
        "subject": "operator@example.com",
        "tenant": None,
        "source": "request",
    }
    assert "/api/sessions/{session_id}/compact" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize(
    ("path", "idempotency_key", "location"),
    [
        ("/api/sessions/missing/compact", "invalid-\x00key", ["body", "idempotency_key"]),
        ("/api/sessions/invalid%00id/compact", "compact-1", ["path", "session_id"]),
    ],
)
def test_explicit_compaction_endpoint_rejects_unpersistable_identifiers(
    path: str,
    idempotency_key: str,
    location: list[str],
) -> None:
    client = TestClient(create_server(CayuApp(enable_logging=False), config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        path,
        json={
            "idempotency_key": idempotency_key,
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 0,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == location


def test_explicit_compaction_endpoint_rejects_unpersistable_nested_text() -> None:
    client = TestClient(create_server(CayuApp(enable_logging=False), config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/sessions/missing/compact",
        json={
            "idempotency_key": "compact-1",
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 0,
            "instructions": "invalid-\x00instructions",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body"]


def test_sse_replay_preserves_canonical_policy_denial_attribution() -> None:
    session_id = "session_policy_denial_replay"
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "push")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_tool_started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={"tool_call_id": "call_1"},
                ),
                Event(
                    id="event_tool_blocked",
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={
                        "tool_name": "exec_command",
                        "tool_call_id": "call_1",
                        "tool_round_id": "round_1",
                        "idempotency_key": "cayu-tool:v1:call_1",
                        "denied_by": "command_policy",
                        "decision": "deny",
                        "reason": "Remote mutation is not allowed.",
                        "result": {
                            "content": "Command denied by policy.",
                            "structured": {"error": "command_denied"},
                            "artifacts": [],
                            "is_error": True,
                        },
                    },
                ),
                Event(
                    id="event_session_completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    blocked = next(
        frame["data"] for frame in frames if frame["data"]["type"] == "tool.call.blocked"
    )
    assert blocked["payload"]["denied_by"] == "command_policy"
    assert blocked["payload"]["decision"] == "deny"
    assert blocked["payload"]["tool_name"] == "exec_command"
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(1),
        public_event_id(2),
        public_event_id(3),
    ]


def _authorized_message_endpoint():
    from cayu.server import AuthContext
    from cayu.sessions.messaging import SessionMessageAccessPolicy

    class MessagePolicy(SessionMessageAccessPolicy):
        def authorize(self, context, *, session_id, session_instance_id, action):
            return (
                context.subject == "operator@example.com"
                and session_id == "session-message-endpoint"
                and action == "enqueue"
            )

    async def authenticate(_request):
        return AuthContext(subject="operator@example.com")

    app = CayuApp(session_message_access_policy=MessagePolicy(), enable_logging=False)
    return app, TestClient(create_server(app, config=ServerConfig.protected(authenticate)))


def test_enqueue_session_message_endpoint_uses_replayable_mutation_contract() -> None:
    app, client = _authorized_message_endpoint()

    async def prepare() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-message-endpoint",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(prepare())
    session_id = "session-message-endpoint"
    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/messages",
        json={
            "idempotency_key": "message-endpoint-1",
            "content": "Please prioritize the failing deployment.",
            "delivery_mode": "next_turn",
        },
        headers={"Cayu-Mutation-ID": "mutation-message-1"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    queued = frames[0]["data"]
    assert queued["type"] == EventType.SESSION_MESSAGE_QUEUED
    assert queued["payload"]["delivery_mode"] == "next_turn"
    assert queued["payload"]["actor"] == {
        "subject": "operator@example.com",
        "tenant": None,
        "source": "http_auth",
    }
    assert "content" not in queued["payload"]

    persisted_events = client.get(f"/api/sessions/{session_id}/events").json()["events"]
    persisted_queued = next(
        event for event in persisted_events if event["type"] == EventType.SESSION_MESSAGE_QUEUED
    )
    assert persisted_queued["id"] == queued["id"]
    accepted = next(
        event for event in persisted_events if event["type"] == EventType.SERVER_MUTATION_ACCEPTED
    )
    assert accepted["payload"] == {
        "mutation_id": "mutation-message-1",
        "mutation_kind": "session.message.enqueue",
        "accepted_event_id": queued["id"],
        "accepted_event_sequence": public_event_sequence(queued["id"]),
        "accepted_event_type": EventType.SESSION_MESSAGE_QUEUED,
    }
    assert "/api/sessions/{session_id}/messages" in client.app.openapi()["paths"]


def test_enqueue_session_message_endpoint_rejects_nonportable_text() -> None:
    _, client = _authorized_message_endpoint()

    response = client.post(
        "/api/sessions/session_1/messages",
        json={
            "idempotency_key": "message-1",
            "content": "hello\u0000",
            "delivery_mode": "next_turn",
        },
    )

    assert response.status_code == 422


def test_run_disconnect_after_http_acceptance_before_first_body_replays_from_start() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_pre_first_disconnect"

    async def exercise_disconnect() -> list[str]:
        messages = await _post_and_disconnect_before_first_body(
            server,
            "/api/run",
            {"prompt": "hello", "session_id": session_id},
        )
        starts = [message for message in messages if message["type"] == "http.response.start"]
        assert [message["status"] for message in starts] == [200]
        assert not any(
            message.get("body") for message in messages if message["type"] == "http.response.body"
        )

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    "detached run did not complete after the observer disconnected"
                )
            await asyncio.sleep(0.01)

        tasks = await task_store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].status is TaskStatus.COMPLETED
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        return [record.event.id for record in records]

    durable_event_ids = asyncio.run(exercise_disconnect())
    assert durable_event_ids

    executed = []

    async def unexpected_run(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_run
    with TestClient(server).stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        replayed = [frame["data"]["id"] for frame in _sse_frames(response) if "data" in frame]

    assert replayed == [
        public_event_id(sequence) for sequence in range(1, len(durable_event_ids) + 1)
    ]
    assert executed == []


def test_existing_session_reconnect_cannot_race_accepted_mutation_transition() -> None:
    app = CayuApp()
    session_id = "session_resume_pre_first_disconnect"
    baseline_id = "event_before_resume"
    executions = 0

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                id=baseline_id,
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())

    async def accepted_resume(request):
        nonlocal executions
        executions += 1
        await app.session_store.transition_status(
            session_id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=SessionStatus.RUNNING,
        )
        resumed = Event(
            id="event_resume_accepted",
            type=EventType.SESSION_RESUMED,
            session_id=session_id,
            agent_name="assistant",
        )
        await app.session_store.append_event(session_id, resumed)
        yield resumed
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)
        completed = Event(
            id="event_resume_completed",
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
            agent_name="assistant",
        )
        await app.session_store.append_event(session_id, completed)
        yield completed

    app.resume = accepted_resume
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def exercise_disconnect() -> None:
        messages = await _post_and_disconnect_before_first_body(
            server,
            "/api/resume",
            {"session_id": session_id, "prompt": "continue"},
        )
        starts = [message for message in messages if message["type"] == "http.response.start"]
        assert [message["status"] for message in starts] == [200]

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("accepted resume did not complete after disconnect")
            await asyncio.sleep(0.01)

    asyncio.run(exercise_disconnect())
    assert executions == 1

    async def unexpected_resume(request):
        raise AssertionError(f"replay re-executed resume for {request.session_id}")
        yield  # pragma: no cover

    app.resume = unexpected_resume
    with TestClient(server).stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored"},
        headers={"Last-Event-ID": f"{session_id}:{baseline_id}"},
    ) as response:
        assert response.status_code == 200
        replayed = [frame["data"]["id"] for frame in _sse_frames(response) if "data" in frame]

    assert replayed == [public_event_id(2), public_event_id(3)]
    assert executions == 1


def test_concurrent_client_run_identity_creates_one_session_and_one_task() -> None:
    class CoordinatedRunStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.claims = 0
            self.both_claiming = asyncio.Event()

        async def create(self, request, *, identity, **kwargs):
            if request.session_id == "session_concurrent_claim":
                self.claims += 1
                if self.claims == 2:
                    self.both_claiming.set()
                await self.both_claiming.wait()
            return await super().create(request, identity=identity, **kwargs)

    store = CoordinatedRunStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=task_store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def submit_concurrently():
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(
                client.post(
                    "/api/run",
                    json={"prompt": "first", "session_id": "session_concurrent_claim"},
                ),
                client.post(
                    "/api/run",
                    json={"prompt": "second", "session_id": "session_concurrent_claim"},
                ),
            )

    responses = asyncio.run(submit_concurrently())
    assert sorted(response.status_code for response in responses) == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json()["detail"] == "Session already exists: session_concurrent_claim"

    tasks = asyncio.run(task_store.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.COMPLETED
    state = asyncio.run(store.load_state("session_concurrent_claim"))
    assert state is not None
    assert state.status is SessionStatus.COMPLETED


def test_run_replay_rejects_malformed_last_event_id_and_unknown_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    malformed = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "not-a-marker"},
    )
    assert malformed.status_code == 422

    for marker in ("missing_session:event_1", "missing_session:"):
        unknown = client.post(
            "/api/run",
            json={"prompt": "hello", "session_id": "missing_session"},
            headers={"Last-Event-ID": marker},
        )
        assert unknown.status_code == 404


def test_run_replay_missing_session_does_not_expose_resolved_private_identity() -> None:
    private_session_id = "private-session-secret"

    class VanishingSessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_state(self, session_id: str):
            if session_id == private_session_id:
                return None
            return await super().load_state(session_id)

    store = VanishingSessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(private_session_id),
        enable_logging=False,
    )

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    public_session_id = app.project_session_id_for_exposure(private_session_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": public_session_id},
        headers={"Last-Event-ID": f"{public_session_id}:"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": f"Session not found: {public_session_id}"}
    assert private_session_id not in response.text


def test_run_replay_rejects_unknown_event_and_mismatched_body_identity() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_marker_validation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            "session_marker_validation",
            Event(
                id="event_seen",
                type=EventType.SESSION_STARTED,
                session_id="session_marker_validation",
                agent_name="assistant",
            ),
        )
        await app.session_store.update_status(
            "session_marker_validation",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_execution
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    unknown_event = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "session_marker_validation:event_missing"},
    )
    mismatched_session = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "session_other:event_seen"},
    )
    malformed_public_alias = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={
            "Last-Event-ID": (
                f"session_marker_validation:cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}"
            )
        },
    )

    assert unknown_event.status_code == 409
    assert "event was not found" in unknown_event.json()["detail"]
    assert mismatched_session.status_code == 422
    assert "does not match" in mismatched_session.json()["detail"]
    assert malformed_public_alias.status_code == 422
    assert "malformed Cayu public event alias" in malformed_public_alias.json()["detail"]
    assert executed == []
    assert client.get("/api/tasks").json() == []


@pytest.mark.parametrize(
    "session_id",
    ["session:colon", " leading-space", "slash/not-allowed", "x" * 129],
)
def test_run_rejects_session_ids_that_are_not_replay_safe(session_id: str) -> None:
    response = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG)).post(
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
    )

    assert response.status_code == 422


def test_run_rejects_duplicate_client_session_id_before_starting_work() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_duplicate",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session_duplicate"},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail.startswith("Session already exists: session_duplicate.")
    assert "app.run creates a new session" in detail
    assert "app.resume(ResumeRequest(" in detail
    assert "pending approval/input" in detail
    assert client.get("/api/tasks").json() == []


def test_replay_of_active_session_times_out_with_structured_error() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_stranded",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status("session_stranded", SessionStatus.RUNNING)
        await app.session_store.append_event(
            "session_stranded",
            Event(
                id="event_seen",
                type=EventType.SESSION_STARTED,
                session_id="session_stranded",
                agent_name="assistant",
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay"},
        headers={"Last-Event-ID": "session_stranded:event_seen"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["kind"] == "observer"
    assert frames[-1]["data"]["code"] == "replay_idle_timeout"
    assert frames[-1]["data"]["retryable"] is True
    assert frames[-1]["data"]["session_id"] == "session_stranded"
    assert frames[-1]["data"]["error_type"] == "TimeoutError"
    assert "session_stranded" in frames[-1]["data"]["error"]


def test_replay_waits_for_terminal_event_after_terminal_status() -> None:
    class DelayedTerminalEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.injected_terminal_race = False
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def query_events(
            self,
            query: EventQuery | None = None,
        ) -> list[EventRecord]:
            records = await super().query_events(query)
            if (
                query is not None
                and query.session_id is not None
                and query.after_sequence is not None
                and query.event_id is None
                and query.event_type is None
                and not query.event_types
                and not self.injected_terminal_race
            ):
                self.injected_terminal_race = True
                session_id = query.session_id
                await self.update_status(session_id, SessionStatus.INTERRUPTED)

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_delayed_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return records

    store = DelayedTerminalEventStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session_replay_terminal_status_race"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_initial_start",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.RUNNING)
        await store.append_event(
            session_id,
            Event(
                id="event_resume_baseline",
                type=EventType.SESSION_RESUMED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_previous_terminal"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(3),
        public_event_id(4),
    ]
    assert frames[-1]["data"]["type"] == EventType.SESSION_INTERRUPTED


def test_replay_recognizes_post_terminal_hook_marker() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_post_terminal_hook"

    async def seed() -> None:
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
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_hook_completed",
                    type=EventType.HOOK_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_hook_completed"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


@pytest.mark.parametrize(
    "hook_event_type",
    [EventType.HOOK_STARTED, EventType.HOOK_COMPLETED, EventType.HOOK_FAILED],
)
def test_replay_does_not_attach_stale_hook_marker_across_operation_start(
    hook_event_type: EventType,
) -> None:
    class DelayedTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def load_state(self, session_id: str):
            state = await super().load_state(session_id)
            if self.terminal_append_task is None:

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_current_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return state

    store = DelayedTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = f"session_replay_stale_{hook_event_type.value}"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_hook",
                    type=hook_event_type,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_previous_terminal",
                        "terminal_event_type": str(EventType.SESSION_INTERRUPTED),
                    },
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_stale_hook"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [public_event_id(4)]


def test_replay_unverified_hook_does_not_erase_observed_terminal_boundary() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_unverified_hook_after_terminal"

    async def seed() -> None:
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
                    id="event_operation_start",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_unverified_hook",
                    type=EventType.HOOK_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_missing_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_operation_start"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(2),
        public_event_id(3),
    ]


def test_replay_does_not_accept_custom_event_as_terminal_lineage() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_forged_terminal_lineage"

    async def seed() -> None:
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
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_custom",
                    type="custom.forged_terminal_lineage",
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_custom"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "replay_idle_timeout"


@pytest.mark.parametrize(
    "post_terminal_event_type",
    [
        EventType.SERVER_MUTATION_ACCEPTED,
        EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
        EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
        EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
    ],
)
def test_replay_recognizes_runtime_post_terminal_marker(
    post_terminal_event_type: EventType,
) -> None:
    app = CayuApp(enable_logging=False)
    session_id = f"session_replay_{post_terminal_event_type.value}"

    async def seed() -> None:
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
                    id="event_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_post_terminal",
                    type=post_terminal_event_type,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_post_terminal"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


def test_replay_cascade_marker_uses_latest_completed_operation_boundary() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_stale_cascade_after_completion"

    async def seed() -> None:
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
                    id="event_previous_interrupt",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_current_completion",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_cascade",
                    type=EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_stale_cascade"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


def test_replay_does_not_attach_stale_cascade_marker_across_operation_start() -> None:
    class DelayedTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def load_state(self, session_id: str):
            state = await super().load_state(session_id)
            if self.terminal_append_task is None:

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_current_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return state

    store = DelayedTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session_replay_stale_cascade_marker"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_cascade",
                    type=EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_stale_cascade"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [public_event_id(4)]


def test_replay_streams_complete_history_in_bounded_pages() -> None:
    app = CayuApp()
    event_count = 1_001

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_replay_bound",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "session_replay_bound",
            [
                Event(
                    id="event_seen",
                    type=EventType.SESSION_STARTED,
                    session_id="session_replay_bound",
                    agent_name="assistant",
                ),
                *[
                    Event(
                        id=f"event_{index}",
                        type="custom.replay",
                        session_id="session_replay_bound",
                        agent_name="assistant",
                    )
                    for index in range(event_count)
                ],
                Event(
                    id="event_replay_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="session_replay_bound",
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status("session_replay_bound", SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay"},
        headers={"Last-Event-ID": "session_replay_bound:event_seen"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert all(frame.get("event") != "error" for frame in frames)
    assert len(frames) == event_count + 1
    event_frames = frames
    assert event_frames[0]["data"]["id"] == public_event_id(2)
    assert event_frames[-2]["data"]["id"] == public_event_id(event_count + 1)
    assert event_frames[-1]["data"]["id"] == public_event_id(event_count + 2)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/api/tool-approvals/resolve",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
                "decision": "approve",
            },
        ),
        (
            "/api/tool-approvals/recover",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/tool-rounds/recover",
            {
                "session_id": "session_approval_replay",
                "round_id": "round_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/user-input/resolve",
            {
                "session_id": "session_approval_replay",
                "input_id": "input_1",
                "answer": "continue",
            },
        ),
        (
            "/api/user-input/recover",
            {
                "session_id": "session_approval_replay",
                "input_id": "input_1",
                "answer": "continue",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/sessions/session_approval_replay/interrupt",
            {},
        ),
        (
            "/api/sessions/session_approval_replay/compact",
            {
                "idempotency_key": "compact-replay",
                "expected_run_epoch": 0,
                "expected_transcript_cursor": 1,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("last_event_id", "expected_event_ids"),
    [
        ("session_approval_replay:event_seen", [public_event_id(2)]),
        (
            "session_approval_replay:",
            [public_event_id(1), public_event_id(2)],
        ),
    ],
)
def test_mutation_routes_replay_without_reexecuting(
    path: str,
    body: dict,
    last_event_id: str,
    expected_event_ids: list[str],
) -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_approval_replay",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "session_approval_replay",
            [
                Event(
                    id="event_seen",
                    type=EventType.SESSION_STARTED,
                    session_id="session_approval_replay",
                    agent_name="assistant",
                ),
                Event(
                    id="event_missed",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="session_approval_replay",
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status("session_approval_replay", SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.resolve_tool_approval = unexpected_execution
    app.recover_tool_approval = unexpected_execution
    app.recover_tool_round = unexpected_execution
    app.resolve_user_input = unexpected_execution
    app.recover_user_input = unexpected_execution
    app.interrupt_session = unexpected_execution
    app.compact_session = unexpected_execution
    app.enqueue_session_message = unexpected_execution
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        path,
        json=body,
        headers={"Last-Event-ID": last_event_id},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in frames] == expected_event_ids
    assert executed == []


def test_session_scoped_replay_rejects_marker_for_different_session() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_requested",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
        headers={"Last-Event-ID": "session_other:event_seen"},
    )

    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]


def test_create_server_startup_recovery_composes_user_lifespan() -> None:
    app = CayuApp()
    calls: list[str] = []
    requests = []

    @asynccontextmanager
    async def user_lifespan(server):
        calls.append("user_start")
        yield
        calls.append("user_stop")

    async def plan_recovery(request):
        calls.append("plan_recovery")
        requests.append(request)
        return RecoveryPlan(
            plan_id="startup-plan-one",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
        )

    async def execute_recovery(request):
        calls.append("execute_recovery")
        assert isinstance(request, RecoveryExecutionRequest)
        assert request.execution_id == f"server-startup:{request.plan.plan_id}"
        assert request.max_concurrency == 8

    async def recover_event_side_effects(*, limit=1000):
        calls.append("recover_event_side_effects")
        assert limit == 1000
        return []

    async def drain_background_interruptions(*, timeout_s):
        calls.append("drain")
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        return True

    async def drain_provider_operation_cancellations(*, timeout_s):
        calls.append("provider_cancellation_drain")
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        return True

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        calls.append("resume_cascades")
        assert interrupting_inactive_for_seconds == 60
        return 0

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    app.recover_persisted_event_side_effects = recover_event_side_effects
    app.drain_background_interruptions = drain_background_interruptions
    app.drain_provider_operation_cancellations = drain_provider_operation_cancellations
    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                    SessionStatus.INTERRUPTED,
                },
                recovery_inactive_after_seconds=60,
            )
        ),
        fastapi_options={"lifespan": user_lifespan},
    )

    with TestClient(server):
        assert calls == [
            "user_start",
            "recover_event_side_effects",
            "plan_recovery",
            "execute_recovery",
            "resume_cascades",
        ]

    assert calls == [
        "user_start",
        "recover_event_side_effects",
        "plan_recovery",
        "execute_recovery",
        "resume_cascades",
        "drain",
        "provider_cancellation_drain",
        "user_stop",
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.selection.statuses == {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }
    assert request.selection.inactive_for_seconds == 60


def test_create_server_startup_recovery_consumes_every_cursor_page() -> None:
    app = CayuApp()
    requests = []
    server_is_ready = threading.Event()
    continuation_started = threading.Event()
    continuation_finished = threading.Event()

    async def plan_recovery(request):
        requests.append(request)
        if request.selection.cursor is None:
            next_cursor = "startup-page-2"
        elif request.selection.cursor == "startup-page-2":
            continuation_started.set()
            for _ in range(100):
                if server_is_ready.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Startup recovery continued before server readiness.")
            next_cursor = "startup-page-3"
        else:
            assert request.selection.cursor == "startup-page-3"
            continuation_finished.set()
            next_cursor = None
        return RecoveryPlan(
            plan_id=f"startup-plan-{len(requests)}",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
            next_cursor=next_cursor,
        )

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
                recovery_inactive_after_seconds=60,
            )
        ),
    )

    with TestClient(server):
        server_is_ready.set()
        assert continuation_started.wait(timeout=10.0)
        assert continuation_finished.wait(timeout=2.0)

    assert [request.selection.cursor for request in requests] == [
        None,
        "startup-page-2",
        "startup-page-3",
    ]
    assert all(request.selection.statuses == {SessionStatus.INTERRUPTED} for request in requests)
    assert len({request.selection.inactive_for_seconds for request in requests}) == 1


def test_create_server_stops_incomplete_recovery_continuation_on_shutdown() -> None:
    app = CayuApp()
    continuation_started = threading.Event()
    continuation_cancelled = threading.Event()

    async def plan_recovery(request):
        if request.selection.cursor is None:
            return RecoveryPlan(
                plan_id="startup-plan-one",
                created_at=datetime.now(UTC),
                request=request,
                items=(),
                inspected_session_count=0,
                next_cursor="startup-page-2",
            )
        continuation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            continuation_cancelled.set()
            raise

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
            )
        ),
    )

    with TestClient(server):
        assert continuation_started.wait(timeout=10.0)

    assert continuation_cancelled.wait(timeout=2.0)


def test_create_server_background_startup_recovery_rejects_cursor_cycles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = CayuApp()
    cursors = []
    continuation_finished = threading.Event()

    async def plan_recovery(request):
        cursors.append(request.selection.cursor)
        if request.selection.cursor is not None:
            continuation_finished.set()
        return RecoveryPlan(
            plan_id=f"startup-plan-{len(cursors)}",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
            next_cursor="repeated-startup-cursor",
        )

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
            )
        ),
    )

    with caplog.at_level(logging.ERROR, logger="cayu.server"), TestClient(server):
        assert continuation_finished.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while (
            "startup recovery returned a repeated cursor" not in caplog.text
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

    assert cursors == [None, "repeated-startup-cursor"]
    assert "startup recovery returned a repeated cursor" in caplog.text


def test_create_server_drains_persisted_event_side_effect_backlog() -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_backlog",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        events = [
            Event(type="custom.backlog", session_id=session.id, payload={"index": index})
            for index in range(1001)
        ]
        await store.append_events(session.id, events)
        return store, events

    store, events = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        assert [event.id for event in sink.events] == [
            public_event_id(index) for index in range(1, len(events) + 1)
        ]


@pytest.mark.parametrize("adapter", ["create_server", "mount_cayu"])
def test_server_startup_bounds_non_returning_side_effect_recovery(
    adapter: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"bounded_startup_{adapter}",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.pending", session_id=session.id)
        await store.append_event(session.id, event)
        return store, event

    store, event = asyncio.run(prepare())
    sink = NeverReturningEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    if adapter == "create_server":
        server = create_server(
            app,
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(event_side_effect_startup_timeout_seconds=0.01)
            ),
        )
        health_path = "/api/health"
    else:
        server = FastAPI()
        mount_cayu(
            server,
            app,
            path="/cayu",
            dashboard=False,
            access=OpenAccess(),
            event_side_effect_startup_timeout_seconds=0.01,
        )
        health_path = "/cayu/api/health"

    started = time.monotonic()
    with (
        caplog.at_level(logging.WARNING, logger="cayu.server"),
        TestClient(server) as client,
    ):
        assert client.get(health_path).json() == {"ok": True}
    assert time.monotonic() - started < 2.0

    delivery = asyncio.run(
        store.get_persisted_event_side_effect_delivery(
            session_id=event.session_id,
            event_id=event.id,
        )
    )
    assert sink.cancelled
    assert delivery is not None
    assert delivery.status is PersistedEventSideEffectStatus.LEASED
    assert "startup recovery exceeded" in caplog.text


def test_create_server_retries_recovery_timeout_error(monkeypatch, caplog) -> None:
    app = CayuApp()
    attempts = 0
    monkeypatch.setattr("cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS", 0.01)

    async def fail_recovery(*, limit):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("private session store timeout")
        return []

    app.recover_persisted_event_side_effects = fail_recovery
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def wait_for_retry():
        status = server.state.cayu_event_side_effect_recovery
        async with asyncio.timeout(5):
            while status.sweep_successes == 0:
                await asyncio.sleep(0.01)
        assert status.sweep_failures == 1
        assert status.consecutive_failures == 0
        assert status.last_error_at is not None
        assert "private session store timeout" not in status.model_dump_json()

    with TestClient(server) as client:
        assert client.portal is not None
        client.portal.call(wait_for_retry)
    assert "startup sweep timed out" in caplog.text
    assert "private session store timeout" not in caplog.text


def test_create_server_retries_crash_claim_after_lease_expiry(monkeypatch) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_lease",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.leased", session_id=session.id)
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(lease_seconds=0.2)
        assert claim is not None
        return store, event

    store, _event = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS",
        0.01,
    )

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        deadline = time.monotonic() + 1.0
        while not sink.events and time.monotonic() < deadline:
            time.sleep(0.01)

    assert [recovered.id for recovered in sink.events] == [public_event_id(1)]


def test_mount_cayu_retries_crash_claim_after_lease_expiry(monkeypatch) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="mounted_side_effect_lease",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.leased", session_id=session.id)
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(lease_seconds=0.2)
        assert claim is not None
        return store, event

    store, _event = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    server = FastAPI()
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS",
        0.01,
    )
    mount_cayu(server, app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        deadline = time.monotonic() + 1.0
        while not sink.events and time.monotonic() < deadline:
            time.sleep(0.01)

    assert [recovered.id for recovered in sink.events] == [public_event_id(1)]


def test_create_server_defers_transient_sink_retry_to_periodic_loop() -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_transient_sink",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.transient", session_id=session.id)
        await store.append_event(session.id, event)
        return store

    store = asyncio.run(prepare())
    app = CayuApp(
        session_store=store,
        event_sinks=[FailOnceEventSink()],
        enable_logging=False,
    )

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        deliveries = asyncio.run(store.list_persisted_event_side_effect_deliveries())

    assert [(delivery.status.value, delivery.attempts) for delivery in deliveries] == [
        ("failed", 1)
    ]
    assert deliveries[0].next_attempt_at is not None
    assert deliveries[0].next_attempt_at > deliveries[0].updated_at


def test_create_server_drains_cascades_when_startup_recovery_fails() -> None:
    app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        assert type(interrupting_inactive_for_seconds) is int
        assert interrupting_inactive_for_seconds >= 0
        calls.append("recover")
        raise RuntimeError("recovery failed after scheduling work")

    async def drain_background_interruptions(*, timeout_s):
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        calls.append("drain")
        return True

    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    app.drain_background_interruptions = drain_background_interruptions
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    with (
        pytest.raises(RuntimeError, match="recovery failed after scheduling work"),
        TestClient(server),
    ):
        pass

    assert calls == ["recover", "drain"]


@pytest.mark.parametrize("value", [True, 0, -1, float("inf")])
def test_create_server_rejects_invalid_interruption_shutdown_grace(value) -> None:
    with pytest.raises(ValueError, match="interruption_shutdown_grace_seconds"):
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(interruption_shutdown_grace_seconds=value)
            ),
        )


@pytest.mark.parametrize("value", [True, 0, -1, float("inf")])
def test_create_server_rejects_invalid_knowledge_publication_shutdown_grace(value) -> None:
    with pytest.raises(ValueError, match="knowledge_publication_shutdown_grace_seconds"):
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(knowledge_publication_shutdown_grace_seconds=value)
            ),
        )


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("adapter", ["create_server", "mount_cayu"])
def test_server_rejects_invalid_event_side_effect_startup_timeout(
    adapter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="event_side_effect_startup_timeout_seconds"):
        if adapter == "create_server":
            create_server(
                CayuApp(),
                config=ServerConfig.local_development(
                    lifecycle=ServerLifecycleConfig(
                        event_side_effect_startup_timeout_seconds=value  # type: ignore[arg-type]
                    )
                ),
            )
        else:
            mount_cayu(
                FastAPI(),
                CayuApp(),
                dashboard=False,
                access=OpenAccess(),
                event_side_effect_startup_timeout_seconds=value,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_mount_cayu_rejects_invalid_interruption_recovery_inactivity(value) -> None:
    with pytest.raises(ValueError, match="interruption_recovery_inactive_after_seconds"):
        mount_cayu(
            FastAPI(),
            CayuApp(),
            dashboard=False,
            access=OpenAccess(),
            interruption_recovery_inactive_after_seconds=value,
        )


def test_client_disconnect_does_not_cancel_detached_run() -> None:
    import time

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_id = None
    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                session_id = json.loads(line[len("data:") :].strip())["session_id"]
                break  # disconnect after the first event

    assert session_id is not None
    # The run is driven by a detached pump, so it still finishes after the disconnect.
    deadline = time.monotonic() + 10
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}").json()["status"]
        if status == "completed":
            break
        time.sleep(0.05)
    assert status == "completed"


def test_run_stream_failure_emits_terminal_structured_error_frame() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    async def broken_run(request):
        yield event_with_durable_sequence(
            Event(
                type=EventType.SESSION_STARTED,
                session_id=request.session_id,
                agent_name=request.agent_name,
            ),
            1,
        )
        raise RuntimeError("run exploded with secret-token " + "x" * SSE_ERROR_TEXT_MAX_BYTES)

    app.run = broken_run

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames
    error_frame = frames[-1]
    assert error_frame.get("event") == "error"
    data = error_frame["data"]
    assert data["type"] == "stream.error"
    assert data["kind"] == "runtime"
    assert data["code"] == "runtime_failed"
    assert data["error_type"] == "RuntimeError"
    assert data["retryable"] is False
    assert data["session_id"].startswith("session-")
    assert "secret-token" not in data["error"]
    assert REDACTED_SECRET in data["error"]
    assert data["error"].endswith("... [truncated]")
    assert len(data["error"].encode("utf-8")) <= SSE_ERROR_TEXT_MAX_BYTES


def test_interrupt_stream_uses_same_typed_redacted_runtime_error_contract() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_stream_error",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_stream_error", SessionStatus.RUNNING
        )

    asyncio.run(seed())

    async def broken_interrupt(request):
        yield event_with_durable_sequence(
            Event(
                id="event_interrupted",
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="assistant",
            ),
            1,
        )
        raise RuntimeError("interrupt failed with secret-token")

    app.interrupt_session = broken_interrupt
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_stream_error/interrupt",
        json={"reason": "operator request"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[0]["data"]["id"] == public_event_id(1)
    error_frame = frames[-1]
    assert error_frame["event"] == "error"
    assert error_frame["data"]["kind"] == "runtime"
    assert error_frame["data"]["code"] == "runtime_failed"
    assert error_frame["data"]["retryable"] is False
    assert error_frame["data"]["session_id"] == "session_interrupt_stream_error"
    assert "secret-token" not in error_frame["data"]["error"]
    assert REDACTED_SECRET in error_frame["data"]["error"]


class AskUserProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1", name="ask_user", arguments={"question": "which env?"}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _ask_user_client() -> TestClient:
    app = CayuApp()
    app.register_provider(AskUserProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
    return TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))


def test_server_resolve_user_input_resumes_paused_session() -> None:
    client = _ask_user_client()
    run_events = _sse_events(client, "/api/run", {"prompt": "deploy"})
    awaiting = next(e for e in run_events if e["type"] == "session.awaiting_user_input")
    session_id = awaiting["session_id"]
    input_id = awaiting["payload"]["input_id"]

    resolved = _sse_events(
        client,
        "/api/user-input/resolve",
        {"session_id": session_id, "input_id": input_id, "answer": "staging"},
    )
    assert resolved[-1]["type"] == "session.completed"
    tool_completed = next(
        e for e in resolved if e["type"] == "tool.call.completed" and e["tool_name"] == "ask_user"
    )
    assert tool_completed["payload"]["result"]["content"] == "staging"


def test_server_resolve_user_input_unknown_session_returns_404() -> None:
    client = _ask_user_client()
    response = client.post(
        "/api/user-input/resolve",
        json={"session_id": "missing", "input_id": "x", "answer": "y"},
    )
    assert response.status_code == 404


def test_server_recover_user_input_route_is_registered() -> None:
    client = _ask_user_client()
    # Unknown session → 404 (route exists and validates the session before streaming).
    response = client.post(
        "/api/user-input/recover",
        json={
            "session_id": "missing",
            "input_id": "x",
            "answer": "y",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "recovered",
        },
    )
    assert response.status_code == 404


def test_server_recover_tool_round_route_is_registered() -> None:
    client = _ask_user_client()
    # Unknown session → 404 (route exists and validates the session before streaming).
    response = client.post(
        "/api/tool-rounds/recover",
        json={
            "session_id": "missing",
            "round_id": "round_1",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "recovered",
        },
    )
    assert response.status_code == 404


def test_oversized_prompt_reports_the_size_limit() -> None:
    from cayu.server.contracts import MAX_CONTROL_PLANE_PROMPT_BYTES

    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/run", json={"prompt": "x" * (MAX_CONTROL_PLANE_PROMPT_BYTES + 1)})

    assert MAX_CONTROL_PLANE_PROMPT_BYTES > 64 * 1024
    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            f"prompt exceeds the maximum encoded size of {MAX_CONTROL_PLANE_PROMPT_BYTES} bytes."
        )
    }
    assert "xxxx" not in response.text

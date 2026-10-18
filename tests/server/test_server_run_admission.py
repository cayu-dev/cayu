"""HTTP server run admission contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    OneShotProvider,
    UsageProvider,
    _price_book_payload,
    _sse_frames,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import DEFAULT_MAX_STEPS, MAX_STEPS
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime._continuation_task_failure import runtime_task_failure_identity_from_task
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.server.routes import _log_mutation_acceptance_failure
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


class UnavailableTargetProvider(OneShotProvider):
    def __init__(self) -> None:
        self.dispatches = 0

    def preflight_model_target(self, *, model: str) -> None:
        raise RuntimeError(f"model target {model!r} is not configured")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.dispatches += 1
        async for event in super().stream(request):
            yield event


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
    assert captured_explicitness == [True, True]
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

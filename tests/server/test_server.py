from __future__ import annotations

# ruff: noqa: E402
import asyncio
import base64
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    ArtifactScope,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    LocalArtifactStore,
    Message,
    MessageRole,
    SecretRedactor,
    Task,
    TaskCreate,
    TaskStatus,
    TextPart,
    ThinkingPart,
    UserInputTool,
    WorkspaceBinding,
    default_price_book,
)
from cayu.artifacts import ArtifactListResult, ArtifactMetadata, ArtifactReadResult, ArtifactStore
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CheckpointCompactionContextPolicy,
    EventQuery,
    EventRecord,
    InMemorySessionStore,
    InterruptSessionRequest,
    PendingActionListResult,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    TranscriptDigestCompactor,
)
from cayu.server import create_server, mount_cayu, mount_dashboard
from cayu.server.routes import (
    _accepted_event_stream_response,
    _detached_event_stream_response,
    _next_replay_poll_interval,
)
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SSE_OBSERVER_MAX_BYTES,
    SSE_OBSERVER_MAX_FRAMES,
    SSE_REPLAY_PAGE_EVENTS,
    SSE_SEND_TIMEOUT_SECONDS,
)


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


class CountingArtifactStore(ArtifactStore):
    id = "counting-artifacts"

    async def put_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        raise NotImplementedError

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        raise FileNotFoundError(artifact_id)

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        requested = min(limit or 0, 10_500)
        artifacts = tuple(
            ArtifactMetadata(
                id=f"artifact_{index:05d}",
                filename=f"artifact-{index:05d}.txt",
                content_type="text/plain",
                size_bytes=0,
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="local-review",
            )
            for index in range(requested)
        )
        return ArtifactListResult(
            artifacts=artifacts,
            total_count=20_000,
            truncated=True,
        )

    async def delete(self, artifact_id: str) -> None:
        return None


class InvalidArtifactDataStore(CountingArtifactStore):
    id = "invalid-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        metadata = ArtifactMetadata.model_construct(
            id=artifact_id,
            filename="invalid.txt",
            content_type="text/\ud800",
            size_bytes=0,
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            metadata={},
        )
        return ArtifactReadResult(metadata=metadata, content=b"", total_bytes=0)


class WrongArtifactDataStore(CountingArtifactStore):
    id = "wrong-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        content = b"different artifact content"
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id="different-artifact",
                filename="different.txt",
                content_type="text/plain",
                size_bytes=len(content),
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            ),
            content=content,
            total_bytes=len(content),
        )


class OverreadArtifactDataStore(CountingArtifactStore):
    id = "overread-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        content = b"store ignored the requested limit"
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id=artifact_id,
                filename="overread.txt",
                content_type="text/plain",
                size_bytes=len(content),
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            ),
            content=content,
            total_bytes=len(content),
        )


class UnavailableArtifactStore(CountingArtifactStore):
    id = "unavailable-artifacts"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        raise PermissionError("Artifact backend is unavailable.")

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        raise PermissionError("Artifact backend is unavailable.")


class InvalidArtifactListStore(CountingArtifactStore):
    id = "invalid-artifact-list"

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        return cast("ArtifactListResult", None)


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _price_book_payload(
    *,
    provider_name: str = "fake",
    model: str = "fake-model",
    input_per_million: str = "1",
    output_per_million: str = "1",
    cache_read_input_per_million: str | None = None,
    standard: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    tier: dict[str, object] = {
        "input_per_million": input_per_million,
        "output_per_million": output_per_million,
    }
    if cache_read_input_per_million is not None:
        tier["cache_read_input_per_million"] = cache_read_input_per_million
    return {
        "price_book_version": "test",
        "generated_at": "2026-07-13",
        "prices": [
            {
                "provider_name": provider_name,
                "model": model,
                "schedules": [
                    {
                        "pricing": {"standard": standard or [tier]},
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


def test_server_uses_app_task_store_for_runs_and_task_list() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(
        create_server(
            app,
            dev=True,
            dashboard_config={
                "apiBaseUrl": "/ignored",
                "priceBook": _price_book_payload(output_per_million="3"),
            },
        )
    )

    assert client.get("/").status_code == 404

    dashboard = client.get("/cayu/")
    assert dashboard.status_code == 200
    assert "root" in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/api"' in dashboard.text
    assert '"priceBook":{"price_book_version":"test"' in dashboard.text

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    tasks = client.get("/api/tasks").json()
    assert len(tasks) == 1
    assert tasks[0]["type"] == "run"
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["worker_id"] is None
    assert tasks[0]["lease_expires_at"] is None


def test_server_dashboard_accepts_default_price_book_config() -> None:
    app = CayuApp()
    price_book = default_price_book()
    client = TestClient(
        create_server(
            app,
            dev=True,
            dashboard_config={"priceBook": price_book},
        )
    )

    dashboard = client.get("/cayu/")

    assert dashboard.status_code == 200
    assert f'"price_book_version":"{price_book.price_book_version}"' in dashboard.text
    assert '"prices":[' in dashboard.text


def test_server_run_rejection_before_session_creates_no_task() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))

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


def test_server_run_failure_before_acceptance_is_generic_and_creates_no_task(caplog) -> None:
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor("secret-token"),
    )

    async def broken_run(request):
        raise OSError("storage failed with secret-token")
        yield  # pragma: no cover

    app.run = broken_run
    client = TestClient(create_server(app, dev=True))

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


def test_server_run_task_setup_failure_finalizes_claimed_session(caplog) -> None:
    class FailingTaskStore(InMemoryTaskStore):
        async def create_running_task(self, request):
            raise OSError("task store unavailable with secret-token")

    app = CayuApp(
        task_store=FailingTaskStore(),
        secret_redactor=SecretRedactor("secret-token"),
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))
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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))
    session_id = "session_factory_failure"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        events = [frame["data"] for frame in _sse_frames(response) if "data" in frame]

    assert [event["type"] for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.TASK_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.TASK_FAILED,
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
    client = TestClient(create_server(app, dev=True))
    session_id = "session_binding_failure"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        events = [frame["data"] for frame in _sse_frames(response) if "data" in frame]

    assert [event["type"] for event in events] == [
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.TASK_STARTED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.TASK_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    tasks = asyncio.run(task_store.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.FAILED
    assert tasks[0].session_id == session_id
    assert tasks[0].error == {
        "message": "binding failed",
        "type": "RuntimeError",
        "session_id": session_id,
    }


def test_server_exposes_agent_environment_and_artifact_inventory(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="fake-model",
            metadata={"team": "platform"},
            provider_options={"temperature": 0},
            system_prompt="Review runtime state.",
        ),
        tools=[UserInputTool()],
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"tenant": "test"}),
            artifact_store=artifact_store,
            workspace_instructions="Use local workspace instructions.",
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment log\nstatus=ok\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"source": "test"},
        )
    )

    client = TestClient(create_server(app, dev=True))

    agents = client.get("/api/agents")
    assert agents.status_code == 200
    agents_body = agents.json()
    assert agents_body["total_count"] == 1
    assert agents_body["agents"][0]["name"] == "reviewer"
    assert agents_body["agents"][0]["metadata"] == {"team": "platform"}
    assert agents_body["agents"][0]["has_system_prompt"] is True
    assert [tool["name"] for tool in agents_body["agents"][0]["tools"]] == ["ask_user"]

    environments = client.get("/api/environments")
    assert environments.status_code == 200
    environments_body = environments.json()
    assert environments_body["total_count"] == 1
    assert environments_body["environments"][0]["name"] == "local-review"
    assert environments_body["environments"][0]["artifact_store_id"] == "test-artifacts"
    assert environments_body["environments"][0]["workspace_instructions"] == "inline"

    artifacts = client.get("/api/artifacts", params={"session_id": "sess_inventory"})
    assert artifacts.status_code == 200
    artifacts_body = artifacts.json()
    assert artifacts_body["total_count"] == 1
    assert artifacts_body["artifacts"][0]["id"] == artifact.id
    assert artifacts_body["artifacts"][0]["artifact_store_id"] == "test-artifacts"
    assert artifacts_body["artifacts"][0]["metadata"] == {"source": "test"}

    artifacts_by_agent = client.get("/api/artifacts", params={"agent_name": "reviewer"})
    assert artifacts_by_agent.status_code == 200
    artifacts_by_agent_body = artifacts_by_agent.json()
    assert artifacts_by_agent_body["total_count"] == 1
    assert artifacts_by_agent_body["artifacts"][0]["id"] == artifact.id

    artifacts_by_other_agent = client.get("/api/artifacts", params={"agent_name": "other"})
    assert artifacts_by_other_agent.status_code == 200
    artifacts_by_other_agent_body = artifacts_by_other_agent.json()
    assert artifacts_by_other_agent_body["total_count"] == 0
    assert artifacts_by_other_agent_body["artifacts"] == []

    read = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "test-artifacts", "max_bytes": 10},
    )
    assert read.status_code == 200
    read_body = read.json()
    assert read_body["artifact"]["id"] == artifact.id
    assert read_body["preview_base64"] == base64.b64encode(b"deployment").decode()
    assert read_body["text_preview"] == "deployment"
    assert read_body["total_bytes"] == len(b"deployment log\nstatus=ok\n")
    assert read_body["truncated"] is True

    json_with_charset = asyncio.run(
        artifact_store.put_bytes(
            b'{"status":"ok"}',
            filename="status.json",
            content_type="application/json; charset=utf-8",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    json_preview = client.get(
        f"/api/artifacts/{json_with_charset.id}",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert json_preview.status_code == 200
    assert json_preview.json()["text_preview"] == '{"status":"ok"}'

    malformed_without_store = client.get("/api/artifacts/not-a-local-artifact-id")
    assert malformed_without_store.status_code == 404
    assert malformed_without_store.json()["detail"] == "Artifact not found"

    malformed_with_store = client.get(
        "/api/artifacts/not-a-local-artifact-id",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert malformed_with_store.status_code == 404
    assert malformed_with_store.json()["detail"] == "Artifact not found"

    padded_id = client.get(
        f"/api/artifacts/%20{artifact.id}%20",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert padded_id.status_code == 404
    assert padded_id.json()["detail"] == "Artifact not found"


def test_artifact_content_endpoint_serves_bounded_downloads_safely(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"tenant": "test"}),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment log\nstatus=ok\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"source": "test"},
        )
    )

    client = TestClient(create_server(app, dev=True))

    missing_store = client.get(f"/api/artifacts/{artifact.id}/content")
    assert missing_store.status_code == 422

    blank_store = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "   "},
    )
    assert blank_store.status_code == 422

    padded_store = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": " test-artifacts "},
    )
    assert padded_store.status_code == 422
    assert "must not start or end with whitespace" in padded_store.json()["detail"]

    content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert content.status_code == 200
    assert content.content == b"deployment log\nstatus=ok\n"
    assert content.headers["content-type"].startswith("text/plain")
    assert content.headers["x-cayu-artifact-id"] == artifact.id
    assert content.headers["x-cayu-artifact-store-id"] == "test-artifacts"
    assert content.headers["content-disposition"].startswith('attachment; filename="deploy.log"')
    assert content.headers["cache-control"] == "private, no-store"

    invalid_id = client.get(
        "/api/artifacts/not-a-local-artifact-id/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert invalid_id.status_code == 404
    assert invalid_id.json()["detail"] == "Artifact not found"

    padded_id = client.get(
        f"/api/artifacts/%20{artifact.id}%20/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert padded_id.status_code == 404
    assert padded_id.json()["detail"] == "Artifact not found"

    for malformed_id in (f"art_{'a' * 300}", "art_%00x", "art_%0Ax"):
        malformed_response = client.get(
            f"/api/artifacts/{malformed_id}/content",
            params={"artifact_store_id": "test-artifacts"},
        )
        assert malformed_response.status_code == 404
        assert malformed_response.json()["detail"] == "Artifact not found"

    oversized_content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts", "max_bytes": 10},
    )
    assert oversized_content.status_code == 413
    assert "exceeds the requested max_bytes" in oversized_content.json()["detail"]

    inline_content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts", "disposition": "inline"},
    )
    assert inline_content.status_code == 200
    assert inline_content.headers["content-disposition"].startswith('inline; filename="deploy.log"')
    assert inline_content.headers["x-content-type-options"] == "nosniff"

    html_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"<script>alert('no inline')</script>",
            filename="unsafe.html",
            content_type="text/html",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    html_content = client.get(
        f"/api/artifacts/{html_artifact.id}/content",
        params={
            "artifact_store_id": "test-artifacts",
            "disposition": "inline",
        },
    )
    assert html_content.status_code == 200
    assert html_content.headers["content-disposition"].startswith(
        'attachment; filename="unsafe.html"'
    )
    assert html_content.headers["x-content-type-options"] == "nosniff"

    svg_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>",
            filename="unsafe.svg",
            content_type="image/svg+xml",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    svg_content = client.get(
        f"/api/artifacts/{svg_artifact.id}/content",
        params={
            "artifact_store_id": "test-artifacts",
            "disposition": "inline",
        },
    )
    assert svg_content.status_code == 200
    assert svg_content.headers["content-disposition"].startswith(
        'attachment; filename="unsafe.svg"'
    )
    assert svg_content.headers["x-content-type-options"] == "nosniff"

    with pytest.raises(ValueError, match="control characters"):
        asyncio.run(
            artifact_store.put_bytes(
                b"bad content type",
                filename="bad-content-type.txt",
                content_type="text/plain\r\nX-Bad: y",
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            )
        )
    with pytest.raises(ValueError, match="surrogate code points"):
        asyncio.run(
            artifact_store.put_bytes(
                b"bad filename",
                filename="bad\ud800.txt",
                content_type="text/plain",
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            )
        )
    unsafe_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"unsafe filename",
            filename="bad/path\r\nX: y.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    unsafe_content = client.get(
        f"/api/artifacts/{unsafe_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert unsafe_content.status_code == 200
    assert 'filename="bad_path__X: y.txt"' in unsafe_content.headers["content-disposition"]
    assert "bad%2Fpath" not in unsafe_content.headers["content-disposition"]
    assert "\r" not in unsafe_content.headers["content-disposition"]
    assert "\n" not in unsafe_content.headers["content-disposition"]
    assert "%0D" not in unsafe_content.headers["content-disposition"]
    assert "%0A" not in unsafe_content.headers["content-disposition"]

    bidi_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"unicode filename controls",
            filename="report\u202efdp\u2066\u2069.exe",
            content_type="application/octet-stream",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    bidi_filename_content = client.get(
        f"/api/artifacts/{bidi_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert bidi_filename_content.status_code == 200
    bidi_disposition = bidi_filename_content.headers["content-disposition"]
    assert 'filename="report_fdp__.exe"' in bidi_disposition
    assert "filename*=UTF-8''report_fdp__.exe" in bidi_disposition
    assert "%E2%80%AE" not in bidi_disposition
    assert "%E2%81%A6" not in bidi_disposition
    assert "%E2%81%A9" not in bidi_disposition

    app.register_environment(
        Environment(
            EnvironmentSpec(name="invalid-artifact-environment"),
            artifact_store=InvalidArtifactDataStore(),
        )
    )
    invalid_store_data = client.get(
        "/api/artifacts/invalid/content",
        params={"artifact_store_id": "invalid-artifact-data"},
    )
    assert invalid_store_data.status_code == 500
    assert invalid_store_data.json() == {"detail": "Artifact store returned invalid artifact data."}

    app.register_environment(
        Environment(
            EnvironmentSpec(name="wrong-artifact-environment"),
            artifact_store=WrongArtifactDataStore(),
        )
    )
    wrong_store_data = client.get(
        "/api/artifacts/requested/content",
        params={"artifact_store_id": "wrong-artifact-data"},
    )
    assert wrong_store_data.status_code == 500
    assert wrong_store_data.json() == {"detail": "Artifact store returned invalid artifact data."}

    app.register_environment(
        Environment(
            EnvironmentSpec(name="overread-artifact-environment"),
            artifact_store=OverreadArtifactDataStore(),
        )
    )
    overread_store_data = client.get(
        "/api/artifacts/requested/content",
        params={"artifact_store_id": "overread-artifact-data", "max_bytes": 1},
    )
    assert overread_store_data.status_code == 500
    assert overread_store_data.json() == {
        "detail": "Artifact store returned invalid artifact data."
    }

    app.register_environment(
        Environment(
            EnvironmentSpec(name="unavailable-artifact-environment"),
            artifact_store=UnavailableArtifactStore(),
        )
    )
    for unavailable_path in (
        "/api/artifacts/requested",
        "/api/artifacts/requested/content",
    ):
        unavailable_store = client.get(
            unavailable_path,
            params={"artifact_store_id": "unavailable-artifacts"},
        )
        assert unavailable_store.status_code == 503
        assert unavailable_store.json() == {"detail": "Artifact store is unavailable."}
        assert unavailable_store.headers["content-type"].startswith("application/json")

    unavailable_list = client.get(
        "/api/artifacts",
        params={"artifact_store_id": "unavailable-artifacts"},
    )
    assert unavailable_list.status_code == 503
    assert unavailable_list.json() == {"detail": "Artifact store is unavailable."}
    assert unavailable_list.headers["content-type"].startswith("application/json")

    app.register_environment(
        Environment(
            EnvironmentSpec(name="invalid-artifact-list-environment"),
            artifact_store=InvalidArtifactListStore(),
        )
    )
    invalid_list = client.get(
        "/api/artifacts",
        params={"artifact_store_id": "invalid-artifact-list"},
    )
    assert invalid_list.status_code == 500
    assert invalid_list.json() == {"detail": "Artifact store returned invalid artifact data."}
    assert invalid_list.headers["content-type"].startswith("application/json")

    long_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"bounded header",
            filename=f"{'a' * 20_000}.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    bounded_header = client.get(
        f"/api/artifacts/{long_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert bounded_header.status_code == 200
    disposition_header = bounded_header.headers["content-disposition"]
    assert len(disposition_header.encode("latin-1")) < 2048
    assert ".txt" in disposition_header

    symlink_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"artifact-content",
            filename="symlink.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    outside_content = tmp_path / "outside-secret.txt"
    outside_content.write_bytes(b"host-secret-data")
    content_path = artifact_store.root / symlink_artifact.id / "content"
    content_path.unlink()
    try:
        content_path.symlink_to(outside_content)
    except OSError:
        pass
    else:
        symlink_content = client.get(
            f"/api/artifacts/{symlink_artifact.id}/content",
            params={"artifact_store_id": "test-artifacts"},
        )
        assert symlink_content.status_code == 500
        assert symlink_content.json() == {
            "detail": "Artifact store returned invalid artifact data."
        }
        assert symlink_content.content != outside_content.read_bytes()


def test_server_control_plane_inventory_redacts_configured_secrets(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="fake-model",
            metadata={"note": "agent secret-token"},
            provider_options={"header": "Bearer secret-token"},
        ),
        tools=[UserInputTool()],
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"note": "env secret-token"}),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment secret-token\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"note": "artifact secret-token"},
        )
    )

    client = TestClient(create_server(app, dev=True))

    agent = client.get("/api/agents").json()["agents"][0]
    environment = client.get("/api/environments").json()["environments"][0]
    artifact_list_item = client.get("/api/artifacts").json()["artifacts"][0]
    artifact_read_body = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "test-artifacts"},
    ).json()
    artifact_read = artifact_read_body["artifact"]

    assert agent["metadata"] == {"note": f"agent {REDACTED_SECRET}"}
    assert agent["provider_options"] == {"header": f"Bearer {REDACTED_SECRET}"}
    assert environment["metadata"] == {"note": f"env {REDACTED_SECRET}"}
    assert artifact_list_item["metadata"] == {"note": f"artifact {REDACTED_SECRET}"}
    assert artifact_read["metadata"] == {"note": f"artifact {REDACTED_SECRET}"}
    assert artifact_read_body["text_preview"] == f"deployment {REDACTED_SECRET}\n"
    assert (
        artifact_read_body["preview_base64"]
        == base64.b64encode(f"deployment {REDACTED_SECRET}\n".encode()).decode()
    )


def test_server_artifact_inventory_rejects_duplicate_store_ids(tmp_path) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="duplicate-store")
    second_store = LocalArtifactStore(tmp_path / "second", store_id="duplicate-store")
    app.register_environment(
        Environment(EnvironmentSpec(name="first"), artifact_store=first_store),
        default=True,
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="second"), artifact_store=second_store),
    )
    asyncio.run(
        first_store.put_bytes(
            b"first",
            filename="first.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="first",
        )
    )
    asyncio.run(
        second_store.put_bytes(
            b"second",
            filename="second.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="second",
        )
    )

    response = TestClient(create_server(app, dev=True)).get("/api/artifacts")

    assert response.status_code == 409
    assert "duplicate-store" in response.json()["detail"]


def test_server_artifact_inventory_paginates_across_registered_stores(tmp_path) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="first-store")
    second_store = LocalArtifactStore(tmp_path / "second", store_id="second-store")
    app.register_environment(
        Environment(EnvironmentSpec(name="first"), artifact_store=first_store),
        default=True,
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="second"), artifact_store=second_store),
    )
    created = [
        (
            "first-store",
            asyncio.run(
                first_store.put_bytes(
                    b"one",
                    filename="one.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="first",
                )
            ),
        ),
        (
            "second-store",
            asyncio.run(
                second_store.put_bytes(
                    b"two",
                    filename="two.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="second",
                )
            ),
        ),
        (
            "first-store",
            asyncio.run(
                first_store.put_bytes(
                    b"three",
                    filename="three.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="first",
                )
            ),
        ),
    ]
    expected_ids = [
        artifact.id
        for _store_id, artifact in sorted(
            created,
            key=lambda item: (item[1].created_at.isoformat(), item[0], item[1].id),
            reverse=True,
        )
    ]
    client = TestClient(create_server(app, dev=True))

    first_page = client.get("/api/artifacts", params={"limit": 2})
    second_page = client.get("/api/artifacts", params={"limit": 2, "offset": 2})

    assert first_page.status_code == 200
    first_body = first_page.json()
    assert [artifact["id"] for artifact in first_body["artifacts"]] == expected_ids[:2]
    assert first_body["total_count"] == 3
    assert first_body["limit"] == 2
    assert first_body["offset"] == 0
    assert first_body["next_offset"] == 2
    assert first_body["truncated"] is True

    assert second_page.status_code == 200
    second_body = second_page.json()
    assert [artifact["id"] for artifact in second_body["artifacts"]] == expected_ids[2:]
    assert second_body["total_count"] == 3
    assert second_body["limit"] == 2
    assert second_body["offset"] == 2
    assert second_body["next_offset"] is None
    assert second_body["truncated"] is False


def test_server_artifact_inventory_rejects_unbounded_offsets(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_environment(
        Environment(EnvironmentSpec(name="local-review"), artifact_store=artifact_store),
        default=True,
    )

    response = TestClient(create_server(app, dev=True)).get(
        "/api/artifacts",
        params={"offset": 10_001},
    )

    assert response.status_code == 422


def test_server_artifact_inventory_does_not_advertise_unusable_next_offset() -> None:
    app = CayuApp()
    app.register_environment(
        Environment(EnvironmentSpec(name="local-review"), artifact_store=CountingArtifactStore()),
        default=True,
    )

    response = TestClient(create_server(app, dev=True)).get(
        "/api/artifacts",
        params={"offset": 10_000, "limit": 500},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["offset"] == 10_000
    assert body["limit"] == 500
    assert body["total_count"] == 20_000
    assert body["artifacts"]
    assert body["next_offset"] is None
    assert body["truncated"] is True


def test_server_exposes_pending_knowledge_review_endpoints() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_git",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                namespace="project:cayu",
                labels={"project": "cayu", "tenant": "trusted"},
                kind="procedure",
                status=KnowledgeStatus.PENDING,
                aspects=["git", "credentials"],
                title="Remote sandbox Git credentials",
                metadata={"review_note": "inspect before approving"},
            ),
            KnowledgeEntry(
                id="active_git",
                text="Active knowledge should not appear in pending review.",
                namespace="project:cayu",
                labels={"project": "cayu", "tenant": "trusted"},
                status=KnowledgeStatus.ACTIVE,
            ),
        ]
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_review_namespace="project:cayu",
        knowledge_review_labels={"project": "cayu", "tenant": "trusted"},
    )
    asyncio.run(
        store.replace_chunks(
            "pending_git",
            [
                KnowledgeChunk(
                    id="pending_git:0",
                    entry_id="pending_git",
                    chunk_index=0,
                    text="Remote sandbox Git pushes should use a brokered credential proxy.",
                )
            ],
        )
    )
    client = TestClient(create_server(app, dev=True))

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    body = pending.json()
    assert [entry["entry_id"] for entry in body["entries"]] == ["pending_git"]
    assert body["entries"][0]["title"] == "Remote sandbox Git credentials"
    assert body["entries"][0]["text_preview"] == "Remote sandbox Git credentials"
    assert body["total_entries_known"] == 1

    detail = client.get("/api/knowledge/pending/pending_git")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert (
        detail_body["text"] == "Remote sandbox Git pushes should use a brokered credential proxy."
    )
    assert detail_body["metadata"] == {"review_note": "inspect before approving"}
    assert [chunk["chunk_id"] for chunk in detail_body["chunks"]] == ["pending_git:0"]
    assert detail_body["chunks"][0]["text"] == detail_body["text"]

    approved = client.post("/api/knowledge/pending_git/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "active"

    empty = client.get("/api/knowledge/pending")
    assert empty.status_code == 200
    assert empty.json()["entries"] == []

    conflict = client.post("/api/knowledge/pending_git/reject")
    assert conflict.status_code == 409
    assert "not 'pending'" in conflict.json()["detail"]

    stale_detail = client.get("/api/knowledge/pending/pending_git")
    assert stale_detail.status_code == 409
    assert "not 'pending'" in stale_detail.json()["detail"]


def test_server_rejects_pending_knowledge_with_archived_status() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_bad",
                text="Do not retain this model-authored knowledge.",
                namespace="project:cayu",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), dev=True))

    rejected = client.post("/api/knowledge/pending_bad/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "archived"

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    assert pending.json()["entries"] == []


def test_server_knowledge_review_reports_missing_store_and_scope_errors() -> None:
    missing_store = TestClient(create_server(CayuApp(), dev=True))
    assert missing_store.get("/api/knowledge/pending").status_code == 404

    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_other",
                text="Other project knowledge.",
                namespace="project:other",
                labels={"project": "other"},
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_review_namespace="project:cayu",
        knowledge_review_labels={"project": "cayu"},
    )
    client = TestClient(create_server(app, dev=True))

    scoped_list = client.get("/api/knowledge/pending")
    assert scoped_list.status_code == 200
    assert scoped_list.json()["entries"] == []

    scoped_approve = client.post("/api/knowledge/pending_other/approve")
    assert scoped_approve.status_code == 403
    assert "outside review namespace" in scoped_approve.json()["detail"]

    scoped_detail = client.get("/api/knowledge/pending/pending_other")
    assert scoped_detail.status_code == 403
    assert "outside review namespace" in scoped_detail.json()["detail"]


def test_server_pending_knowledge_detail_validates_chunk_limits() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_git",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), dev=True))

    response = client.get("/api/knowledge/pending/pending_git?max_chunks=0")
    assert response.status_code == 422
    assert "max_chunks" in str(response.json()["detail"])

    too_many_chunks = client.get("/api/knowledge/pending/pending_git?max_chunks=51")
    assert too_many_chunks.status_code == 422
    assert "max_chunks" in str(too_many_chunks.json()["detail"])

    too_many_bytes = client.get("/api/knowledge/pending/pending_git?max_bytes=128001")
    assert too_many_bytes.status_code == 422
    assert "max_bytes" in str(too_many_bytes.json()["detail"])


def test_run_threads_inbound_traceparent_into_session_metadata() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))
    tasks = client.get("/api/tasks").json()

    assert len(tasks) == 1
    assert tasks[0]["id"] == "leased_task"
    assert tasks[0]["type"] == "review"
    assert tasks[0]["title"] is None
    assert tasks[0]["status"] == "claimed"
    assert tasks[0]["status_reason"] is None
    assert tasks[0]["status_payload"] is None
    assert tasks[0]["session_id"] is None
    assert tasks[0]["parent_task_id"] is None
    assert tasks[0]["assigned_agent_name"] == "assistant"
    assert tasks[0]["worker_id"] == "worker_a"
    assert tasks[0]["completed_at"] is None
    assert isinstance(tasks[0]["lease_expires_at"], str)
    assert isinstance(tasks[0]["created_at"], str)
    assert tasks[0]["description"] is None
    assert "input" not in tasks[0]
    assert "result" not in tasks[0]
    assert "error" not in tasks[0]
    assert "metadata" not in tasks[0]
    assert isinstance(tasks[0]["updated_at"], str)


def test_server_task_list_filters_lifecycle_states() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="blocked_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        await task_store.create_task(
            TaskCreate(
                task_id="ready_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        await task_store.block_task(
            "blocked_task",
            reason="Waiting on upstream import",
            payload={"dependency": "import_123"},
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))
    response = client.get(
        "/api/tasks",
        params={
            "status": TaskStatus.BLOCKED.value,
            "type": "review",
            "assigned_agent_name": "assistant",
        },
    )

    assert response.status_code == 200
    tasks = response.json()
    assert [task["id"] for task in tasks] == ["blocked_task"]
    assert tasks[0]["status"] == "blocked"
    assert tasks[0]["status_reason"] == "Waiting on upstream import"
    assert tasks[0]["status_payload"] == {"dependency": "import_123"}

    oldest_first_response = client.get(
        "/api/tasks",
        params={"order_by": "created_at_asc"},
    )
    assert oldest_first_response.status_code == 200
    assert [task["id"] for task in oldest_first_response.json()] == [
        "blocked_task",
        "ready_task",
    ]

    search_response = client.get("/api/tasks", params={"q": "upstream"})
    assert search_response.status_code == 200
    assert [task["id"] for task in search_response.json()] == ["blocked_task"]


def test_server_task_detail_returns_full_payload() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="detail_task",
                type="review",
                title="Inspect detail",
                description="Full task payload should stay off the list endpoint.",
                input={"document": "invoice.pdf", "amount": 42},
                metadata={"tenant": "acme", "priority": "high"},
            )
        )
        await task_store.start_task("detail_task", session_id="sess_detail")
        await task_store.complete_task("detail_task", {"accepted": True})

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))

    list_response = client.get("/api/tasks")
    assert list_response.status_code == 200
    list_task = list_response.json()[0]
    assert list_task["id"] == "detail_task"
    assert "input" not in list_task
    assert "result" not in list_task
    assert "error" not in list_task
    assert "metadata" not in list_task
    assert "started_at" not in list_task

    detail_response = client.get("/api/tasks/detail_task")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["id"] == "detail_task"
    assert detail["description"] == "Full task payload should stay off the list endpoint."
    assert detail["session_id"] == "sess_detail"
    assert detail["input"] == {"document": "invoice.pdf", "amount": 42}
    assert detail["result"] == {"accepted": True}
    assert detail["error"] is None
    assert detail["metadata"] == {"tenant": "acme", "priority": "high"}
    assert isinstance(detail["started_at"], str)


def test_server_task_detail_reports_missing_store_and_task() -> None:
    missing_store_client = TestClient(create_server(CayuApp(), dev=True))
    missing_store_response = missing_store_client.get("/api/tasks/task_1")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))
    missing_task_response = client.get("/api/tasks/missing_task")
    assert missing_task_response.status_code == 404
    assert missing_task_response.json()["detail"] == "Task not found: missing_task"


def test_server_task_lifecycle_endpoints_hold_and_resume_tasks() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="review_task",
                type="review",
                input={"document": "invoice.pdf"},
                metadata={"tenant": "acme"},
            )
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))
    block_response = client.post(
        "/api/tasks/review_task/block",
        json={
            "reason": "Waiting on operator",
            "payload": {"queue": "ops"},
        },
    )

    assert block_response.status_code == 200
    blocked = block_response.json()
    assert blocked["id"] == "review_task"
    assert blocked["status"] == "blocked"
    assert blocked["status_reason"] == "Waiting on operator"
    assert blocked["status_payload"] == {"queue": "ops"}
    assert blocked["input"] == {"document": "invoice.pdf"}
    assert blocked["metadata"] == {"tenant": "acme"}

    list_response = client.get("/api/tasks", params={"status": "blocked"})
    assert list_response.status_code == 200
    listed_tasks = list_response.json()
    assert [task["id"] for task in listed_tasks] == ["review_task"]
    assert "input" not in listed_tasks[0]
    assert "metadata" not in listed_tasks[0]

    resume_response = client.post("/api/tasks/review_task/resume")

    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["status"] == "pending"
    assert resumed["status_reason"] is None
    assert resumed["status_payload"] is None


def test_server_task_lifecycle_endpoints_support_pause_and_needs_attention() -> None:
    task_store = InMemoryTaskStore()

    async def setup_tasks() -> None:
        await task_store.create_task(TaskCreate(task_id="pause_task", type="review"))
        await task_store.create_task(TaskCreate(task_id="attention_task", type="review"))

    asyncio.run(setup_tasks())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))

    pause_response = client.post(
        "/api/tasks/pause_task/pause",
        json={"reason": "Worker maintenance"},
    )
    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "paused"
    assert pause_response.json()["status_reason"] == "Worker maintenance"

    attention_response = client.post(
        "/api/tasks/attention_task/needs-attention",
        json={"payload": {"field": "amount"}},
    )
    assert attention_response.status_code == 200
    assert attention_response.json()["status"] == "needs_attention"
    assert attention_response.json()["status_payload"] == {"field": "amount"}


def test_server_task_lifecycle_endpoints_report_invalid_transitions() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="attached_task", type="review"))
        await task_store.start_task("attached_task", session_id="sess_attached")

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))

    hold_response = client.post(
        "/api/tasks/attached_task/block",
        json={"reason": "not allowed"},
    )
    assert hold_response.status_code == 409
    assert "already attached to session sess_attached" in hold_response.json()["detail"]

    resume_response = client.post("/api/tasks/attached_task/resume")
    assert resume_response.status_code == 409
    assert "not paused, blocked, or waiting for attention" in resume_response.json()["detail"]


def test_server_task_lifecycle_endpoints_report_missing_task_store_and_task() -> None:
    missing_store_client = TestClient(create_server(CayuApp(), dev=True))

    missing_store_response = missing_store_client.post("/api/tasks/task_1/block")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))

    missing_task_response = client.post("/api/tasks/missing_task/block")
    assert missing_task_response.status_code == 404
    assert "missing_task" in missing_task_response.json()["detail"]


def test_server_task_lifecycle_endpoints_validate_request_body() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="task_1", type="review"))

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), dev=True))
    response = client.post("/api/tasks/task_1/block", json={"reason": "   "})

    assert response.status_code == 422


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

    client = TestClient(create_server(app, dev=True))
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
            "requested_model": None,
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
    client = TestClient(create_server(app, dev=True))

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


def test_server_run_defaults_and_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    captured: list[int] = []
    captured_models: list[str | None] = []
    original_run = app.run

    def spy_run(request: RunRequest):
        captured.append(request.max_steps)
        captured_models.append(request.model)
        return original_run(request)

    app.run = spy_run  # type: ignore[method-assign]
    client = TestClient(create_server(app, dev=True))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "max_steps": 7, "model": "request-model"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured == [20, 7]
    assert captured_models == [None, "request-model"]


def test_server_resume_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

    started = _session_started_event(client, "/api/run", {"prompt": "hello"}, {})
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
        json={"session_id": session_id, "prompt": "again", "max_steps": 42},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured == [42]


@pytest.mark.parametrize("path", ["/api/run", "/api/resume"])
@pytest.mark.parametrize("bad_value", [0, 257, -1])
def test_server_rejects_out_of_range_max_steps(path: str, bad_value: int) -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

    body: dict = {"prompt": "hello", "max_steps": bad_value}
    if path == "/api/resume":
        body["session_id"] = "session-does-not-matter"
    response = client.post(path, json=body)
    assert response.status_code == 422


def test_server_lists_sessions_with_label_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

    response = client.get("/api/sessions?label=cayu:agent=builder")

    assert response.status_code == 200
    assert response.json()["sessions"] == []


def test_server_rejects_invalid_session_label_filters() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, dev=True))

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

    client = TestClient(create_server(app, dev=True))
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
    assert body["usage"]["usage"]["total_tokens"] == 24
    assert body["provider_breakdown"] == [
        {
            "provider_name": "fake",
            "model": None,
            "session_count": 2,
            "model_steps": 2,
            "usage": {
                "provider_name": "fake",
                "requested_model": "fake-model",
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
        }
    ]
    assert body["model_breakdown"] == [
        {
            "provider_name": "fake",
            "model": "fake-model",
            "session_count": 2,
            "model_steps": 2,
            "usage": {
                "provider_name": "fake",
                "requested_model": "fake-model",
                "model": "fake-model",
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
        }
    ]
    assert body["cost"]["session_count"] == 2
    assert body["cost"]["total_cost"] == "0.00020"
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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
    assert body["usage"]["usage"]["total_tokens"] == 12
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
    client = TestClient(create_server(app, dev=True))

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


def test_server_pending_actions_lists_blocking_session_work() -> None:
    app = CayuApp()

    def pending_tool_call(tool_call_id: str, tool_name: str) -> dict[str, object]:
        return {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": {},
            "policy_decision": None,
            "reason": None,
            "metadata": {},
            "active_taint_labels": [],
        }

    def approval_checkpoint(
        *,
        approval_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "pending_tool_approval": {
                "approval_id": approval_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments or {},
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        **pending_tool_call(tool_call_id, tool_name),
                        "arguments": arguments or {},
                    }
                ],
            }
        }

    def user_input_checkpoint(
        *,
        input_id: str,
        tool_call_id: str,
        tool_name: str,
        question: str,
        options: list[str],
    ) -> dict[str, object]:
        return {
            "pending_user_input": {
                "input_id": input_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "question": question,
                "options": options,
                "arguments": {},
                "agent_name": "assistant",
                "tool_calls": [pending_tool_call(tool_call_id, tool_name)],
            }
        }

    def tool_round_checkpoint(
        *,
        round_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "pending_tool_round": {
                "round_id": round_id,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        **pending_tool_call(tool_call_id, tool_name),
                        "arguments": arguments or {},
                    }
                ],
            }
        }

    async def create(
        session_id: str,
        status: SessionStatus,
        events: list[Event],
        checkpoint: dict[str, object] | None = None,
    ) -> None:
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
        if checkpoint is not None:
            await app.session_store.checkpoint(session_id, checkpoint)

    async def seed() -> None:
        await create(
            "pending_approval",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="approval_requested",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="pending_approval",
                    tool_name="deploy",
                    payload={
                        "approval": {
                            "approval_id": "approval_1",
                            "tool_name": "deploy",
                            "reason": "production write",
                            "arguments": {"service": "api"},
                        }
                    },
                )
            ],
            checkpoint=approval_checkpoint(
                approval_id="approval_1",
                tool_call_id="call_deploy",
                tool_name="deploy",
                arguments={"service": "api"},
            ),
        )
        await create(
            "pending_user_input",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="awaiting_user_input",
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id="pending_user_input",
                    payload={
                        "input_id": "input_1",
                        "tool_call_id": "call_ask",
                        "question": "Deploy now?",
                        "options": ["yes", "no"],
                    },
                )
            ],
            checkpoint=user_input_checkpoint(
                input_id="input_1",
                tool_call_id="call_ask",
                tool_name="ask_user",
                question="Deploy now?",
                options=["yes", "no"],
            ),
        )
        await create(
            "manual_recovery",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="manual_recovery_event",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="manual_recovery",
                    payload={
                        "interruption_type": "tool_approval_required",
                        "manual_recovery_required": True,
                        "approval_id": "approval_2",
                        "tool_call_id": "call_refund",
                        "tool_name": "refund",
                        "error": "tool outcome unknown",
                    },
                )
            ],
            checkpoint=approval_checkpoint(
                approval_id="approval_2",
                tool_call_id="call_refund",
                tool_name="refund",
                arguments={"invoice_id": "inv_123"},
            ),
        )
        await create(
            "manual_tool_round_recovery",
            SessionStatus.FAILED,
            [
                Event(
                    id="manual_tool_round_started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="manual_tool_round_recovery",
                    tool_name="charge_card",
                    payload={
                        "tool_round_id": "round_crashed",
                        "tool_call_id": "call_charge",
                    },
                ),
                Event(
                    id="manual_tool_round_recovery_event",
                    type=EventType.SESSION_FAILED,
                    session_id="manual_tool_round_recovery",
                    payload={
                        "interruption_type": "runtime_interrupted",
                        "manual_recovery_required": True,
                        "tool_round_id": "round_crashed",
                        "tool_call_id": "call_charge",
                        "tool_name": "charge_card",
                        "error": "tool outcome unknown",
                    },
                ),
            ],
            checkpoint=tool_round_checkpoint(
                round_id="round_crashed",
                tool_call_id="call_charge",
                tool_name="charge_card",
                arguments={"amount": 42},
            ),
        )
        await create(
            "missing_checkpoint",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="missing_checkpoint_event",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="missing_checkpoint",
                    payload={
                        "interruption_type": "tool_approval_required",
                        "manual_recovery_required": True,
                        "approval_id": "approval_missing",
                        "tool_call_id": "call_missing",
                        "tool_name": "refund",
                        "error": "tool outcome unknown",
                    },
                )
            ],
        )
        await create(
            "resumed_approval",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="old_approval",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="resumed_approval",
                    payload={"approval": {"approval_id": "old_approval", "tool_name": "deploy"}},
                ),
                Event(
                    id="resumed_after_old_approval",
                    type=EventType.SESSION_RESUMED,
                    session_id="resumed_approval",
                ),
            ],
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, dev=True))

    response = client.get("/api/pending-actions")
    assert response.status_code == 200
    body = response.json()
    assert body["inspected_candidate_count"] == 4
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert body["total_count"] == 4
    actions_by_session = {action["session"]["id"]: action for action in body["actions"]}
    assert set(actions_by_session) == {
        "manual_recovery",
        "manual_tool_round_recovery",
        "pending_user_input",
        "pending_approval",
    }
    approval = actions_by_session["pending_approval"]
    assert approval["kind"] == "tool_approval"
    assert approval["approval_id"] == "approval_1"
    assert approval["arguments"] == {"service": "api"}
    user_input = actions_by_session["pending_user_input"]
    assert user_input["kind"] == "user_input"
    assert user_input["input_id"] == "input_1"
    assert user_input["question"] == "Deploy now?"
    assert user_input["options"] == ["yes", "no"]
    tool_round = actions_by_session["manual_tool_round_recovery"]
    assert tool_round["kind"] == "manual_recovery"
    assert tool_round["round_id"] == "round_crashed"
    assert tool_round["tool_call_id"] == "call_charge"
    assert tool_round["approval_id"] is None
    assert tool_round["input_id"] is None
    assert tool_round["arguments"] == {"amount": 42}
    approval_recovery = actions_by_session["manual_recovery"]
    assert approval_recovery["kind"] == "manual_recovery"
    assert approval_recovery["arguments"] == {"invoice_id": "inv_123"}

    filtered = client.get("/api/pending-actions?kind=user_input&q=deploy")
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total_count"] == 1
    assert filtered_body["actions"][0]["session"]["id"] == "pending_user_input"

    exact = client.get("/api/pending-actions?session_id=manual_recovery")
    assert exact.status_code == 200
    exact_body = exact.json()
    assert exact_body["inspected_candidate_count"] == 1
    assert exact_body["total_count"] == 1
    assert exact_body["actions"][0]["kind"] == "manual_recovery"

    tool_round_exact = client.get("/api/pending-actions?session_id=manual_tool_round_recovery")
    assert tool_round_exact.status_code == 200
    tool_round_exact_body = tool_round_exact.json()
    assert tool_round_exact_body["inspected_candidate_count"] == 1
    assert tool_round_exact_body["total_count"] == 1
    assert tool_round_exact_body["actions"][0]["round_id"] == "round_crashed"

    stale_exact = client.get("/api/pending-actions?session_id=missing_checkpoint")
    assert stale_exact.status_code == 200
    stale_body = stale_exact.json()
    assert stale_body["inspected_candidate_count"] == 0
    assert stale_body["total_count"] == 0


def test_server_pending_actions_uses_one_store_native_query() -> None:
    class PendingActionStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.pending_query_count = 0

        async def query_pending_actions(self, query=None):
            self.pending_query_count += 1
            return PendingActionListResult()

        async def list_sessions(self, query=None):
            raise AssertionError("pending-action route must not list candidate sessions")

        async def query_events(self, query=None):
            raise AssertionError("pending-action route must not query per-session events")

        async def load_checkpoint(self, session_id: str):
            raise AssertionError("pending-action route must not load per-session checkpoints")

    store = PendingActionStore()
    client = TestClient(create_server(CayuApp(session_store=store), dev=True))

    response = client.get("/api/pending-actions")

    assert response.status_code == 200
    assert response.json() == {
        "actions": [],
        "issues": [],
        "next_cursor": None,
        "has_more": False,
        "total_count": None,
        "inspected_candidate_count": 0,
    }
    assert store.pending_query_count == 1


def test_server_pending_actions_returns_413_for_oversized_page() -> None:
    class OversizedPendingActionStore(InMemorySessionStore):
        async def query_pending_actions(self, query=None):
            from cayu.runtime.sessions import PendingActionResultTooLarge

            raise PendingActionResultTooLarge(2 * 1024 * 1024)

    client = TestClient(
        create_server(CayuApp(session_store=OversizedPendingActionStore()), dev=True)
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 413
    assert "2097152-byte result limit" in response.json()["detail"]


def test_server_pending_actions_rejects_invalid_cursor_as_400() -> None:
    client = TestClient(create_server(CayuApp(), dev=True))

    response = client.get("/api/pending-actions?cursor=not-a-cursor")

    assert response.status_code == 400
    assert "Invalid session cursor" in response.json()["detail"]


def test_server_pending_actions_does_not_misclassify_store_failure_as_400() -> None:
    class FailingPendingActionStore(InMemorySessionStore):
        async def query_pending_actions(self, query=None):
            raise ValueError("persisted pending-action projection is corrupt")

    client = TestClient(
        create_server(CayuApp(session_store=FailingPendingActionStore()), dev=True),
        raise_server_exceptions=False,
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 500


def test_server_run_rejects_request_budget_reservations() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

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
    assert "Request budget limits must not use reservations" in response.text


def test_server_session_usage_returns_404_for_missing_session() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))
    response = client.get("/api/sessions/missing/usage")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_usage_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
    response = client.post(
        "/api/sessions/tiered_cost/cost",
        json={"pricing": price_book},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_cost"] == "0.00010"
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

    client = TestClient(create_server(app, dev=True))
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
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": {
            "provider_name": None,
            "model": None,
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
            "reasoning_output_tokens": 0,
            "requested_model": None,
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
                    "requested_model": None,
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
                    "requested_model": None,
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
    assert cost_response.json()["total_cost"] == "0.00020"
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
    assert summary_body["usage"]["usage"]["total_tokens"] == 24
    assert summary_body["cost"]["total_cost"] == "0.00020"

    missing_summary_response = client.post(
        "/api/causal-budgets/missing/summary",
        json=pricing_body,
    )
    assert missing_summary_response.status_code == 404
    assert missing_summary_response.json() == {"detail": "Causal budget not found"}


def test_server_rejects_a_merged_flat_and_model_catalog_pricing_body() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
    response = client.post(
        "/api/sessions/session_1/cost",
        json={"pricing": _price_book_payload(input_per_million="-1")},
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

    client = TestClient(create_server(app, dev=True))
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
    assert body["events"]["total_events"] == 6
    assert body["events"]["counts_by_type"] == {
        "model.completed": 1,
        "model.started": 1,
        "model.text.delta": 1,
        "session.completed": 1,
        "session.started": 1,
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
        "provider_names": ["fake"],
        "models": ["fake-model"],
        "usage": {
            "provider_name": None,
            "model": None,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "reasoning_output_tokens": 0,
            "requested_model": None,
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
    response = client.get("/api/sessions/missing/summary")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_server_session_summary_rejects_blank_session_id() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
    response = client.get("/api/sessions/state_1/state")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "state_1"
    assert body["status"] == "running"
    assert body["interruption_cascade"] == "none"
    assert body["updated_at"]
    assert body["last_activity_at"]
    assert set(body) == {
        "session_id",
        "status",
        "updated_at",
        "last_activity_at",
        "interruption_cascade",
    }


def test_server_session_state_returns_404_and_validates_id() -> None:
    client = TestClient(create_server(CayuApp(), dev=True))

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

    client = TestClient(create_server(app, dev=True))

    first_page = client.get("/api/sessions/events_1/events?limit=2")
    assert first_page.status_code == 200
    first_body = first_page.json()
    assert first_body["session_id"] == "events_1"
    assert first_body["order_by"] == "sequence_asc"
    assert first_body["has_more"] is True
    assert first_body["next_sequence"] == 2
    assert first_body["scan_through_sequence"] == 2
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
    assert second_body["order_by"] == "sequence_asc"
    assert second_body["next_sequence"] == 3
    assert second_body["scan_through_sequence"] == 3
    assert [event["id"] for event in second_body["events"]] == ["event_3"]

    latest_page = client.get("/api/sessions/events_1/events?order_by=sequence_desc&limit=2")
    assert latest_page.status_code == 200
    latest_body = latest_page.json()
    assert latest_body["order_by"] == "sequence_desc"
    assert latest_body["has_more"] is True
    assert latest_body["next_sequence"] == 2
    assert latest_body["scan_through_sequence"] == 3
    assert [event["id"] for event in latest_body["events"]] == ["event_3", "event_2"]

    older_page = client.get(
        "/api/sessions/events_1/events?order_by=sequence_desc&before_sequence=2&limit=2"
    )
    assert older_page.status_code == 200
    older_body = older_page.json()
    assert older_body["order_by"] == "sequence_desc"
    assert older_body["has_more"] is False
    assert older_body["next_sequence"] == 1
    assert older_body["scan_through_sequence"] is None
    assert [event["id"] for event in older_body["events"]] == ["event_1"]

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

    client = TestClient(create_server(app, dev=True))
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
    assert [event["id"] for event in body["events"]] == ["event_filter_1"]

    bounded_response = client.get(
        "/api/sessions/events_filters/events",
        params={"event_type": "tool.call.completed", "limit": 1},
    )
    assert bounded_response.status_code == 200
    bounded_body = bounded_response.json()
    assert bounded_body["has_more"] is True
    assert bounded_body["scan_through_sequence"] == 1
    assert [event["id"] for event in bounded_body["events"]] == ["event_filter_1"]


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

    client = TestClient(create_server(app, dev=True))
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
        (event_id, "event_lookup", {"source": "selected"})
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
    client = TestClient(create_server(app, dev=True))

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
    assert [event["id"] for event in body["events"]] == ["useful_new", "useful_old"]
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))

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
    assert (
        client.get(
            "/api/sessions/events_validation/events?exclude_event_type=not.valid"
        ).status_code
        == 422
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

    async def fail_unbounded_session_load(*_args, **_kwargs):
        raise AssertionError("transcript pagination must use the bounded state projection")

    app.session_store.load = fail_unbounded_session_load  # type: ignore[method-assign]

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))
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

    client = TestClient(create_server(app, dev=True))

    assert client.get("/api/sessions/transcript_validation/transcript?limit=0").status_code == 422
    assert (
        client.get("/api/sessions/transcript_validation/transcript?role=invalid").status_code == 422
    )
    assert client.get("/api/sessions/%20/transcript").status_code == 422


def test_dashboard_routes_fall_back_to_index_without_masking_api_or_assets() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))

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
    parent.mount("/product", create_server(CayuApp(), dev=True))

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
    parent.mount("/product", create_server(CayuApp(), dev=True))

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
    disabled = TestClient(create_server(CayuApp(), dev=True, dashboard_path=None))
    assert disabled.get("/cayu/").status_code == 404

    custom = TestClient(create_server(CayuApp(), dev=True, dashboard_path="/inspector"))
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
            dev=True,
            dashboard_path="/cayu",
            api_path="/cayu/api",
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

    mount_cayu(server, cayu_app, path="/cayu", dev=True)

    client = TestClient(server)
    dashboard = client.get("/cayu/knowledge")

    assert dashboard.status_code == 200
    assert '<div id="root"></div>' in dashboard.text
    assert '<base href="/cayu/" />' in dashboard.text
    assert '"basePath":"/cayu"' in dashboard.text
    assert '"apiBaseUrl":"/cayu/api"' in dashboard.text
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/api/health").status_code == 404
    assert client.get("/cayu/api/missing").status_code == 404


def test_mount_cayu_can_disable_dashboard_for_api_only_services() -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, dev=True)

    client = TestClient(server)
    assert client.get("/cayu/api/health").json() == {"ok": True}
    assert client.get("/cayu/").status_code == 404


def test_mount_cayu_composes_background_interruption_drain() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    drain_timeouts = []
    resume_calls = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_before):
        resume_calls.append(interrupting_inactive_before)
        return 0

    async def drain_background_interruptions(*, timeout_s):
        drain_timeouts.append(timeout_s)
        return True

    cayu_app.drain_background_interruptions = drain_background_interruptions
    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    mount_cayu(
        server,
        cayu_app,
        path="/cayu",
        dashboard=False,
        dev=True,
        interruption_shutdown_grace_seconds=2.5,
    )

    with TestClient(server):
        pass

    assert drain_timeouts == [2.5]
    assert len(resume_calls) == 1
    assert resume_calls[0] < datetime.now(UTC)


def test_mount_cayu_drains_cascades_when_startup_recovery_fails() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_before):
        assert interrupting_inactive_before < datetime.now(UTC)
        calls.append("recover")
        raise RuntimeError("mounted recovery failed")

    async def drain_background_interruptions(*, timeout_s):
        assert timeout_s == 10.0
        calls.append("drain")
        return True

    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    cayu_app.drain_background_interruptions = drain_background_interruptions
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, dev=True)

    with (
        pytest.raises(RuntimeError, match="mounted recovery failed"),
        TestClient(server),
    ):
        pass

    assert calls == ["recover", "drain"]


def test_run_rejects_blank_prompt_and_agent_before_runtime() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    client = TestClient(create_server(app, dev=True))

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
    for bad_max_steps in (True, "7"):
        assert (
            client.post(
                "/api/tool-approvals/resolve",
                json={
                    "session_id": "session_1",
                    "approval_id": "approval_1",
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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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


def test_dev_mode_resolution_restamps_body_resolved_by_as_request_source() -> None:
    from cayu import ResolutionActorSource

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

    asyncio.run(create_interrupted_session("session_dev_actor"))
    asyncio.run(create_interrupted_session("session_dev_tool_round_actor"))

    captured = []
    captured_tool_round = []

    async def resolve_tool_approval(request):
        captured.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    async def recover_tool_round(request):
        captured_tool_round.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resolve_tool_approval = resolve_tool_approval
    app.recover_tool_round = recover_tool_round
    client = TestClient(create_server(app, dev=True))

    # A dev-mode body can assert an identity but never verified/system
    # provenance: the server re-stamps the source as "request".
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "decision": "approve",
            "resolved_by": {"subject": "operator@example.com", "source": "system"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    actor = captured[0].resolved_by
    assert actor is not None
    assert actor.subject == "operator@example.com"
    assert actor.source is ResolutionActorSource.REQUEST

    with client.stream(
        "POST",
        "/api/tool-rounds/recover",
        json={
            "session_id": "session_dev_tool_round_actor",
            "round_id": "round_1",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "verified externally",
            "resolved_by": {"subject": "round-operator@example.com", "source": "system"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    round_actor = captured_tool_round[0].resolved_by
    assert round_actor is not None
    assert round_actor.subject == "round-operator@example.com"
    assert round_actor.source is ResolutionActorSource.REQUEST

    # Reserved system subjects cannot be claimed through the body: the
    # request-source re-stamp trips the reserved-prefix validation.
    response = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "decision": "approve",
            "resolved_by": {"subject": "cayu:approval-expiry", "source": "system"},
        },
    )
    assert response.status_code == 400
    assert "reserved for system actors" in response.json()["detail"]
    assert len(captured) == 1

    # No body actor means no provenance in dev mode.
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    assert captured[1].resolved_by is None


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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_idempotent/interrupt",
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "already interrupted" in body


def _lifecycle_store_and_client(seed) -> tuple[InMemorySessionStore, TestClient]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, dev=True))
    asyncio.run(seed(store))
    return store, client


def _create_session(store: InMemorySessionStore, session_id: str, **kwargs):
    return store.create(
        RunRequest(
            agent_name="builder",
            session_id=session_id,
            messages=[Message.text("user", "x")],
            **kwargs,
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


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
        await _create_session(store, "sess_meta", metadata={"a": 1})

    _, client = _lifecycle_store_and_client(seed)

    response = client.patch("/api/sessions/sess_meta/metadata", json={"metadata": {"b": [1, 2]}})
    assert response.status_code == 200
    assert response.json()["metadata"] == {"b": [1, 2]}
    missing = client.patch("/api/sessions/sess_missing/metadata", json={"metadata": {}})
    assert missing.status_code == 404
    # Typo'd key / missing field must 422, never silently wipe metadata.
    assert client.patch("/api/sessions/sess_meta/metadata", json={"metadat": {}}).status_code == 422
    assert client.patch("/api/sessions/sess_meta/metadata", json={}).status_code == 422
    assert client.get("/api/sessions/sess_meta").json()["metadata"] == {"b": [1, 2]}


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
    client = TestClient(create_server(app, dev=True))

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


def _sse_frames(response) -> list[dict]:
    """Collect SSE frames as dicts with optional `id`, `event`, and parsed `data`."""
    frames: list[dict] = []
    current: dict = {}
    for line in response.iter_lines():
        if not line.strip():
            if current:
                frames.append(current)
                current = {}
            continue
        if line.startswith("id:"):
            current["id"] = line[len("id:") :].strip()
        elif line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:") :].strip())
    if current:
        frames.append(current)
    return frames


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
    server = create_server(app, dev=True)
    session_id = "session_factory_acceptance_boundary"

    async def exercise() -> tuple[bool, list[dict[str, Any]]]:
        request_task = asyncio.create_task(
            _post_and_disconnect_before_first_body(
                server,
                "/api/run",
                {"prompt": "hello", "session_id": session_id},
            )
        )
        await asyncio.wait_for(factory.started.wait(), timeout=1)
        done, _ = await asyncio.wait({request_task}, timeout=0.2)
        accepted_before_release = request_task in done
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert state.status is SessionStatus.RUNNING
        active_runs = app._active_session_run_records(session_id)
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
    server = create_server(app, dev=True)
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
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        messages = await asyncio.wait_for(request_task, timeout=1)

        active_runs = app._active_session_run_records(session_id)
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
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
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
        active_runs = app._active_session_run_records(session_id)
        assert len(active_runs) == 1
        assert not active_runs[0].runtime_task.done()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        observed = [message async for message in response.body_iterator]
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[0] == EventType.SESSION_STARTED
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
        await asyncio.wait_for(factory.started.wait(), timeout=1)
        await asyncio.wait_for(factory.cancelled.wait(), timeout=1)
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[0] == EventType.ENVIRONMENT_FACTORY_STARTED
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_interrupt_during_run_acceptance_finishes_task_bookkeeping() -> None:
    class BlockingTaskStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.release_create = asyncio.Event()

        async def create_running_task(self, request):
            self.create_started.set()
            await self.release_create.wait()
            return await super().create_running_task(request)

    task_store = BlockingTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_during_acceptance_bookkeeping"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="operator stop")
                )
            ]

        response_task = asyncio.create_task(
            _accepted_event_stream_response(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        task_id="task_interrupt_during_acceptance_bookkeeping",
                        messages=[Message.text("user", "hello")],
                    )
                ),
                cayu_app=app,
                session_id=session_id,
                after_accept=lambda _event: task_store.create_running_task(
                    TaskCreate(
                        task_id="task_interrupt_during_acceptance_bookkeeping",
                        type="run",
                        session_id=session_id,
                    )
                ),
            )
        )
        await asyncio.wait_for(task_store.create_started.wait(), timeout=1)
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
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[0] == EventType.SESSION_STARTED
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_run_route_interrupt_during_acceptance_state_read_keeps_task_linked() -> None:
    class BlockingAcceptanceStateStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_next_state_read = True
            self.state_read_started = asyncio.Event()
            self.release_state_read = asyncio.Event()

        async def load_state(self, session_id: str):
            if self.block_next_state_read:
                self.block_next_state_read = False
                self.state_read_started.set()
                await self.release_state_read.wait()
            return await super().load_state(session_id)

    session_store = BlockingAcceptanceStateStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, dev=True)
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
            await asyncio.wait_for(session_store.state_read_started.wait(), timeout=1)
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
        await asyncio.wait_for(callback_started.wait(), timeout=1)
        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task

        release_callback.set()
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        active_runs = app._active_session_run_records(session_id)
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
        assert app._active_session_run_records(session_id) == ()
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
                return Event(
                    id="event_before_runtime_failure",
                    type="custom.before_runtime_failure",
                    session_id="session_cancelled_stream_cleanup",
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

    assert messages[0]["id"] == ("session_cancelled_stream_cleanup:event_before_runtime_failure")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "runtime_failed"


def test_accepted_stream_driver_finishes_when_response_start_send_fails() -> None:
    async def exercise() -> bool:
        completed = asyncio.Event()

        async def event_stream() -> AsyncIterator[Event]:
            yield Event(
                id="event_response_start_failure",
                type="custom.accepted",
                session_id="session_response_start_failure",
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
                for event in events:
                    yield event
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
                yield Event(
                    id=f"event_{index}",
                    type="custom.observer",
                    session_id=request.session_id,
                )
        finally:
            completed = True

    app.run = synchronous_burst  # type: ignore[method-assign]
    client = TestClient(create_server(app, dev=True))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert completed is True
    assert len(frames) == SSE_OBSERVER_MAX_FRAMES + 1
    assert [frame["data"]["id"] for frame in frames] == [
        f"event_{index}" for index in range(SSE_OBSERVER_MAX_FRAMES + 1)
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
    client = TestClient(create_server(app, dev=True))

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
    assert replayed[-1]["data"]["type"] == "session.completed"
    assert queries[0].event_id == frames[0]["data"]["id"]
    assert queries[1].limit == SSE_REPLAY_PAGE_EVENTS

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


def test_streaming_mutation_id_creates_an_exact_durable_acceptance_event() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))
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
    assert marker["payload"] == {
        "mutation_id": mutation_id,
        "mutation_kind": "run",
        "accepted_event_id": frames[0]["data"]["id"],
        "accepted_event_type": frames[0]["data"]["type"],
    }
    assert frames.index(next(frame for frame in frames if frame["data"] == marker)) > 0

    events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"event_type": EventType.SERVER_MUTATION_ACCEPTED},
    ).json()["events"]
    assert [event["id"] for event in events] == [marker["id"]]


def test_streaming_mutation_id_header_rejects_unsafe_values_before_execution() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, dev=True))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session-invalid-mutation-id"},
        headers={"Cayu-Mutation-ID": "invalid mutation id"},
    )

    assert response.status_code == 422
    assert asyncio.run(app.session_store.load_state("session-invalid-mutation-id")) is None


def test_explicit_compaction_endpoint_uses_replayable_mutation_contract() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=TranscriptDigestCompactor(),
            max_user_turns=1,
        ),
    )

    async def prepare() -> tuple[int, int]:
        session = await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-explicit-compact-endpoint",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await app.session_store.append_transcript_messages(session.id, transcript)
        completed = await app.session_store.update_status(session.id, SessionStatus.COMPLETED)
        return completed.run_epoch, len(transcript)

    run_epoch, transcript_cursor = asyncio.run(prepare())
    client = TestClient(create_server(app, dev=True))
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
    client = TestClient(create_server(CayuApp(enable_logging=False), dev=True))

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
    client = TestClient(create_server(CayuApp(enable_logging=False), dev=True))

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
    client = TestClient(create_server(app, dev=True))

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
        "event_tool_started",
        "event_tool_blocked",
        "event_session_completed",
    ]


def test_enqueue_session_message_endpoint_uses_replayable_mutation_contract() -> None:
    app = CayuApp(enable_logging=False)

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
    client = TestClient(create_server(app, dev=True))
    session_id = "session-message-endpoint"
    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/messages",
        json={
            "idempotency_key": "message-endpoint-1",
            "content": "Please prioritize the failing deployment.",
            "delivery_mode": "next_turn",
            "requested_by": {"subject": "operator@example.com"},
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
        "source": "request",
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
        "accepted_event_type": EventType.SESSION_MESSAGE_QUEUED,
    }
    assert "/api/sessions/{session_id}/messages" in client.get("/openapi.json").json()["paths"]


def test_enqueue_session_message_endpoint_rejects_nonportable_text() -> None:
    app = CayuApp(enable_logging=False)
    client = TestClient(create_server(app, dev=True))

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
    server = create_server(app, dev=True)
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

    assert replayed == durable_event_ids
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
    server = create_server(app, dev=True)

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

    assert replayed == ["event_resume_accepted", "event_resume_completed"]
    assert executions == 1


def test_concurrent_client_run_identity_creates_one_session_and_one_task() -> None:
    class CoordinatedRunStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.claims = 0
            self.both_claiming = asyncio.Event()

        async def create(self, request, *, identity):
            if request.session_id == "session_concurrent_claim":
                self.claims += 1
                if self.claims == 2:
                    self.both_claiming.set()
                await self.both_claiming.wait()
            return await super().create(request, identity=identity)

    store = CoordinatedRunStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=task_store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, dev=True)

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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

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

    assert unknown_event.status_code == 409
    assert "event was not found" in unknown_event.json()["detail"]
    assert mismatched_session.status_code == 422
    assert "does not match" in mismatched_session.json()["detail"]
    assert executed == []
    assert client.get("/api/tasks").json() == []


@pytest.mark.parametrize(
    "session_id",
    ["session:colon", " leading-space", "slash/not-allowed", "x" * 129],
)
def test_run_rejects_session_ids_that_are_not_replay_safe(session_id: str) -> None:
    response = TestClient(create_server(CayuApp(), dev=True)).post(
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
    client = TestClient(create_server(app, dev=True))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session_duplicate"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Session already exists: session_duplicate"
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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

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

        transport = httpx.ASGITransport(app=create_server(app, dev=True))
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
        "event_resume_baseline",
        "event_delayed_terminal",
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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

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

        transport = httpx.ASGITransport(app=create_server(app, dev=True))
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
    assert [frame["data"]["id"] for frame in frames] == ["event_current_terminal"]


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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_operation_start"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert [frame["data"]["id"] for frame in frames] == [
        "event_terminal",
        "event_unverified_hook",
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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

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
def test_replay_recognizes_framework_post_terminal_marker(
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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

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
    client = TestClient(create_server(app, dev=True, replay_idle_timeout_s=0.01))

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

        transport = httpx.ASGITransport(app=create_server(app, dev=True))
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
    assert [frame["data"]["id"] for frame in frames] == ["event_current_terminal"]


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
    client = TestClient(create_server(app, dev=True))

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
    assert event_frames[0]["data"]["id"] == "event_0"
    assert event_frames[-2]["data"]["id"] == f"event_{event_count - 1}"
    assert event_frames[-1]["data"]["id"] == "event_replay_terminal"


def test_oversized_replay_frame_remains_durable_and_fails_live_observer_clearly() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_large_replay",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "session_large_replay",
            [
                Event(
                    id="event_seen",
                    type=EventType.SESSION_STARTED,
                    session_id="session_large_replay",
                    agent_name="assistant",
                ),
                Event(
                    id="event_large",
                    type="custom.large",
                    session_id="session_large_replay",
                    agent_name="assistant",
                    payload={"value": "x" * SSE_EVENT_DATA_MAX_BYTES},
                ),
            ],
        )
        await app.session_store.update_status("session_large_replay", SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    client = TestClient(create_server(app, dev=True))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay"},
        headers={"Last-Event-ID": "session_large_replay:event_seen"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert len(frames) == 1
    assert frames[0]["event"] == "error"
    assert frames[0]["data"]["kind"] == "observer"
    assert frames[0]["data"]["code"] == "event_frame_too_large"
    assert frames[0]["data"]["retryable"] is False

    async def load_large_event() -> Event:
        records = await app.session_store.query_events(
            EventQuery(session_id="session_large_replay", event_id="event_large", limit=1)
        )
        assert len(records) == 1
        return records[0].event

    durable_event = asyncio.run(load_large_event())
    assert len(cast("str", durable_event.payload["value"])) == SSE_EVENT_DATA_MAX_BYTES


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/api/tool-approvals/resolve",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
                "decision": "approve",
            },
        ),
        (
            "/api/tool-approvals/recover",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
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
        (
            "/api/sessions/session_approval_replay/messages",
            {
                "idempotency_key": "message-replay",
                "content": "queued steering that must not execute",
                "delivery_mode": "next_turn",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("last_event_id", "expected_event_ids"),
    [
        ("session_approval_replay:event_seen", ["event_missed"]),
        ("session_approval_replay:", ["event_seen", "event_missed"]),
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
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

    response = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_requested",
            "approval_id": "approval_1",
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

    async def recover(request):
        calls.append("recover")
        requests.append(request)
        return []

    async def drain_background_interruptions(*, timeout_s):
        calls.append("drain")
        assert timeout_s == 10.0
        return True

    async def resume_pending_interruption_cascades(*, interrupting_inactive_before):
        calls.append("resume_cascades")
        assert interrupting_inactive_before < datetime.now(UTC)
        return 0

    app.recover_incomplete_sessions = recover
    app.drain_background_interruptions = drain_background_interruptions
    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    server = create_server(
        app,
        dev=True,
        lifespan=user_lifespan,
        startup_recovery_statuses={
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        },
        recovery_inactive_after_seconds=60,
    )

    with TestClient(server):
        assert calls == ["user_start", "recover", "resume_cascades"]

    assert calls == ["user_start", "recover", "resume_cascades", "drain", "user_stop"]
    assert len(requests) == 1
    request = requests[0]
    assert request.statuses == {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
    }
    assert request.reason == "server_startup_recovery"
    assert request.metadata == {"source": "create_server"}
    assert request.inactive_before is not None
    assert request.inactive_before < datetime.now(UTC)


def test_create_server_drains_cascades_when_startup_recovery_fails() -> None:
    app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_before):
        assert interrupting_inactive_before < datetime.now(UTC)
        calls.append("recover")
        raise RuntimeError("recovery failed after scheduling work")

    async def drain_background_interruptions(*, timeout_s):
        assert timeout_s == 10.0
        calls.append("drain")
        return True

    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    app.drain_background_interruptions = drain_background_interruptions
    server = create_server(app, dev=True)

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
            dev=True,
            interruption_shutdown_grace_seconds=value,
        )


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_mount_cayu_rejects_invalid_interruption_recovery_inactivity(value) -> None:
    with pytest.raises(ValueError, match="interruption_recovery_inactive_after_seconds"):
        mount_cayu(
            FastAPI(),
            CayuApp(),
            dashboard=False,
            dev=True,
            interruption_recovery_inactive_after_seconds=value,
        )


def test_client_disconnect_does_not_cancel_detached_run() -> None:
    import time

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, dev=True))

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
    client = TestClient(create_server(app, dev=True))

    async def broken_run(request):
        yield Event(
            type=EventType.SESSION_STARTED,
            session_id=request.session_id,
            agent_name=request.agent_name,
        )
        raise RuntimeError("run exploded with secret-token " + "x" * 1000)

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
        yield Event(
            id="event_interrupted",
            type=EventType.SESSION_INTERRUPTED,
            session_id=request.session_id,
            agent_name="assistant",
        )
        raise RuntimeError("interrupt failed with secret-token")

    app.interrupt_session = broken_interrupt
    client = TestClient(create_server(app, dev=True))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_stream_error/interrupt",
        json={"reason": "operator request"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[0]["data"]["id"] == "event_interrupted"
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


def _sse_events(client: TestClient, path: str, body: dict) -> list[dict]:
    with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        return [frame["data"] for frame in _sse_frames(response) if "data" in frame]


def _ask_user_client() -> TestClient:
    app = CayuApp()
    app.register_provider(AskUserProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
    return TestClient(create_server(app, dev=True))


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

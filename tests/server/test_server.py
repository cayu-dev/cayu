from __future__ import annotations

# ruff: noqa: E402
import asyncio
import base64
import json
import logging
import re
import threading
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

import cayu.runtime.sessions as sessions_module

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    ArtifactScope,
    BillingIdentity,
    CayuApp,
    CompletionResultResolverRef,
    CompletionVerifierRef,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
    LocalArtifactStore,
    LocalWorkspace,
    Message,
    MessageRole,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    SecretRedactor,
    SQLiteSessionStore,
    Task,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryPolicy,
    TaskRetrySettlementRequest,
    TaskStatus,
    TextPart,
    ThinkingPart,
    UserInputTool,
    WorkContractDraft,
    WorkCriterion,
    WorkspaceBinding,
    WorkspaceBranch,
    WorkspaceBranchCapabilities,
    WorkspaceBranchRequest,
    default_price_book,
    work_contract_from_draft,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.artifacts import ArtifactListResult, ArtifactMetadata, ArtifactReadResult, ArtifactStore
from cayu.artifacts.attachments import FileAttachment, FileAttachmentKind
from cayu.core.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    event_with_durable_sequence,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import FilePart, ProviderStatePart
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.runtime import (
    CheckpointCompactionContextPolicy,
    Dispatcher,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    EventQuery,
    EventRecord,
    ForkSessionRequest,
    IncompleteSessionsRecoveryPage,
    InMemoryEventSink,
    InMemorySessionStore,
    InterruptSessionRequest,
    ModelTarget,
    PendingActionIssue,
    PendingActionIssueCode,
    PendingActionListResult,
    PendingActionQuery,
    PersistedEventSideEffectStatus,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionListResult,
    SessionStatus,
    TerminalEventPublicationUncertain,
    TranscriptDigestCompactor,
)
from cayu.runtime._continuation_task_failure import runtime_task_failure_identity_from_task
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    REDACTED_CUSTOM_EVENT_TYPE,
    public_event_envelope_alias_field,
    public_event_id,
    public_event_linkage_id,
    public_event_sequence,
)
from cayu.runtime.budgets import InMemoryBudgetStore
from cayu.runtime.checkpoints import CURRENT_CHECKPOINT_SCHEMA_VERSION
from cayu.runtime.provider_operations import (
    PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES,
)
from cayu.runtime.sessions import run_request_with_runtime_generated_authority
from cayu.runtime.usage import CacheUsageMetrics, UsageMetrics
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
    _add_usage_metrics,
    _detached_event_stream_response,
    _log_mutation_acceptance_failure,
    _next_replay_poll_interval,
    _serialize_pending_action,
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
from cayu.tools import ExecCommandTool


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


_LOCAL_SERVER_CONFIG = ServerConfig.local_development()
_SHORT_REPLAY_SERVER_CONFIG = ServerConfig.local_development(
    lifecycle=ServerLifecycleConfig(replay_idle_timeout_s=0.01)
)


class OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class UnavailableTargetProvider(OneShotProvider):
    def __init__(self) -> None:
        self.dispatches = 0

    def preflight_model_target(self, *, model: str) -> None:
        raise RuntimeError(f"model target {model!r} is not configured")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.dispatches += 1
        async for event in super().stream(request):
            yield event


class UsageProvider(ModelProvider):
    name = "fake"
    usage_dialect = UsageDialect.OPENAI

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


def _aggregate_usage_json(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cache_read_tokens: int,
    cached_input_tokens: int,
    uncached_input_tokens: int,
) -> dict:
    return {
        "input_tokens": str(input_tokens),
        "output_tokens": str(output_tokens),
        "total_tokens": str(total_tokens),
        "reasoning_output_tokens": "0",
        "cache": {
            "read_tokens": str(cache_read_tokens),
            "write_tokens": "0",
            "write_5m_tokens": "0",
            "write_1h_tokens": "0",
            "write_unknown_ttl_tokens": "0",
            "cached_input_tokens": str(cached_input_tokens),
            "uncached_input_tokens": str(uncached_input_tokens),
        },
    }


def test_server_usage_aggregation_preserves_cache_ttl_buckets() -> None:
    combined = _add_usage_metrics(
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

    assert combined.cache.write_tokens == 12
    assert combined.cache.write_5m_tokens == 9
    assert combined.cache.write_1h_tokens == 3
    assert combined.cache.write_unknown_ttl_tokens == 1


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


def test_environment_capability_projection_preserves_controls_and_redacts_detail(
    tmp_path,
) -> None:
    canary = "extension-capability-secret"

    class DetailWorkspace(LocalWorkspace):
        def branch_capabilities(self) -> WorkspaceBranchCapabilities:
            return WorkspaceBranchCapabilities(detail_code=canary)

    app = CayuApp(
        secret_redactor=SecretRedactor([canary, "unsupported"]),
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="detail-workspace"),
            workspace=DetailWorkspace(tmp_path),
        ),
        default=True,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/environments")

    assert response.status_code == 200
    capabilities = response.json()["environments"][0]["workspace_branch_capabilities"]
    assert capabilities["publication"] == "unsupported"
    assert capabilities["recovery"] == "unsupported"
    assert capabilities["retention"] == "unsupported"
    assert capabilities["detail_code"] == "[REDACTED_SECRET]"
    assert canary not in response.text


def test_environment_capability_projection_rejects_mutated_evidence_safely(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "mutated-server-capability-secret"

    class SecretBearingWrongType:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    candidate = WorkspaceBranchCapabilities()
    object.__setattr__(candidate, "detail_code", SecretBearingWrongType())

    class MutatedCapabilityWorkspace(LocalWorkspace):
        def branch_capabilities(self) -> WorkspaceBranchCapabilities:
            return candidate

    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="mutated-capability"),
            workspace=MutatedCapabilityWorkspace(tmp_path),
        ),
        default=True,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        response = client.get("/api/environments")

    captured = capsys.readouterr()
    assert response.status_code == 200
    assert response.json()["environments"][0]["workspace_branch_capabilities"]["detail_code"] == (
        "workspace_branch_capability_evidence_invalid"
    )
    diagnostic_text = "\n".join(
        [
            response.text,
            repr(response),
            *(str(item.message) for item in caught),
            caplog.text,
            captured.out,
            captured.err,
        ]
    )
    assert canary not in diagnostic_text


def test_server_exposes_agent_environment_and_artifact_inventory(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root)

    async def attach_branch() -> WorkspaceBranch:
        from cayu.workspaces.revisions import (
            WorkspaceRevisionObservationLimits,
            observe_deterministic_workspace,
        )

        observation = await observe_deterministic_workspace(
            workspace,
            observer="server-lifecycle-test",
            limits=WorkspaceRevisionObservationLimits(),
        )
        created = await workspace.create_branch(WorkspaceBranchRequest(baseline=observation))
        assert created.branch is not None
        return created.branch

    attached_branch = asyncio.run(attach_branch())
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
        tools=[UserInputTool(), ExecCommandTool()],
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"tenant": "test"}),
            artifact_store=artifact_store,
            workspace=workspace,
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    agents = client.get("/api/agents")
    assert agents.status_code == 200
    agents_body = agents.json()
    assert agents_body["total_count"] == 1
    assert agents_body["agents"][0]["name"] == "reviewer"
    assert agents_body["agents"][0]["metadata"] == {"team": "platform"}
    assert agents_body["agents"][0]["has_system_prompt"] is True
    tools = {tool["name"]: tool for tool in agents_body["agents"][0]["tools"]}
    assert set(tools) == {"ask_user", "exec_command"}
    assert tools["ask_user"]["workspace_mutation"] is False
    assert tools["exec_command"]["workspace_mutation"] is True
    agent_detail = client.get("/api/agents/reviewer")
    assert agent_detail.status_code == 200
    detail_tools = {tool["name"]: tool for tool in agent_detail.json()["agents"][0]["tools"]}
    assert detail_tools["ask_user"]["workspace_mutation"] is False
    assert detail_tools["exec_command"]["workspace_mutation"] is True

    environments = client.get("/api/environments")
    assert environments.status_code == 200
    environments_body = environments.json()
    assert environments_body["total_count"] == 1
    assert environments_body["environments"][0]["name"] == "local-review"
    assert environments_body["environments"][0]["artifact_store_id"] == "test-artifacts"
    assert environments_body["environments"][0]["workspace_instructions"] == "inline"
    assert environments_body["environments"][0]["workspace_branch_capabilities"] == {
        "detail_code": "process_local_workspace_branches",
        "isolation": True,
        "lifecycle_inspection": "attached",
        "net_changes": True,
        "publication": "cooperative_atomic",
        "recovery": "process_local",
        "retention": "process_local",
    }
    assert environments_body["environments"][0]["workspace_branch_lifecycle"] == {
        "attached_count": 1,
        "statuses": ["active"],
        "truncated": False,
    }
    assert attached_branch.lifecycle_status.value == "active"

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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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


def test_server_artifact_inventory_remains_usable_after_duplicate_store_rejection(
    tmp_path,
) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="duplicate-store")
    second_store = LocalArtifactStore(tmp_path / "second", store_id="duplicate-store")
    app.register_environment(
        Environment(EnvironmentSpec(name="first"), artifact_store=first_store),
        default=True,
    )
    with pytest.raises(ValueError, match="different registered store: duplicate-store"):
        app.register_environment(
            Environment(EnvironmentSpec(name="second"), artifact_store=second_store),
            default=True,
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    capabilities = client.get("/api/contract").json()["capabilities"]
    response = client.get("/api/artifacts")

    assert app.list_environments() == ("first",)
    assert app.get_environment().spec.name == "first"
    assert capabilities["surfaces"]["artifacts"]["read"]["enabled"] is True
    assert response.status_code == 200
    assert [artifact["filename"] for artifact in response.json()["artifacts"]] == ["first.txt"]


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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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


def test_server_artifact_inventory_includes_factory_registered_store(tmp_path) -> None:
    class Factory(EnvironmentFactory):
        def __init__(self, artifact_store: ArtifactStore, *, allow_create: bool = True) -> None:
            self.artifact_store = artifact_store
            self.allow_create = allow_create
            self.create_calls = 0

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.create_calls += 1
            if not self.allow_create:
                raise AssertionError("artifact API must not materialize the environment factory")
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    artifact_store=self.artifact_store,
                )
            )

    artifact_root = tmp_path / "artifacts"
    first_store = LocalArtifactStore(artifact_root, store_id="factory-artifacts")
    first_factory = Factory(first_store)
    first_app = CayuApp()
    first_app.register_environment_factory(
        EnvironmentSpec(name="factory-env"),
        first_factory,
        artifact_store=first_store,
        default=True,
    )
    created = asyncio.run(
        first_app.get_environment_factory().create(
            EnvironmentFactoryRequest(
                session_id="sess_factory",
                agent_name="agent",
                environment_name="factory-env",
            )
        )
    )
    created_store = created.environment.artifact_store
    assert created_store is first_store
    artifact = asyncio.run(
        created_store.put_bytes(
            b"factory output",
            filename="result.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_factory",
            environment_name="factory-env",
        )
    )

    restarted_store = LocalArtifactStore(artifact_root, store_id="factory-artifacts")
    restarted_factory = Factory(restarted_store, allow_create=False)
    restarted_app = CayuApp()
    restarted_app.register_environment_factory(
        EnvironmentSpec(name="factory-env"),
        restarted_factory,
        artifact_store=restarted_store,
        default=True,
    )
    client = TestClient(create_server(restarted_app, config=_LOCAL_SERVER_CONFIG))

    environments = client.get("/api/environments")
    artifacts = client.get("/api/artifacts", params={"session_id": "sess_factory"})
    read = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "factory-artifacts"},
    )

    assert environments.status_code == 200
    assert environments.json()["environments"][0]["artifact_store_id"] == "factory-artifacts"
    assert artifacts.status_code == 200
    assert [item["id"] for item in artifacts.json()["artifacts"]] == [artifact.id]
    assert read.status_code == 200
    assert read.json()["text_preview"] == "factory output"
    assert first_factory.create_calls == 1
    assert restarted_factory.create_calls == 0


def test_server_artifact_inventory_rejects_unbounded_offsets(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_environment(
        Environment(EnvironmentSpec(name="local-review"), artifact_store=artifact_store),
        default=True,
    )

    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
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

    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
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
    access_scope = KnowledgeAccessScope.privileged()
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=access_scope,
        knowledge_review_namespace="project:cayu",
        knowledge_review_labels={"project": "cayu", "tenant": "trusted"},
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    body = pending.json()
    assert [entry["entry_id"] for entry in body["entries"]] == ["pending_git"]
    assert body["entries"][0]["revision"] == 1
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
    assert [chunk["chunk_id"] for chunk in detail_body["chunks"]] == ["pending_git:r1:0"]
    assert detail_body["chunks"][0]["entry_revision"] == 1
    assert detail_body["chunks"][0]["text"] == detail_body["text"]

    review_headers = {"Idempotency-Key": "dashboard-review-pending-git"}
    approved = client.post(
        "/api/knowledge/pending_git/approve",
        headers=review_headers,
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "active"
    assert approved.json()["revision"] == 2

    replay = client.post(
        "/api/knowledge/pending_git/approve",
        headers=review_headers,
    )
    assert replay.status_code == 200
    assert replay.json() == approved.json()
    activation = asyncio.run(
        store.load_activation_receipt(
            "dashboard-review-pending-git",
            access_scope=access_scope,
        )
    )
    assert activation is not None
    assert activation.replayed is False
    assert activation.authority.decision.policy_identity == "cayu:trusted-local-development"
    assert activation.authority.decision.annotations == {
        "channel": "trusted-local-http",
        "identity_source": "local_development",
    }

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
    store = _TestKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_bad",
                text="Do not retain this model-authored knowledge.",
                namespace="project:cayu",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

    rejected = client.post("/api/knowledge/pending_bad/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "archived"

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    assert pending.json()["entries"] == []


def test_server_maps_concurrent_knowledge_review_revision_to_conflict() -> None:
    class RacingKnowledgeStore(InMemoryKnowledgeStore):
        async def approve_pending_entry(self, authority, **kwargs):
            request = authority.request
            assert request.expected_revision is not None
            raise KnowledgeRevisionConflict(
                request.candidate_entry.id,
                expected_revision=request.expected_revision,
                actual_revision=request.expected_revision + 1,
            )

    store = RacingKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_race",
                text="Concurrent review candidate.",
                status=KnowledgeStatus.PENDING,
            )
        ],
        access_scope=KnowledgeAccessScope.privileged(),
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/knowledge/pending_race/approve")

    assert response.status_code == 409
    assert "revision conflict" in response.json()["detail"]


def test_server_knowledge_review_reports_missing_store_and_scope_errors() -> None:
    missing_store = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    assert missing_store.get("/api/knowledge/pending").status_code == 404

    store = _TestKnowledgeStore(
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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    store = _TestKnowledgeStore(
        [
            KnowledgeEntry(
                id="pending_git",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                status=KnowledgeStatus.PENDING,
            )
        ]
    )
    client = TestClient(create_server(CayuApp(knowledge_store=store), config=_LOCAL_SERVER_CONFIG))

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

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
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


def test_server_task_status_omits_private_work_contract_fingerprint() -> None:
    secret = "private-verification-threshold"
    task_store = InMemoryTaskStore()
    contract = work_contract_from_draft(
        WorkContractDraft(
            contract_id="private-status-contract",
            version=1,
            objective=f"Approve only when the score exceeds {secret}.",
            criteria=(
                WorkCriterion(
                    criterion_id="threshold",
                    ordinal=1,
                    description=f"Confirm the private threshold {secret}.",
                ),
            ),
            verifier=CompletionVerifierRef(
                verifier_id="private-status-verifier",
                version="v1",
                configuration_fingerprint="0" * 64,
            ),
            result_resolver=CompletionResultResolverRef(
                resolver_id="private-status-result",
                version="v1",
                configuration_fingerprint="1" * 64,
            ),
        )
    )

    async def setup_task() -> None:
        await task_store.publish_work_contract(contract)
        task = await task_store.create_task(
            TaskCreate(
                task_id="private-status-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await task_store.claim_task("ordinary-worker")
        assert claimed is not None
        assert claimed.id == task.id
        await task_store.hold_claimed_work_contract_task(
            task.id,
            worker_id="ordinary-worker",
            contract=contract.reference(),
        )

    asyncio.run(setup_task())
    client = TestClient(
        create_server(
            CayuApp(
                task_store=task_store,
                secret_redactor=SecretRedactor(secret),
            ),
            config=_LOCAL_SERVER_CONFIG,
        )
    )

    list_response = client.get("/api/tasks")
    detail_response = client.get("/api/tasks/private-status-task")
    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    listed = list_response.json()[0]
    detailed = detail_response.json()
    expected_status = {
        "contract_id": contract.contract_id,
        "contract_version": contract.version,
    }
    assert listed["status_payload"] == expected_status
    assert detailed["status_payload"] == expected_status
    for rendered in (list_response.text, detail_response.text):
        assert secret not in rendered
        assert contract.fingerprint not in rendered


def test_server_task_endpoints_serialize_availability_in_canonical_utc() -> None:
    task_store = InMemoryTaskStore()
    local_time = datetime(
        2026,
        8,
        8,
        9,
        30,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="scheduled_task",
                type="review",
                available_at=local_time,
            )
        )
    )

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    task_list = client.get("/api/tasks")
    task_detail = client.get("/api/tasks/scheduled_task")

    assert task_list.status_code == 200
    assert task_detail.status_code == 200
    assert task_list.json()[0]["available_at"] == "2026-08-08T04:00:00+00:00"
    assert task_detail.json()["available_at"] == "2026-08-08T04:00:00+00:00"


def test_server_task_endpoints_project_retry_series_authority() -> None:
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="retry_task",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=3,
                    max_elapsed_seconds=60,
                    max_total_tokens=100,
                    max_estimated_cost=Decimal("1.25"),
                    cost_currency="usd",
                    initial_backoff_seconds=2,
                ),
            )
        )
    )

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    projected = client.get("/api/tasks").json()[0]["retry_series"]
    detail = client.get("/api/tasks/retry_task").json()["retry_series"]

    assert detail == projected
    assert projected["attempt"] == 1
    assert projected["causal_budget_id"] == projected["series_id"]
    assert projected["attempts_remaining"] == 2
    assert projected["tokens_remaining"] == "100"
    assert projected["estimated_cost_remaining"] == "1.25"
    assert projected["elapsed_deadline"] is not None
    assert projected["next_eligible_at"] is None
    assert projected["disposition"] == "active"
    assert projected["policy"]["cost_currency"] == "USD"


@pytest.mark.parametrize("secret", ["1.25", "0"])
def test_server_task_retry_numeric_authority_survives_secret_collisions(secret: str) -> None:
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id=f"retry_numeric_secret_{secret.replace('.', '_')}",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_total_tokens=100,
                    max_estimated_cost=Decimal("1.25"),
                ),
            )
        )
    )
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    list_response = client.get("/api/tasks")
    detail_response = client.get(f"/api/tasks/retry_numeric_secret_{secret.replace('.', '_')}")

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    for projected in (
        list_response.json()[0]["retry_series"],
        detail_response.json()["retry_series"],
    ):
        assert projected["cumulative_tokens"] == "0"
        assert projected["tokens_remaining"] == "100"
        assert projected["cumulative_estimated_cost"] == "0"
        assert projected["estimated_cost_remaining"] == "1.25"
        assert projected["policy"]["max_total_tokens"] == "100"
        assert projected["policy"]["max_estimated_cost"] == "1.25"


def test_server_task_retry_series_redacts_opaque_causal_budget_authority() -> None:
    secret = "TASK_RETRY_CAUSAL_BUDGET_SECRET_CANARY"
    task_store = InMemoryTaskStore()
    task = asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="retry_secret",
                type="review",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
    )
    assert task.retry_series is not None
    task_store._tasks[task.id] = task.model_copy(
        update={"retry_series": task.retry_series.model_copy(update={"causal_budget_id": secret})}
    )
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/tasks/retry_secret"
    )

    assert response.status_code == 200
    assert response.json()["retry_series"]["causal_budget_id"] == "[REDACTED_SECRET]"
    assert secret not in response.text


def test_server_task_retry_series_redacts_caller_controlled_currency(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "secretcurrency"
    task_store = InMemoryTaskStore()

    async def create_and_settle() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="retry_currency_secret",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    cost_currency=secret,
                ),
            )
        )
        claimed = await task_store.claim_task("currency-worker")
        assert claimed is not None
        assert claimed.retry_series is not None
        await task_store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="currency-worker",
                idempotency_key="currency-settlement",
                causal_budget_id=claimed.retry_series.causal_budget_id,
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )
        )

    asyncio.run(create_and_settle())
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    responses = [
        client.get("/api/tasks"),
        client.get("/api/tasks/retry_currency_secret"),
    ]

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json()[0]["retry_series"]["policy"]["cost_currency"] == (
        "[REDACTED_SECRET]"
    )
    assert responses[1].json()["retry_series"]["policy"]["cost_currency"] == ("[REDACTED_SECRET]")
    assert responses[0].json()[0]["status_payload"]["cost_currency"] == "[REDACTED_SECRET]"
    assert responses[1].json()["status_payload"]["cost_currency"] == "[REDACTED_SECRET]"
    captured = capsys.readouterr()
    diagnostic_text = "\n".join(
        [
            *(record.getMessage() for record in caplog.records),
            *(str(warning.message) for warning in recwarn),
            captured.out,
            captured.err,
            *(response.text for response in responses),
        ]
    )
    assert secret not in diagnostic_text.casefold()


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

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
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
        await task_store.start_task(
            "detail_task",
            session_id="sess_detail",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "detail_task",
                "sess_detail",
            ),
        )
        await task_store.complete_task("detail_task", {"accepted": True})

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

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
    missing_store_client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    missing_store_response = missing_store_client.get("/api/tasks/task_1")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

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
        await task_store.start_task(
            "attached_task",
            session_id="sess_attached",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "attached_task",
                "sess_attached",
            ),
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

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
    missing_store_client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    missing_store_response = missing_store_client.post("/api/tasks/task_1/block")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

    missing_task_response = client.post("/api/tasks/missing_task/block")
    assert missing_task_response.status_code == 404
    assert "missing_task" in missing_task_response.json()["detail"]


def test_server_task_lifecycle_endpoints_validate_request_body() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="task_1", type="review"))

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
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

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.get("/api/sessions/usage_1/usage")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "usage_1",
        "model_steps": 1,
        "tool_calls": 0,
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
    captured_targets: list[ModelTarget | None] = []
    original_run = app.run

    def spy_run(request: RunRequest):
        captured.append(request.max_steps)
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

    assert captured == [20, 7]
    assert captured_targets == [
        None,
        ModelTarget(provider_name="fake", model="request-model"),
    ]


def test_server_resume_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    started = _session_started_event(
        client,
        "/api/run",
        {"prompt": "hello", "max_steps": 42},
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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    body: dict = {"prompt": "hello", "max_steps": bad_value}
    if path == "/api/resume":
        body["session_id"] = "session-does-not-matter"
    response = client.post(path, json=body)
    assert response.status_code == 422


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
        {"currency": "USD", "model_steps": "1", "total_cost": "1"}
    ]
    assert body["cost"]["billing_breakdown"] == {
        "identified_model_steps": "0",
        "groups": [],
        "remainder": None,
        "accuracy": {"kind": "exact", "limit": None, "reason": None},
    }
    assert body["session_breakdown"] == {
        "groups": [
            {
                "session_id": "aggregate-usage",
                "status": "pending",
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
                    "currencies": [{"currency": "USD", "model_steps": "1", "total_cost": "1"}],
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
            from cayu.runtime.aggregates import UsageRollupStoreResult

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


def test_server_pending_actions_lists_blocking_session_work() -> None:
    app = CayuApp()
    private_approval_reason = "pending-approval-policy-secret-canary"
    approval_round_id = f"tround_{'3' * 32}"
    user_input_round_id = f"tround_{'4' * 32}"
    recovery_round_id = f"tround_{'5' * 32}"

    def execution_identity(tool_round_id: str) -> dict[str, str]:
        return {
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_round_id": tool_round_id,
        }

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
        reason: str | None = None,
    ) -> dict[str, object]:
        pending_call = {
            **pending_tool_call(tool_call_id, tool_name),
            "arguments": arguments or {},
            "policy_decision": "require_approval",
        }
        return {
            "pending_tool_approval": {
                **execution_identity(approval_round_id),
                "approval_id": approval_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments or {},
                "agent_name": "assistant",
                "publish_arguments": True,
                "reason": reason,
                "tool_calls": [pending_call],
            },
            "pending_tool_round": {
                **execution_identity(approval_round_id),
                "agent_name": "assistant",
                "tool_calls": [pending_call],
                "policy_state": "planned",
                "policy_context_version": 1,
            },
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
                **execution_identity(user_input_round_id),
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
                **execution_identity(round_id),
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
        created = await app.session_store.create(
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
            pending_user_input = checkpoint.get("pending_user_input")
            if type(pending_user_input) is dict:
                pending_user_input.setdefault("session_id", session_id)
                pending_user_input.setdefault("session_instance_id", created.instance_id)
                pending_user_input.setdefault(
                    "source_interaction_id",
                    f"interaction:{session_id}",
                )
                pending_user_input.setdefault("source_run_epoch", created.run_epoch)
                pending_user_input.setdefault(
                    "execution_profile_fingerprint",
                    "e" * 64,
                )
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
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        **execution_identity(approval_round_id),
                        "approval_id": "approval_1",
                        "tool_call_id": "call_deploy",
                        "approval": {
                            **execution_identity(approval_round_id),
                            "approval_id": "approval_1",
                            "tool_call_id": "call_deploy",
                            "tool_name": "deploy",
                            "reason": private_approval_reason,
                            "arguments": {"service": "api"},
                            "agent_name": "assistant",
                            "tool_calls": [
                                {
                                    **pending_tool_call("call_deploy", "deploy"),
                                    "arguments": {"service": "api"},
                                    "policy_decision": "require_approval",
                                }
                            ],
                        },
                    },
                )
            ],
            checkpoint=approval_checkpoint(
                approval_id="approval_1",
                tool_call_id="call_deploy",
                tool_name="deploy",
                arguments={"service": "api"},
                reason=private_approval_reason,
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
                        **execution_identity(user_input_round_id),
                        "input_id": "input_1",
                        "tool_call_id": "call_ask",
                        "question": "pending-user-input-secret-canary",
                        "options": ["yes", "pending-user-input-secret-canary"],
                    },
                )
            ],
            checkpoint=user_input_checkpoint(
                input_id="input_1",
                tool_call_id="call_ask",
                tool_name="ask_user",
                question="pending-user-input-secret-canary",
                options=["yes", "pending-user-input-secret-canary"],
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
                        **execution_identity(approval_round_id),
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
                        **execution_identity(recovery_round_id),
                        "tool_call_id": "call_charge",
                    },
                ),
                Event(
                    id="manual_tool_round_recovery_event",
                    type=EventType.SESSION_FAILED,
                    session_id="manual_tool_round_recovery",
                    payload={
                        **execution_identity(recovery_round_id),
                        "interruption_type": "runtime_interrupted",
                        "manual_recovery_required": True,
                        "tool_call_id": "call_charge",
                        "tool_name": "charge_card",
                        "error": "tool outcome unknown",
                    },
                ),
            ],
            checkpoint=tool_round_checkpoint(
                round_id=recovery_round_id,
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
    recovered_rounds = []

    async def recover_tool_round(request):
        recovered_rounds.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
        )

    app.recover_tool_round = recover_tool_round
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with patch.object(
        app.session_store,
        "query_events",
        side_effect=AssertionError("pending-action serialization must use its bounded event"),
    ):
        response = client.get("/api/pending-actions")
    assert response.status_code == 200
    body = response.json()
    assert private_approval_reason not in response.text
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
    assert approval["approval_id"] == public_event_linkage_id(
        approval["event"]["sequence"],
        "approval_id",
    )
    assert approval["arguments"] is None
    assert approval["session"]["runtime_build_provenance"]["availability"] == "unavailable"
    user_input = actions_by_session["pending_user_input"]
    assert user_input["kind"] == "user_input"
    assert user_input["input_id"] == public_event_linkage_id(
        user_input["event"]["sequence"],
        "input_id",
    )
    assert user_input["detail"] == "Input required"
    assert user_input["question"] is None
    assert user_input["options"] == []
    assert "pending-user-input-secret-canary" not in response.text
    tool_round = actions_by_session["manual_tool_round_recovery"]
    assert tool_round["kind"] == "manual_recovery"
    assert tool_round["round_id"] == public_event_linkage_id(
        tool_round["event"]["sequence"],
        "tool_round_id",
    )
    assert tool_round["tool_call_id"] == public_event_linkage_id(
        tool_round["event"]["sequence"],
        "tool_call_id",
    )
    assert tool_round["approval_id"] is None
    assert tool_round["input_id"] is None
    assert tool_round["arguments"] is None
    assert tool_round["event"]["type"] == EventType.SESSION_FAILED
    assert tool_round["event"]["payload"] == {}

    pending_approval = asyncio.run(
        app._runtime_session_store.query_pending_actions(
            PendingActionQuery(session_id="pending_approval", limit=1)
        )
    ).actions[0]
    checkpoint_only_approval = pending_approval.model_copy(
        update={
            "source_linkage": {},
            "event": pending_approval.event.model_copy(
                update={
                    "event": pending_approval.event.event.model_copy(update={"payload": {}}),
                },
                deep=True,
            ),
        },
        deep=True,
    )
    checkpoint_only_payload = _serialize_pending_action(app, checkpoint_only_approval)
    assert checkpoint_only_payload["approval_id"] is None
    assert checkpoint_only_payload["round_id"] is None
    assert checkpoint_only_payload["tool_call_id"] is None

    with patch.object(
        app.session_store,
        "query_events",
        side_effect=AssertionError("pending-action serialization must not query per action"),
    ):
        full_page = [_serialize_pending_action(app, pending_approval) for _ in range(200)]
        assert len(full_page) == 200

    contradictory_approval = pending_approval.model_copy(
        update={"approval_id": "different-approval"},
        deep=True,
    )
    with pytest.raises(RuntimeError, match="disagrees with its durable event authority"):
        _serialize_pending_action(app, contradictory_approval)
    approval_recovery = actions_by_session["manual_recovery"]
    assert approval_recovery["kind"] == "manual_recovery"
    assert approval_recovery["arguments"] is None

    filtered = client.get("/api/pending-actions?kind=user_input&q=input")
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
    exact_tool_round = tool_round_exact_body["actions"][0]
    assert exact_tool_round["round_id"] == public_event_linkage_id(
        exact_tool_round["event"]["sequence"],
        "tool_round_id",
    )

    with client.stream(
        "POST",
        "/api/tool-rounds/recover",
        json={
            "session_id": "manual_tool_round_recovery",
            "round_id": exact_tool_round["round_id"],
            "tool_call_id": exact_tool_round["tool_call_id"],
            "outcome": "completed",
            "message": "verified externally",
        },
    ) as recovery_response:
        assert recovery_response.status_code == 200
        list(recovery_response.iter_lines())

    assert len(recovered_rounds) == 1
    assert recovered_rounds[0].round_id == recovery_round_id
    assert recovered_rounds[0].tool_call_id == "call_charge"

    history = client.get("/api/sessions/manual_tool_round_recovery/events").json()["events"]
    failed = next(event for event in history if event["type"] == EventType.SESSION_FAILED)
    assert failed["payload"]["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert failed["payload"]["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert failed["payload"]["tool_round_id"] == tool_round["round_id"]
    assert failed["payload"]["tool_call_id"] == tool_round["tool_call_id"]
    assert recovery_round_id not in repr(failed)

    stale_exact = client.get("/api/pending-actions?session_id=missing_checkpoint")
    assert stale_exact.status_code == 200
    stale_body = stale_exact.json()
    assert stale_body["inspected_candidate_count"] == 0
    assert stale_body["total_count"] == 0


def test_control_plane_redacts_legacy_session_event_transcript_pending_and_task_data() -> None:
    secret = "legacy-control-plane-secret-canary"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    session_id = f"legacy-{secret}-session"
    model_step_id = f"mstep_{'1' * 32}"
    model_attempt_id = f"matt_{'2' * 32}"
    tool_round_id = f"tround_{'3' * 32}"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
                metadata={secret: f"session value {secret}"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="legacy_secret_approval",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        "approval_id": "legacy_approval",
                        "tool_call_id": "legacy_call",
                        "model_step_id": model_step_id,
                        "model_attempt_id": model_attempt_id,
                        "tool_round_id": tool_round_id,
                        "approval": {
                            "approval_id": "legacy_approval",
                            "tool_round_id": tool_round_id,
                            "model_step_id": model_step_id,
                            "model_attempt_id": model_attempt_id,
                            "tool_call_id": "legacy_call",
                            "tool_name": "deploy",
                            "arguments": {secret: f"checkpoint value {secret}"},
                            "agent_name": "assistant",
                            "tool_calls": [
                                {
                                    "tool_call_id": "legacy_call",
                                    "tool_name": "deploy",
                                    "arguments": {secret: f"checkpoint value {secret}"},
                                    "policy_decision": None,
                                    "reason": None,
                                    "metadata": {},
                                    "active_taint_labels": [],
                                }
                            ],
                        },
                    },
                )
            ],
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [
                Message.text("user", f"legacy transcript {secret}"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        ProviderStatePart(
                            provider="fake",
                            state={secret: f"provider state {secret}"},
                        ),
                    ),
                ),
            ],
        )
        await app.session_store.checkpoint(
            session_id,
            {
                "pending_tool_approval": {
                    "approval_id": "legacy_approval",
                    "tool_round_id": tool_round_id,
                    "model_step_id": model_step_id,
                    "model_attempt_id": model_attempt_id,
                    "tool_call_id": "legacy_call",
                    "tool_name": "deploy",
                    "arguments": {secret: f"checkpoint value {secret}"},
                    "agent_name": "assistant",
                    "publish_arguments": True,
                    "tool_calls": [
                        {
                            "tool_call_id": "legacy_call",
                            "tool_name": "deploy",
                            "arguments": {secret: f"checkpoint value {secret}"},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await task_store.create_task(
            TaskCreate(
                task_id="legacy_secret_task",
                type="review",
                title=f"legacy task {secret}",
                description=f"legacy task description {secret}",
                input={secret: f"task value {secret}"},
                metadata={secret: f"task metadata {secret}"},
            )
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    public_session_id = app.project_session_id_for_exposure(session_id)
    assert public_session_id != session_id

    responses = [
        client.get(f"/api/sessions/{public_session_id}"),
        client.get(f"/api/sessions/{public_session_id}/events"),
        client.get(f"/api/sessions/{public_session_id}/transcript"),
        client.get(f"/api/pending-actions?session_id={public_session_id}"),
        client.get("/api/tasks"),
        client.get("/api/tasks/legacy_secret_task"),
        client.get("/api/sessions"),
    ]

    for response in responses:
        assert response.status_code == 200
        rendered = json.dumps(response.json(), sort_keys=True)
        assert secret not in rendered

    assert REDACTED_SECRET in "".join(
        json.dumps(response.json(), sort_keys=True) for response in responses
    )

    pending_body = responses[3].json()
    assert pending_body["actions"][0]["session"]["id"] == public_session_id
    assert pending_body["actions"][0]["arguments"] is None
    assert pending_body["next_cursor"] is None
    assert responses[0].json()["id"] == public_session_id
    assert responses[-1].json()["sessions"][0]["id"] == public_session_id


def test_control_plane_rejects_secret_bearing_session_cursor_authority() -> None:
    secret = "cursor-secret-canary"
    secret_session_id = f"session-{secret}"
    safe_session_id = "session-safe-cursor-boundary"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def seed_pending_approval(session_id: str, suffix: str) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        await store.append_event(
            session_id,
            Event(
                id=f"approval-event-{suffix}",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=session_id,
                tool_name="deploy",
                payload={
                    "approval": {
                        "approval_id": f"approval-{suffix}",
                        "tool_name": "deploy",
                        "arguments": {},
                    }
                },
            ),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_tool_approval": {
                    "approval_id": f"approval-{suffix}",
                    "tool_call_id": f"call-{suffix}",
                    "tool_name": "deploy",
                    "arguments": {},
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": f"call-{suffix}",
                            "tool_name": "deploy",
                            "arguments": {},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )

    async def seed() -> None:
        # Pending-action and session listings sort newest first by default, so
        # the second record becomes the keyset boundary for a one-item page.
        await seed_pending_approval(safe_session_id, "safe")
        await seed_pending_approval(secret_session_id, "legacy")

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    responses = [
        client.get("/api/sessions", params={"limit": 1}),
        client.post("/api/sessions/summary", params={"limit": 1}),
        client.get("/api/pending-actions", params={"limit": 1}),
    ]

    for response in responses:
        assert response.status_code == 409
        rendered = json.dumps(response.json(), sort_keys=True)
        assert secret not in rendered
        assert response.json()["detail"] == (
            "Session pagination cannot continue because its cursor authority "
            "contains a configured workload secret."
        )


def test_control_plane_preserves_secret_free_opaque_custom_store_cursor() -> None:
    class OpaqueCursorStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def list_sessions(self, query=None):
            del query
            return SessionListResult(
                sessions=[],
                next_cursor="custom-store:opaque-page-2",
                total_count=0,
            )

    app = CayuApp(
        session_store=OpaqueCursorStore(),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "custom-store:opaque-page-2"


def test_control_plane_short_secret_preserves_typed_response_envelopes() -> None:
    secret = "a"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    session_id = "sess_001"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="bot",
                session_id=session_id,
                messages=[],
                metadata={"data": "safe"},
            ),
            identity=SessionIdentity(provider_name="prov", model="mdl"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id=secret * 100,
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="bot",
                    payload={"data": "safe"},
                )
            ],
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [Message.text("user", "safe")],
        )
        await task_store.create_task(
            TaskCreate(
                task_id="tsk_001",
                type="review",
                title="safe",
                input={"data": "safe"},
            )
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_response = client.get(f"/api/sessions/{session_id}")
    events_response = client.get(f"/api/sessions/{session_id}/events")
    transcript_response = client.get(f"/api/sessions/{session_id}/transcript")
    task_list_response = client.get("/api/tasks")
    task_detail_response = client.get("/api/tasks/tsk_001")

    assert session_response.status_code == 200
    assert events_response.status_code == 200
    assert transcript_response.status_code == 200
    assert task_list_response.status_code == 200
    assert task_detail_response.status_code == 200

    session_body = session_response.json()
    assert "agent_name" in session_body
    assert session_body["status"] == "pending"
    redacted_untrusted = app.redact_json({"data": "safe"})
    assert session_body["metadata"] == redacted_untrusted

    event_body = events_response.json()["events"][0]
    assert event_body["type"] == "session.started"
    assert len(event_body["id"]) <= EVENT_ID_MAX_CHARS
    assert "payload" in event_body

    transcript_body = transcript_response.json()["messages"][0]
    assert transcript_body["role"] == "user"
    assert transcript_body["content"][0]["type"] == "text"

    assert task_list_response.json()[0]["status"] == "pending"
    assert task_detail_response.json()["input"] == redacted_untrusted


def test_control_plane_short_secret_preserves_file_attachment_protocol() -> None:
    secret = "image"
    session_id = "legacy_file_attachment_protocol"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    attachment = FileAttachment(
        artifact_id="safe-artifact",
        kind=FileAttachmentKind.IMAGE,
        filename=f"private-{secret}-file.png",
        content_type="image/png",
        size_bytes=12,
        metadata={f"private-{secret}-key": f"private {secret} value"},
    )

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session_id,
            [
                Message(
                    role=MessageRole.USER,
                    content=(
                        FilePart(
                            attachment=attachment.model_dump(mode="json"),
                        ),
                    ),
                )
            ],
        )

    asyncio.run(seed())
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        f"/api/sessions/{session_id}/transcript"
    )

    assert response.status_code == 200
    payload = response.json()["messages"][0]["content"][0]["attachment"]
    serialized = FileAttachment.model_validate(payload)
    assert serialized.kind is FileAttachmentKind.IMAGE
    assert serialized.content_type == "image/png"
    assert serialized.type == attachment.type
    assert secret not in serialized.filename
    assert REDACTED_SECRET in serialized.filename
    assert secret not in repr(serialized.metadata)
    assert REDACTED_SECRET in repr(serialized.metadata)


def test_pending_action_issue_preserves_typed_timestamp_for_short_secret() -> None:
    observed_at = datetime(2026, 7, 27, tzinfo=UTC)

    class IssueStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del query
            del checkpoint_root_guard
            return PendingActionListResult(
                issues=[
                    PendingActionIssue(
                        code=PendingActionIssueCode.SOURCE_INVALID,
                        session_id="safe-session",
                        agent_name="safe-agent",
                        status=SessionStatus.INTERRUPTED,
                        updated_at=observed_at,
                        detail="safe detail",
                    )
                ],
                total_count=1,
                inspected_candidate_count=1,
            )

    app = CayuApp(
        session_store=IssueStore(),
        secret_redactor=SecretRedactor("2"),
        enable_logging=False,
    )
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/pending-actions"
    )

    assert response.status_code == 200
    assert (
        datetime.fromisoformat(response.json()["issues"][0]["updated_at"].replace("Z", "+00:00"))
        == observed_at
    )


def test_control_plane_preserves_runtime_timestamps_for_short_secret() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor("2"),
        enable_logging=False,
    )
    session_id = "safe-session"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="bot",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="safe-event",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                )
            ],
        )
        task = await task_store.create_task(
            TaskCreate(
                task_id="safe-task",
                type="review",
            )
        )
        await task_store.claim_task("safe-worker")
        await task_store.complete_task(
            task.id,
            {"status": "done"},
            worker_id="safe-worker",
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session = client.get(f"/api/sessions/{session_id}")
    event = client.get(f"/api/sessions/{session_id}/events")
    tasks = client.get("/api/tasks")
    task = client.get("/api/tasks/safe-task")

    assert session.status_code == 200
    assert event.status_code == 200
    assert tasks.status_code == 200
    assert task.status_code == 200
    for value in (
        session.json()["created_at"],
        session.json()["updated_at"],
        event.json()["events"][0]["timestamp"],
        tasks.json()[0]["created_at"],
        tasks.json()[0]["updated_at"],
        tasks.json()[0]["completed_at"],
        task.json()["started_at"],
    ):
        assert value is not None
        datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_server_pending_actions_uses_one_store_native_query() -> None:
    class PendingActionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.pending_query_count = 0

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            self.pending_query_count += 1
            return PendingActionListResult()

        async def list_sessions(self, query=None):
            raise AssertionError("pending-action route must not list candidate sessions")

        async def query_events(self, query=None):
            raise AssertionError("pending-action route must not query per-session events")

        async def load_checkpoint(self, session_id: str):
            raise AssertionError("pending-action route must not load per-session checkpoints")

    store = PendingActionStore()
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))

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
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            from cayu.runtime.sessions import PendingActionResultTooLarge

            raise PendingActionResultTooLarge(2 * 1024 * 1024)

    client = TestClient(
        create_server(
            CayuApp(session_store=OversizedPendingActionStore()), config=_LOCAL_SERVER_CONFIG
        )
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 413
    assert "2097152-byte result limit" in response.json()["detail"]


def test_server_pending_actions_reports_future_checkpoint_without_exposing_contents() -> None:
    app = CayuApp()
    session_id = "future_checkpoint"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)
        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await app.session_store.checkpoint(
                session_id,
                {
                    "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                    "pending_user_input": {"secret": "must-not-escape"},
                },
            )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/pending-actions")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "checkpoint_kind": "root",
        "observed_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
        "reason": "checkpoint_schema_version_too_new",
        "recovery_disposition": "cannot_migrate",
        "resumable_in_place": False,
        "session_id": session_id,
        "supported_max_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "supported_min_version": 1,
    }
    assert "must-not-escape" not in response.text
    error_schema = client.get("/openapi.json").json()["paths"]["/api/pending-actions"]["get"][
        "responses"
    ]["409"]["content"]["application/json"]["schema"]
    assert error_schema == {"$ref": "#/components/schemas/CheckpointCompatibilityErrorResponse"}


def test_server_pending_actions_rejects_invalid_cursor_as_400() -> None:
    client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/pending-actions?cursor=not-a-cursor")

    assert response.status_code == 400
    assert "Invalid session cursor" in response.json()["detail"]


def test_server_pending_actions_does_not_misclassify_store_failure_as_400() -> None:
    class FailingPendingActionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            raise ValueError("persisted pending-action projection is corrupt")

    client = TestClient(
        create_server(
            CayuApp(session_store=FailingPendingActionStore()), config=_LOCAL_SERVER_CONFIG
        ),
        raise_server_exceptions=False,
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 500


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
        "currency": "USD",
        "model_steps": 1,
        "priced_model_steps": 1,
        "unpriced_model_steps": 0,
        "total_cost": "0.000011",
        "line_items": [
            {
                "model_step": 1,
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
            "arguments": {"path": "notes/result.txt"},
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


def test_generated_session_and_interaction_aliases_remain_distinct_and_addressable() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def seed() -> tuple[str, str, str]:
        first_session_events = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
                max_steps=20,
            ),
        )
        first_session_id = first_session_events[0].session_id
        assert {event.session_id for event in first_session_events} == {first_session_id}
        assert (
            len(
                {
                    event.interaction_id
                    for event in first_session_events
                    if event.interaction_id is not None
                }
            )
            == 1
        )
        second_session_events = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "other")],
                max_steps=20,
            ),
        )
        return first_session_id, first_session_events[0].id, second_session_events[0].session_id

    first_session_id, first_event_id, second_session_id = asyncio.run(seed())
    assert public_event_envelope_alias_field(first_session_id) == "session_id"
    assert public_event_envelope_alias_field(second_session_id) == "session_id"
    assert first_session_id != second_session_id

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    replay = client.post(
        "/api/resume",
        json={"session_id": first_session_id, "prompt": "must not dispatch"},
        headers={"Last-Event-ID": f"{first_session_id}:{first_event_id}"},
    )
    assert replay.status_code == 200

    response = client.get(f"/api/sessions/{first_session_id}/interactions?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == first_session_id
    assert len(body["interactions"]) == 1
    interaction_ids = [item["interaction_id"] for item in body["interactions"]]
    assert len(set(interaction_ids)) == 1
    assert all(
        public_event_envelope_alias_field(interaction_id) == "interaction_id"
        for interaction_id in interaction_ids
    )

    interaction = body["interactions"][0]
    detail = client.get(
        f"/api/sessions/{first_session_id}/interactions/{interaction['interaction_id']}"
    )
    assert detail.status_code == 200
    assert detail.json() == interaction

    events = client.get(
        f"/api/sessions/{first_session_id}/events",
        params={"interaction_id": interaction["interaction_id"]},
    )
    assert events.status_code == 200
    event_body = events.json()
    assert event_body["session_id"] == first_session_id
    assert event_body["events"]
    assert {event["session_id"] for event in event_body["events"]} == {first_session_id}
    assert {event["interaction_id"] for event in event_body["events"]} == {
        interaction["interaction_id"]
    }

    transcript = client.get(
        f"/api/sessions/{first_session_id}/transcript",
        params={"interaction_id": interaction["interaction_id"]},
    )
    assert transcript.status_code == 200
    assert transcript.json()["session_id"] == first_session_id
    assert transcript.json()["messages"]
    assert {
        message["interaction_id"]
        for message in transcript.json()["messages"]
        if message["interaction_id"] is not None
    } == {interaction["interaction_id"]}
    assert (
        client.get(
            f"/api/sessions/{second_session_id}/interactions/{interaction['interaction_id']}"
        ).status_code
        == 409
    )

    resumed = client.post(
        "/api/resume",
        json={"session_id": first_session_id, "prompt": "continue normally"},
    )
    assert resumed.status_code == 200
    assert '"type":"session.completed"' in resumed.text


def test_alias_shaped_legacy_private_authority_requires_positive_projection() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("_"), enable_logging=False)
    private_session_id = "cayu_authority_v1.legacy.session_id." + "A" * 43
    private_interaction_id = "cayu_authority_v1.legacy.interaction_id." + "B" * 43

    async def scenario() -> tuple[str, str]:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_transcript_messages(
            private_session_id,
            [Message.text("user", "legacy")],
            interaction_id=private_interaction_id,
        )
        return (
            app.project_session_id_for_exposure(private_session_id),
            app.project_interaction_id_for_exposure(
                private_interaction_id,
                session_id=private_session_id,
            ),
        )

    public_session_id, public_interaction_id = asyncio.run(scenario())
    assert public_session_id != private_session_id
    assert public_interaction_id != private_interaction_id
    assert public_event_envelope_alias_field(public_session_id) == "session_id"
    assert public_event_envelope_alias_field(public_interaction_id) == "interaction_id"


def test_transcript_only_raw_interaction_alias_conflict_is_rejected() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    session_id = "session"
    private_interaction_id = "other-private-interaction"
    conflicting_raw_id = app.project_interaction_id_for_exposure(
        private_interaction_id,
        session_id=session_id,
    )

    async def scenario() -> None:
        await app.session_store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=private_interaction_id,
            ),
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [Message.text("user", "legacy")],
            interaction_id=conflicting_raw_id,
        )
        with pytest.raises(ValueError, match="ambiguous"):
            await app._resolve_public_interaction_id(
                session_id=session_id,
                value=conflicting_raw_id,
            )

    asyncio.run(scenario())


def test_in_memory_nested_turn_interaction_alias_round_trips() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    session_id = "session"
    private_interaction_id = "nested-private-interaction"

    async def scenario() -> None:
        await app.session_store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.TURN_COMPLETED,
                session_id=session_id,
                payload={"interaction_ids": [private_interaction_id]},
            ),
        )
        public_interaction_id = app.project_interaction_id_for_exposure(
            private_interaction_id,
            session_id=session_id,
        )
        assert (
            await app._resolve_public_interaction_id(
                session_id=session_id,
                value=public_interaction_id,
            )
            == private_interaction_id
        )

    asyncio.run(scenario())


def test_server_generated_session_authority_survives_short_secret_collision() -> None:
    store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        secret_redactor=SecretRedactor("-"),
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    events = _sse_events(client, "/api/run", {"agent": "assistant", "prompt": "first"})
    public_session_id = events[0]["session_id"]
    public_interaction_id = next(
        event["interaction_id"] for event in events if event["type"] == "interaction.started"
    )
    turn_completed = next(event for event in events if event["type"] == "turn.completed")

    async def private_state() -> tuple[str, str, list[str]]:
        sessions = await store.list_sessions()
        assert len(sessions.sessions) == 1
        private_session_id = sessions.sessions[0].id
        turn_records = await store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_type=EventType.TURN_COMPLETED,
            )
        )
        interaction_records = await store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_type=EventType.INTERACTION_STARTED,
            )
        )
        assert len(turn_records) == 1
        assert len(interaction_records) == 1
        private_interaction_id = interaction_records[0].event.interaction_id
        assert private_interaction_id is not None
        private_interaction_ids = turn_records[0].event.payload["interaction_ids"]
        assert type(private_interaction_ids) is list
        assert all(type(value) is str for value in private_interaction_ids)
        return private_session_id, private_interaction_id, private_interaction_ids

    private_session_id, private_interaction_id, private_interaction_ids = asyncio.run(
        private_state()
    )
    tasks = asyncio.run(task_store.list_tasks())

    assert private_session_id.startswith("session-")
    assert public_session_id == app.project_session_id_for_exposure(private_session_id)
    assert private_interaction_ids == [private_interaction_id]
    assert turn_completed["payload"]["interaction_ids"] == [public_interaction_id]
    assert len(tasks) == 1
    assert tasks[0].session_id == private_session_id
    assert tasks[0].status is TaskStatus.COMPLETED

    resumed = _sse_events(
        client,
        "/api/resume",
        {"session_id": public_session_id, "prompt": "second"},
    )
    assert resumed[-1]["type"] == "session.completed"
    assert {event["session_id"] for event in resumed} == {public_session_id}


def test_generated_session_causal_budget_alias_round_trips_through_http() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    private_session_id = "generated-private-session"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                causal_budget_id=private_session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    public_session_id = app.project_session_id_for_exposure(private_session_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{public_session_id}/usage"),
        client.post(
            f"/api/causal-budgets/{public_session_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{public_session_id}/summary",
            json=pricing_body,
        ),
    )
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["causal_budget_id"] == public_session_id
    assert responses[0].json()["session_ids"] == [public_session_id]
    assert responses[1].json()["session_ids"] == [public_session_id]
    assert responses[2].json()["usage"]["causal_budget_id"] == public_session_id
    assert responses[2].json()["cost"]["causal_budget_id"] == public_session_id


def test_public_session_lineage_aliases_round_trip_through_list_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor("private"),
        enable_logging=False,
    )
    private_root_id = "private-root-session"
    child_id = "child-session"

    async def seed() -> None:
        identity = SessionIdentity(provider_name="fake", model="fakemodel")
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_root_id,
                messages=[],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=child_id,
                parent_session_id=private_root_id,
                causal_budget_id=private_root_id,
                messages=[],
            ),
            identity=identity,
        )

    asyncio.run(seed())
    public_root_id = app.project_session_id_for_exposure(private_root_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    unfiltered = client.get("/api/sessions", params={"order_by": "created_at_asc"})
    assert unfiltered.status_code == 200, unfiltered.text
    sessions = unfiltered.json()["sessions"]
    assert [session["id"] for session in sessions] == [public_root_id, child_id]
    assert sessions[0]["causal_budget_id"] == public_root_id
    assert sessions[1]["parent_session_id"] == public_root_id
    assert sessions[1]["causal_budget_id"] == public_root_id

    parent_list = client.get(
        "/api/sessions",
        params={"parent_session_id": public_root_id},
    )
    parent_summary = client.post(
        "/api/sessions/summary",
        params={"parent_session_id": public_root_id},
    )
    causal_list = client.get(
        "/api/sessions",
        params={"causal_budget_id": public_root_id, "order_by": "created_at_asc"},
    )
    causal_summary = client.post(
        "/api/sessions/summary",
        params={"causal_budget_id": public_root_id, "order_by": "created_at_asc"},
    )

    assert parent_list.status_code == 200, parent_list.text
    assert [session["id"] for session in parent_list.json()["sessions"]] == [child_id]
    assert parent_summary.status_code == 200, parent_summary.text
    assert [item["session"]["id"] for item in parent_summary.json()["sessions"]] == [child_id]
    assert causal_list.status_code == 200, causal_list.text
    assert [session["id"] for session in causal_list.json()["sessions"]] == [
        public_root_id,
        child_id,
    ]
    assert causal_summary.status_code == 200, causal_summary.text
    assert [item["session"]["id"] for item in causal_summary.json()["sessions"]] == [
        public_root_id,
        child_id,
    ]


def test_alias_shaped_raw_causal_budget_round_trips_through_all_http_summaries() -> None:
    session_store = InMemorySessionStore()
    raw_causal_budget_id = session_store.public_authority_alias_codec.encode(
        "not-a-session",
        field_name="session_id",
    )
    app = CayuApp(session_store=session_store, enable_logging=False)

    async def seed() -> None:
        await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="raw-causal-budget-session",
                causal_budget_id=raw_causal_budget_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{raw_causal_budget_id}/usage"),
        client.post(
            f"/api/causal-budgets/{raw_causal_budget_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{raw_causal_budget_id}/summary",
            json=pricing_body,
        ),
    )
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["causal_budget_id"] == raw_causal_budget_id


def test_non_session_causal_budget_secret_is_redacted_from_all_http_summaries() -> None:
    private_causal_budget_id = "legacy-secret-budget"
    app = CayuApp(secret_redactor=SecretRedactor("secret"), enable_logging=False)

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-one",
                causal_budget_id=private_causal_budget_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{private_causal_budget_id}/usage"),
        client.post(
            f"/api/causal-budgets/{private_causal_budget_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{private_causal_budget_id}/summary",
            json=pricing_body,
        ),
    )
    public_causal_budget_id = f"legacy-{REDACTED_SECRET}-budget"

    for response in responses:
        assert response.status_code == 200, response.text
        assert private_causal_budget_id not in response.text
        assert response.json()["causal_budget_id"] == public_causal_budget_id
    assert responses[2].json()["usage"]["causal_budget_id"] == public_causal_budget_id
    assert responses[2].json()["cost"]["causal_budget_id"] == public_causal_budget_id


def test_direct_app_resume_accepts_a_generated_public_session_alias() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[list[Event], list[Event]]:
        initial = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
            ),
        )
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=initial[0].session_id,
                    messages=[Message.text("user", "second")],
                )
            )
        ]
        return initial, resumed

    initial, resumed = asyncio.run(scenario())

    assert public_event_envelope_alias_field(initial[0].session_id) == "session_id"
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert {event.session_id for event in resumed} == {initial[0].session_id}


def test_direct_app_fork_and_dispatch_accept_generated_public_session_alias() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[str, list[Event], list[Event], list[Event], Any, Any]:
        initial = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
            ),
        )
        public_session_id = initial[0].session_id
        forked = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id=public_session_id)
            )
        ]
        grandchild = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id=forked[0].session_id)
            )
        ]
        inline = [
            event
            async for event in app.dispatch_inline(
                DispatchRequest(
                    session_id=public_session_id,
                    dispatch_id="inline_dispatch",
                    messages=[Message.text("user", "inline dispatch")],
                )
            )
        ]
        handle = await app.dispatch(
            DispatchRequest(
                session_id=public_session_id,
                dispatch_id="submitted_dispatch",
                messages=[Message.text("user", "submitted dispatch")],
            )
        )
        stored_fork_id = await app._resolve_public_session_id(forked[0].session_id)
        stored_fork = await app.session_store.load(stored_fork_id)
        assert stored_fork is not None
        return public_session_id, forked, grandchild, inline, handle, stored_fork

    public_session_id, forked, grandchild, inline, handle, stored_fork = asyncio.run(scenario())

    assert public_event_envelope_alias_field(public_session_id) == "session_id"
    assert [event.type for event in forked] == [EventType.SESSION_FORKED]
    assert public_event_envelope_alias_field(forked[0].session_id) == "session_id"
    assert [event.type for event in grandchild] == [EventType.SESSION_FORKED]
    assert public_event_envelope_alias_field(grandchild[0].session_id) == "session_id"
    assert grandchild[0].session_id != forked[0].session_id
    assert inline[-1].type is EventType.SESSION_COMPLETED, inline[-1].payload
    assert {event.session_id for event in inline} == {public_session_id}
    assert handle.status is DispatchStatus.COMPLETED
    assert handle.session_id == public_session_id
    assert stored_fork.runtime_build_provenance.origin.value != "legacy_record"
    assert "-" in stored_fork.runtime_build_provenance.recipe


def test_exact_secret_source_alias_rejects_fork_before_child_creation() -> None:
    private_session_id = "exact-session-secret"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(private_session_id),
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[str, ...]:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[Message.text("user", "legacy")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        await store.update_status(private_session_id, SessionStatus.COMPLETED)
        public_session_id = app.project_session_id_for_exposure(private_session_id)
        with pytest.raises(ValueError, match="source_session.id contains a workload secret"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(source_session_id=public_session_id)
                )
            ]
        sessions = await store.list_sessions()
        return tuple(session.id for session in sessions.sessions)

    assert asyncio.run(scenario()) == (private_session_id,)


def test_dispatch_projects_a_private_session_id_returned_by_a_dispatcher() -> None:
    class EchoDispatcher(Dispatcher):
        async def submit(self, runtime, request):
            del runtime
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                backend="echo",
            )

    app = CayuApp(
        dispatcher=EchoDispatcher(),
        secret_redactor=SecretRedactor("-"),
        enable_logging=False,
    )
    handle = asyncio.run(
        app.dispatch(
            DispatchRequest(
                session_id="private-session-id",
                dispatch_id="safe_dispatch",
                messages=[Message.text("user", "queued")],
            )
        )
    )

    assert handle.session_id == app.project_session_id_for_exposure("private-session-id")


def test_public_session_alias_resolution_uses_the_store_index_without_a_scan() -> None:
    class EndlessSessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.list_calls = 0

        async def list_sessions(self, query=None):
            del query
            self.list_calls += 1
            return SessionListResult(
                sessions=[],
                next_cursor=f"opaque-page-{self.list_calls}",
            )

    store = EndlessSessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor("private"),
        enable_logging=False,
    )
    alias = app.project_session_id_for_exposure("private-session")

    async def resolve() -> None:
        with pytest.raises(ValueError, match="was not found"):
            await app._resolve_public_session_id(alias)

    asyncio.run(resolve())

    assert store.list_calls == 0


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


def test_mount_cayu_composes_background_interruption_drain() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    drain_timeouts = []
    recovery_drain_timeouts = []
    environment_drain_timeouts = []
    knowledge_drain_timeouts = []
    knowledge_seals = 0
    resume_calls = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_before):
        resume_calls.append(interrupting_inactive_before)
        return 0

    async def drain_background_interruptions(*, timeout_s):
        drain_timeouts.append(timeout_s)
        return True

    async def drain_environment_cleanups(*, timeout_s):
        environment_drain_timeouts.append(timeout_s)
        return True

    async def drain_recovery_cleanups(*, timeout_s):
        recovery_drain_timeouts.append(timeout_s)
        return True

    def seal_knowledge_publications():
        nonlocal knowledge_seals
        knowledge_seals += 1

    async def drain_knowledge_publications(*, timeout_s):
        knowledge_drain_timeouts.append(timeout_s)
        return True

    cayu_app.drain_background_interruptions = drain_background_interruptions
    cayu_app.drain_recovery_cleanups = drain_recovery_cleanups
    cayu_app.drain_environment_cleanups = drain_environment_cleanups
    cayu_app.seal_knowledge_publications = seal_knowledge_publications
    cayu_app.drain_knowledge_publications = drain_knowledge_publications
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

    assert drain_timeouts == [2.5]
    assert recovery_drain_timeouts == [2.5]
    assert environment_drain_timeouts == [2.5]
    assert knowledge_seals == 1
    assert knowledge_drain_timeouts == [1.5]
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
    from cayu import ResolutionActorSource

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


def test_action_routes_resolve_schema_owned_nested_public_linkage_aliases() -> None:
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        for session_id, payload in (
            (
                "session_nested_approval_alias",
                {
                    "interruption_type": "tool_approval_required",
                    "approval": {
                        "approval_id": "approval-private",
                        "tool_round_id": "round-private",
                        "tool_call_id": "call-private",
                    },
                },
            ),
            (
                "session_nested_input_alias",
                {
                    "interruption_type": "user_input_required",
                    "user_input": {
                        "input_id": "input-private",
                        "tool_call_id": "input-call-private",
                    },
                },
            ),
        ):
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
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    payload=payload,
                ),
            )
            await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    approval_sequence = asyncio.run(
        app.session_store.query_events(
            EventQuery(session_id="session_nested_approval_alias", limit=1)
        )
    )[0].sequence
    input_sequence = asyncio.run(
        app.session_store.query_events(EventQuery(session_id="session_nested_input_alias", limit=1))
    )[0].sequence
    approvals = []
    inputs = []

    async def resolve_approval(request):
        approvals.append(request)
        yield await app.emit_event(
            Event(
                type=EventType.SESSION_RESUMED,
                session_id=request.session_id,
            )
        )

    async def resolve_input(request):
        inputs.append(request)
        yield await app.emit_event(
            Event(
                type=EventType.SESSION_RESUMED,
                session_id=request.session_id,
            )
        )

    app.resolve_tool_approval = resolve_approval
    app.resolve_user_input = resolve_input
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_nested_approval_alias",
            "approval_id": public_event_linkage_id(approval_sequence, "approval_id"),
            "tool_round_id": public_event_linkage_id(approval_sequence, "tool_round_id"),
            "tool_call_id": public_event_linkage_id(approval_sequence, "tool_call_id"),
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    with client.stream(
        "POST",
        "/api/user-input/resolve",
        json={
            "session_id": "session_nested_input_alias",
            "input_id": public_event_linkage_id(input_sequence, "input_id"),
            "answer": "continue",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    malformed_alias = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_nested_approval_alias",
            "approval_id": f"cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}:approval_id",
            "tool_round_id": public_event_linkage_id(
                approval_sequence,
                "tool_round_id",
            ),
            "tool_call_id": public_event_linkage_id(
                approval_sequence,
                "tool_call_id",
            ),
            "decision": "approve",
        },
    )

    assert malformed_alias.status_code == 409
    assert "malformed" in malformed_alias.json()["detail"]
    assert (
        approvals[0].approval_id,
        approvals[0].tool_round_id,
        approvals[0].tool_call_id,
    ) == (
        "approval-private",
        "round-private",
        "call-private",
    )
    assert inputs[0].input_id == "input-private"
    assert len(approvals) == 1


def test_action_linkage_disambiguates_legacy_raw_values_from_public_aliases() -> None:
    app = CayuApp(enable_logging=False)

    async def seed(
        session_id: str,
        *,
        approval_id: str,
        alias_target: str | None = None,
    ) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        if alias_target is not None:
            await app.session_store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    payload={
                        "approval_id": alias_target,
                        "tool_round_id": "alias-round",
                        "tool_call_id": "alias-call",
                    },
                ),
            )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
                payload={
                    "interruption_type": "tool_approval_required",
                    "approval": {
                        "approval_id": approval_id,
                        "tool_round_id": "legacy-round",
                        "tool_call_id": "legacy-call",
                    },
                },
            ),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(
        seed(
            "session_linkage_raw_only",
            approval_id="cayu_event_01:approval_id",
        )
    )
    asyncio.run(
        seed(
            "session_linkage_same",
            approval_id="cayu_event_2:approval_id",
        )
    )
    asyncio.run(
        seed(
            "session_linkage_conflict",
            approval_id="cayu_event_3:approval_id",
            alias_target="different-private-approval",
        )
    )
    legacy_approval_ids = {
        "session_linkage_raw_only": "cayu_event_01:approval_id",
        "session_linkage_same": "cayu_event_2:approval_id",
        "session_linkage_conflict": "cayu_event_3:approval_id",
    }

    async def pending_actions(query):
        approval_id = legacy_approval_ids[query.session_id]
        return SimpleNamespace(actions=[SimpleNamespace(approval_id=approval_id)])

    app.session_store.query_pending_actions = pending_actions
    captured = []

    async def resolve(request):
        captured.append(request)
        yield await app.emit_event(
            Event(type=EventType.SESSION_RESUMED, session_id=request.session_id)
        )

    app.resolve_tool_approval = resolve
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    for session_id, approval_id in (
        ("session_linkage_raw_only", "cayu_event_01:approval_id"),
        ("session_linkage_same", "cayu_event_2:approval_id"),
    ):
        with client.stream(
            "POST",
            "/api/tool-approvals/resolve",
            json={
                "session_id": session_id,
                "approval_id": approval_id,
                "tool_round_id": "legacy-round",
                "tool_call_id": "legacy-call",
                "decision": "approve",
            },
        ) as response:
            if response.status_code != 200:
                response.read()
            assert response.status_code == 200, f"{session_id}: {response.text}"
            list(response.iter_lines())

    conflict = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_linkage_conflict",
            "approval_id": "cayu_event_3:approval_id",
            "tool_round_id": "legacy-round",
            "tool_call_id": "legacy-call",
            "decision": "approve",
        },
    )

    assert [request.approval_id for request in captured] == [
        "cayu_event_01:approval_id",
        "cayu_event_2:approval_id",
    ]
    assert conflict.status_code == 409
    assert "ambiguous" in conflict.json()["detail"]


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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    # An open-access body can assert an identity but never verified/system
    # provenance: the server re-stamps the source as "request".
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
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
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
            "resolved_by": {"subject": "cayu:approval-expiry", "source": "system"},
        },
    )
    assert response.status_code == 400
    assert "reserved for system actors" in response.json()["detail"]
    assert len(captured) == 1

    # No body actor means no provenance under open access.
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
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


def _lifecycle_store_and_client(seed) -> tuple[InMemorySessionStore, TestClient]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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
        await asyncio.wait_for(factory.started.wait(), timeout=1)
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
        await asyncio.wait_for(provider.started.wait(), timeout=1)
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
        await asyncio.wait_for(factory.started.wait(), timeout=1)
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
    app.register_provider(OneShotProvider(), default=True)
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
    assert observed_types[:2] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_STARTED,
    ]
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_run_route_interrupt_during_acceptance_state_read_keeps_task_linked() -> None:
    class BlockingAcceptanceStateStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

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
            await asyncio.wait_for(store.marker_started.wait(), timeout=1)
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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
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
        "accepted_event_sequence": public_event_sequence(queued["id"]),
        "accepted_event_type": EventType.SESSION_MESSAGE_QUEUED,
    }
    assert "/api/sessions/{session_id}/messages" in client.get("/openapi.json").json()["paths"]


def test_enqueue_session_message_endpoint_rejects_nonportable_text() -> None:
    app = CayuApp(enable_logging=False)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

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

    async def recover(request):
        calls.append("recover")
        requests.append(request)
        return IncompleteSessionsRecoveryPage()

    async def recover_event_side_effects(*, limit=1000):
        calls.append("recover_event_side_effects")
        assert limit == 1000
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
    app.recover_persisted_event_side_effects = recover_event_side_effects
    app.drain_background_interruptions = drain_background_interruptions
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
            "recover",
            "resume_cascades",
        ]

    assert calls == [
        "user_start",
        "recover_event_side_effects",
        "recover",
        "resume_cascades",
        "drain",
        "user_stop",
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.statuses == {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }
    assert request.reason == "server_startup_recovery"
    assert request.metadata == {"source": "create_server"}
    assert request.inactive_before is not None
    assert request.inactive_before < datetime.now(UTC)


def test_create_server_startup_recovery_consumes_every_cursor_page() -> None:
    app = CayuApp()
    requests = []
    server_is_ready = threading.Event()
    continuation_started = threading.Event()
    continuation_finished = threading.Event()

    async def recover(request):
        requests.append(request)
        if request.cursor is None:
            return IncompleteSessionsRecoveryPage(next_cursor="startup-page-2")
        if request.cursor == "startup-page-2":
            continuation_started.set()
            for _ in range(100):
                if server_is_ready.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Startup recovery continued before server readiness.")
            return IncompleteSessionsRecoveryPage(next_cursor="startup-page-3")
        assert request.cursor == "startup-page-3"
        continuation_finished.set()
        return IncompleteSessionsRecoveryPage()

    app.recover_incomplete_sessions = recover
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
        assert continuation_started.wait(timeout=2.0)
        assert continuation_finished.wait(timeout=2.0)

    assert [request.cursor for request in requests] == [
        None,
        "startup-page-2",
        "startup-page-3",
    ]
    assert all(request.statuses == {SessionStatus.INTERRUPTED} for request in requests)
    assert all(request.reason == "server_startup_recovery" for request in requests)
    assert all(request.metadata == {"source": "create_server"} for request in requests)
    assert len({request.inactive_before for request in requests}) == 1


def test_create_server_stops_incomplete_recovery_continuation_on_shutdown() -> None:
    app = CayuApp()
    continuation_started = threading.Event()
    continuation_cancelled = threading.Event()

    async def recover(request):
        if request.cursor is None:
            return IncompleteSessionsRecoveryPage(next_cursor="startup-page-2")
        continuation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            continuation_cancelled.set()
            raise

    app.recover_incomplete_sessions = recover
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
            )
        ),
    )

    with TestClient(server):
        assert continuation_started.wait(timeout=2.0)

    assert continuation_cancelled.wait(timeout=2.0)


def test_create_server_background_startup_recovery_rejects_cursor_cycles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = CayuApp()
    cursors = []
    continuation_finished = threading.Event()

    async def recover(request):
        cursors.append(request.cursor)
        if request.cursor is not None:
            continuation_finished.set()
        return IncompleteSessionsRecoveryPage(next_cursor="repeated-startup-cursor")

    app.recover_incomplete_sessions = recover
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


def test_create_server_does_not_swallow_recovery_timeout_error() -> None:
    app = CayuApp()

    async def fail_recovery(*, limit):
        raise TimeoutError("session store timed out")

    app.recover_persisted_event_side_effects = fail_recovery

    with (
        pytest.raises(TimeoutError, match="session store timed out"),
        TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)),
    ):
        pass


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


def _sse_events(client: TestClient, path: str, body: dict) -> list[dict]:
    with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        return [frame["data"] for frame in _sse_frames(response) if "data" in frame]


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

from __future__ import annotations

# ruff: noqa: E402
import asyncio
import json
from collections.abc import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    CayuApp,
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    Message,
    MessageRole,
    TaskCreate,
    TaskStatus,
    TextPart,
    ThinkingPart,
)
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
    client = TestClient(create_server(app))

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
    client = TestClient(create_server(CayuApp(knowledge_store=store)))

    rejected = client.post("/api/knowledge/pending_bad/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "archived"

    pending = client.get("/api/knowledge/pending")
    assert pending.status_code == 200
    assert pending.json()["entries"] == []


def test_server_knowledge_review_reports_missing_store_and_scope_errors() -> None:
    missing_store = TestClient(create_server(CayuApp()))
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
    client = TestClient(create_server(app))

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
    client = TestClient(create_server(CayuApp(knowledge_store=store)))

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
    client = TestClient(create_server(app))

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
    client = TestClient(create_server(app))

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

    client = TestClient(create_server(CayuApp(task_store=task_store)))
    tasks = client.get("/api/tasks").json()

    assert len(tasks) == 1
    assert tasks[0]["id"] == "leased_task"
    assert tasks[0]["type"] == "review"
    assert tasks[0]["title"] is None
    assert tasks[0]["status"] == "running"
    assert tasks[0]["status_reason"] is None
    assert tasks[0]["status_payload"] is None
    assert tasks[0]["session_id"] is None
    assert tasks[0]["worker_id"] == "worker_a"
    assert tasks[0]["completed_at"] is None
    assert isinstance(tasks[0]["lease_expires_at"], str)
    assert isinstance(tasks[0]["created_at"], str)
    assert "parent_task_id" not in tasks[0]
    assert "assigned_agent_name" not in tasks[0]
    assert "description" not in tasks[0]
    assert "input" not in tasks[0]
    assert "result" not in tasks[0]
    assert "error" not in tasks[0]
    assert "metadata" not in tasks[0]
    assert "updated_at" not in tasks[0]


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

    client = TestClient(create_server(CayuApp(task_store=task_store)))
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

    client = TestClient(create_server(CayuApp(task_store=task_store)))
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

    client = TestClient(create_server(CayuApp(task_store=task_store)))

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

    client = TestClient(create_server(CayuApp(task_store=task_store)))

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
    missing_store_client = TestClient(create_server(CayuApp()))

    missing_store_response = missing_store_client.post("/api/tasks/task_1/block")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store)))

    missing_task_response = client.post("/api/tasks/missing_task/block")
    assert missing_task_response.status_code == 404
    assert "missing_task" in missing_task_response.json()["detail"]


def test_server_task_lifecycle_endpoints_validate_request_body() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="task_1", type="review"))

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store)))
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

    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["status"] == "interrupted"


def test_server_run_defaults_and_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    captured: list[int] = []
    original_run = app.run

    def spy_run(request: RunRequest):
        captured.append(request.max_steps)
        return original_run(request)

    app.run = spy_run  # type: ignore[method-assign]
    client = TestClient(create_server(app))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    with client.stream("POST", "/api/run", json={"prompt": "hello", "max_steps": 7}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured == [20, 7]


def test_server_resume_overrides_max_steps() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

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
    client = TestClient(create_server(app))

    body: dict = {"prompt": "hello", "max_steps": bad_value}
    if path == "/api/resume":
        body["session_id"] = "session-does-not-matter"
    response = client.post(path, json=body)
    assert response.status_code == 422


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
    assert [session["id"] for session in response.json()["sessions"]] == ["sess_builder_invoice"]


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
    client = TestClient(create_server(app))

    response = client.get("/api/sessions?label=cayu:agent=builder")

    assert response.status_code == 200
    assert response.json()["sessions"] == []


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
                "requested_model": "fake-model",
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


def _lifecycle_store_and_client(seed) -> tuple[InMemorySessionStore, TestClient]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))
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
    assert client.get("/api/sessions/sess_lab").json()["session"]["labels"] == {"stage": "review"}


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
    assert client.get("/api/sessions/sess_meta").json()["session"]["metadata"] == {"b": [1, 2]}


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
    assert detail["session"]["metadata"] == {"secret": "value"}


def test_transcript_pagination_terminates_when_excluding_thinking() -> None:
    # Regression: with include_thinking=false the store drops thinking-only records from a
    # page, so the route must advance next_offset by the window size (not the returned
    # record count) or pagination stalls on an empty page (reviewer's repro: thinking-only
    # first record + limit=1).
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app))

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


def test_run_stream_carries_resumable_event_ids_and_replays_on_last_event_id() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    assert frames
    session_id = frames[0]["data"]["session_id"]
    # Every frame carries a resumable id of the form `<session_id>:<event_id>`.
    for frame in frames:
        assert frame["id"] == f"{session_id}:{frame['data']['id']}"

    # A reconnect with Last-Event-ID replays the persisted events the client missed
    # instead of starting a new run.
    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello"},
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        assert response.status_code == 200
        replayed = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in replayed] == [
        frame["data"]["id"] for frame in frames[1:]
    ]
    assert replayed[-1]["data"]["type"] == "session.completed"
    # No new session was created by the replay request.
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [session["id"] for session in sessions] == [session_id]


def test_run_replay_rejects_malformed_last_event_id_and_unknown_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    malformed = client.post(
        "/api/run",
        json={"prompt": "hello"},
        headers={"Last-Event-ID": "not-a-marker"},
    )
    assert malformed.status_code == 422

    unknown = client.post(
        "/api/run",
        json={"prompt": "hello"},
        headers={"Last-Event-ID": "missing_session:event_1"},
    )
    assert unknown.status_code == 404


def test_client_disconnect_does_not_cancel_detached_run() -> None:
    import time

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

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
        status = client.get(f"/api/sessions/{session_id}").json()["session"]["status"]
        if status == "completed":
            break
        time.sleep(0.05)
    assert status == "completed"


def test_run_stream_failure_emits_terminal_structured_error_frame() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app))

    async def broken_run(request):
        raise RuntimeError("run exploded before streaming")
        yield  # pragma: no cover - makes this an async generator

    app.run = broken_run

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames
    error_frame = frames[-1]
    assert error_frame.get("event") == "error"
    assert error_frame["data"] == {
        "type": "stream.error",
        "error": "run exploded before streaming",
        "error_type": "RuntimeError",
    }

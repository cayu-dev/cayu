from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    REDACTED_SECRET,
    CayuApp,
    InMemoryTaskStore,
    Message,
    SecretRedactor,
    TaskCreate,
)
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime.tasks import TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH
from cayu.server import ServerConfig, create_server


def _client(
    session_store: InMemorySessionStore,
    task_store: InMemoryTaskStore | None,
    *,
    secret_redactor: SecretRedactor | None = None,
) -> TestClient:
    return TestClient(
        create_server(
            CayuApp(
                session_store=session_store,
                task_store=task_store,
                secret_redactor=secret_redactor,
                enable_logging=False,
            ),
            config=ServerConfig.local_development(),
        )
    )


async def _session(
    store: InMemorySessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
) -> None:
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[Message.text("user", "topology")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


def test_session_topology_projects_bounded_task_links_and_typed_edges() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def seed() -> None:
        await _session(sessions, "root")
        await _session(sessions, "focus", parent_session_id="root")
        await _session(sessions, "child", parent_session_id="focus")
        await tasks.create_task(
            TaskCreate(
                task_id="task-parent",
                type="workflow",
                title="Coordinator",
                session_id="focus",
                assigned_agent_name="assistant",
                input={"customer_secret": "not topology"},
                metadata={"internal": True},
            )
        )
        await tasks.create_task(
            TaskCreate(
                task_id="task-child-a",
                type="step",
                title="First",
                session_id="focus",
                parent_task_id="task-parent",
            )
        )
        await tasks.create_task(
            TaskCreate(
                task_id="task-child-b",
                type="step",
                title="Second",
                session_id="child",
                parent_task_id="task-parent",
            )
        )

    asyncio.run(seed())
    client = _client(sessions, tasks)

    response = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["focus"],
            "child_limit": 10,
            "linked_task_session_ids": ["focus", "child"],
            "expanded_task_parent_ids": ["task-parent"],
            "task_session_limit": 1,
            "task_child_limit": 1,
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["cross_store_atomic"] is False
    projection = body["task_projection"]
    assert projection["status"] == "available"
    assert projection["observed_at"] is not None
    assert [branch["session_id"] for branch in projection["session_branches"]] == [
        "focus",
        "child",
    ]
    focus_branch = projection["session_branches"][0]
    assert [task["id"] for task in focus_branch["tasks"]] == ["task-parent"]
    assert focus_branch["has_more"] is True
    assert isinstance(focus_branch["next_cursor"], str)
    assert [task["id"] for task in projection["expanded_parents"]] == ["task-parent"]
    task_branch = projection["child_branches"][0]
    assert [task["id"] for task in task_branch["children"]] == ["task-child-a"]
    assert task_branch["has_more"] is True
    assert isinstance(task_branch["next_cursor"], str)

    edges = {
        (edge["kind"], edge["source_id"], edge["target_id"], edge["target_loaded"])
        for edge in body["edges"]
    }
    assert ("session_parent", "focus", "root", True) in edges
    assert ("session_parent", "child", "focus", True) in edges
    assert ("task_session", "task-parent", "focus", True) in edges
    assert ("task_parent", "task-child-a", "task-parent", True) in edges
    assert ("task_session", "task-child-a", "focus", True) in edges

    rendered = json.dumps(projection, sort_keys=True)
    assert "customer_secret" not in rendered
    assert '"input"' not in rendered
    assert '"metadata"' not in rendered
    assert '"result"' not in rendered
    assert '"error"' not in rendered
    assert '"worker_id"' not in rendered
    assert '"lease_expires_at"' not in rendered
    assert '"status_payload"' not in rendered

    continuation = client.post(
        "/api/sessions/focus/topology",
        json={
            "expanded_parent_ids": ["focus"],
            "child_limit": 10,
            # The omitted session selection continues to mean the focus session,
            # including when its scope-bound cursor is present.
            "task_session_cursors": {
                "focus": focus_branch["next_cursor"],
            },
            "expanded_task_parent_ids": ["task-parent"],
            "task_child_cursors": {
                "task-parent": task_branch["next_cursor"],
            },
            "task_session_limit": 1,
            "task_child_limit": 1,
        },
    )
    assert continuation.status_code == 200
    continued = continuation.json()["task_projection"]
    assert [task["id"] for task in continued["session_branches"][0]["tasks"]] == ["task-child-a"]
    assert [task["id"] for task in continued["child_branches"][0]["children"]] == ["task-child-b"]


def test_session_topology_reports_optional_task_projection_states() -> None:
    sessions = InMemorySessionStore()
    asyncio.run(_session(sessions, "focus"))

    missing = _client(sessions, None).post(
        "/api/sessions/focus/topology",
        json={},
    )
    assert missing.status_code == 200
    assert missing.json()["task_projection"] == {
        "status": "not_configured",
        "observed_at": None,
        "session_branches": [],
        "expanded_parents": [],
        "child_branches": [],
        "unique_node_count": 0,
    }

    class UnsupportedTaskTopologyStore(InMemoryTaskStore):
        supports_task_topology = False

        async def query_task_topology(self, query):
            raise AssertionError("Unsupported task topology must not be called.")

    unsupported = _client(sessions, UnsupportedTaskTopologyStore()).post(
        "/api/sessions/focus/topology",
        json={},
    )
    assert unsupported.status_code == 200
    assert unsupported.json()["task_projection"]["status"] == "unsupported"
    assert unsupported.json()["focus"]["id"] == "focus"

    class InconsistentTaskTopologyStore(InMemoryTaskStore):
        async def query_task_topology(self, query):
            return await super().query_task_topology(
                query.model_copy(
                    update={
                        "linked_session_ids": ("different-session",),
                    }
                )
            )

    inconsistent = _client(sessions, InconsistentTaskTopologyStore()).post(
        "/api/sessions/focus/topology",
        json={},
    )
    assert inconsistent.status_code == 409
    assert inconsistent.headers["cache-control"] == "private, no-store"
    assert inconsistent.json()["detail"] == (
        "The task store returned an inconsistent topology projection."
    )


def test_task_topology_requires_linked_sessions_to_be_loaded() -> None:
    sessions = InMemorySessionStore()
    asyncio.run(_session(sessions, "focus"))
    client = _client(sessions, InMemoryTaskStore())

    response = client.post(
        "/api/sessions/focus/topology",
        json={"linked_task_session_ids": ["unloaded-session"]},
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["detail"].startswith(
        "linked_task_session_ids may contain only sessions loaded"
    )


def test_task_topology_sanitizes_invalid_task_inputs_and_reports_missing_parents() -> None:
    secret = "task-topology-request-secret"
    sessions = InMemorySessionStore()
    asyncio.run(_session(sessions, "focus"))
    client = _client(
        sessions,
        InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
    )

    invalid = client.post(
        "/api/sessions/focus/topology",
        json={
            "linked_task_session_ids": [
                f"session-{secret}",
                *(f"session-{index}" for index in range(50)),
            ]
        },
    )
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid session topology request."}
    assert invalid.headers["cache-control"] == "private, no-store"
    assert secret not in invalid.text
    assert REDACTED_SECRET not in invalid.text

    missing = client.post(
        "/api/sessions/focus/topology",
        json={"expanded_task_parent_ids": ["missing-task"]},
    )
    assert missing.status_code == 404
    assert missing.headers["cache-control"] == "private, no-store"
    assert missing.json()["detail"] == "A requested expanded task parent was not found."


def test_task_topology_rejects_cycles_and_structural_secret_redaction() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def seed() -> None:
        await _session(sessions, "focus")
        await tasks.create_task(
            TaskCreate(
                task_id="task-cycle-a",
                type="step",
                session_id="focus",
                parent_task_id="task-cycle-b",
            )
        )
        await tasks.create_task(
            TaskCreate(
                task_id="task-cycle-b",
                type="step",
                session_id="focus",
                parent_task_id="task-cycle-a",
            )
        )

    asyncio.run(seed())

    cycle = _client(sessions, tasks).post(
        "/api/sessions/focus/topology",
        json={"task_session_limit": 1},
    )
    assert cycle.status_code == 409
    assert cycle.headers["cache-control"] == "private, no-store"
    assert cycle.json()["detail"] == ("The loaded durable task topology contains a cycle.")

    safe_tasks = InMemoryTaskStore()

    async def seed_safe() -> None:
        await safe_tasks.create_task(
            TaskCreate(
                task_id="task-secret",
                type="step",
                title="secret",
                session_id="focus",
            )
        )

    asyncio.run(seed_safe())
    redactor = SecretRedactor("secret")
    redacted = _client(
        sessions,
        safe_tasks,
        secret_redactor=redactor,
    ).post("/api/sessions/focus/topology", json={})

    assert redacted.status_code == 409
    assert redacted.headers["cache-control"] == "private, no-store"
    assert redacted.json()["detail"] == (
        "Task topology identity cannot cross the configured redaction boundary."
    )
    assert "task-secret" not in redacted.text
    assert REDACTED_SECRET not in redacted.text


def test_task_topology_reports_bounded_ancestry_exhaustion() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def seed() -> None:
        await _session(sessions, "focus")
        parent_id: str | None = None
        for index in range(TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH + 2):
            task_id = f"task-depth-{index:03d}"
            await tasks.create_task(
                TaskCreate(
                    task_id=task_id,
                    type="step",
                    session_id=("focus" if index == TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH + 1 else None),
                    parent_task_id=parent_id,
                )
            )
            parent_id = task_id

    asyncio.run(seed())
    response = _client(sessions, tasks).post(
        "/api/sessions/focus/topology",
        json={},
    )

    assert response.status_code == 413
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["detail"] == (
        "Task topology ancestry exceeds the server's bounded validation limits."
    )


def test_task_topology_redacts_display_text_without_changing_identity() -> None:
    secret = "abcdefgh"
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def seed() -> None:
        await _session(sessions, "focus")
        await tasks.create_task(
            TaskCreate(
                task_id="safe-task-id",
                type="step",
                title=f"Customer value {secret}",
                session_id="focus",
            )
        )
        await tasks.create_task(
            TaskCreate(
                task_id="redaction-expansion",
                type="step",
                title=secret * 500,
                session_id="focus",
            )
        )

    asyncio.run(seed())
    response = _client(
        sessions,
        tasks,
        secret_redactor=SecretRedactor(secret),
    ).post("/api/sessions/focus/topology", json={})

    assert response.status_code == 200
    projected = {
        task["id"]: task
        for task in response.json()["task_projection"]["session_branches"][0]["tasks"]
    }
    task = projected["safe-task-id"]
    assert task["id"] == "safe-task-id"
    assert task["title"] == f"Customer value {REDACTED_SECRET}"
    expanded = projected["redaction-expansion"]
    assert expanded["title"] is None
    assert "title" in expanded["truncated_fields"]
    assert secret not in response.text

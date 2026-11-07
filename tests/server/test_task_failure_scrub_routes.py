"""`POST /api/tasks/{task_id}/scrub-error` and `POST /api/tasks/scrub-errors`."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import AuthenticatedAccess, ServerConfig, create_server
from cayu.server.auth import AuthContext
from cayu.server.proxy_identity import ControlPlaneRoles
from cayu.tasks import InMemoryTaskStore, TaskCreate

_TOKEN = "gh" + "s_" + "Q7" * 18


def _store_with_failures(*task_ids: str, error: dict | None = None) -> InMemoryTaskStore:
    store = InMemoryTaskStore()

    async def fill() -> None:
        for task_id in task_ids:
            await store.create_task(TaskCreate(task_id=task_id, type="scout_run"))
            await store.fail_task(
                task_id, error or {"code": "TimeoutExpired", "message": f"push with {_TOKEN}"}
            )

    asyncio.run(fill())
    return store


def _client(store, **app_options) -> TestClient:
    app = CayuApp(task_store=store, enable_logging=False, **app_options)
    return TestClient(create_server(app, config=ServerConfig.local_development()))


def test_scrub_error_masks_one_task_and_reports_counts_only(capsys):
    store = _store_with_failures("task-a")
    with _client(store) as client:
        preview = client.post("/api/tasks/task-a/scrub-error", json={"dry_run": True})
        assert preview.status_code == 200, preview.text
        assert preview.json() == {
            "task_id": "task-a",
            "applied": False,
            "dry_run": True,
            "fields": ["error"],
            "masked_values": 1,
            "receipts": 0,
            "skipped_reason": None,
        }
        assert _TOKEN in client.get("/api/tasks/task-a").text
        assert capsys.readouterr().err == ""

        scrubbed = client.post("/api/tasks/task-a/scrub-error")
        assert scrubbed.status_code == 200, scrubbed.text
        assert scrubbed.json()["applied"] is True
        detail = client.get("/api/tasks/task-a").json()
        assert _TOKEN not in json.dumps(detail)
        assert detail["error"]["message"] == "push with [REDACTED_SECRET]"
        assert detail["error"]["cayu_scrub"]["scrubs"][0]["masked_values"] == 1

        assert client.post("/api/tasks/task-missing/scrub-error").status_code == 404
    audit = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    assert [(line["event"], line["task_id"], line["fields"]) for line in audit] == [
        ("cayu.task.failure_scrubbed", "task-a", ["error"])
    ]
    assert _TOKEN not in json.dumps(audit)


def test_scrub_error_masks_registered_secrets_first():
    from cayu.vaults import SecretRedactor

    secret = "registered-" + "value-1234"
    store = _store_with_failures("task-a", error={"message": f"failed with {secret}"})
    with _client(store, secret_redactor=SecretRedactor([secret])) as client:
        assert client.post("/api/tasks/task-a/scrub-error").json()["applied"] is True
        assert secret not in client.get("/api/tasks/task-a").text


def test_scrub_errors_sweeps_pages_and_reports_only_changes():
    store = _store_with_failures("task-c", "task-a", "task-b")
    asyncio.run(store.create_task(TaskCreate(task_id="task-clean", type="scout_run")))
    asyncio.run(store.fail_task("task-clean", {"message": "nothing secret"}))
    with _client(store) as client:
        first = client.post("/api/tasks/scrub-errors", json={"limit": 2, "dry_run": True})
        assert first.status_code == 200, first.text
        page = first.json()
        assert page["examined"] == 2 and page["next_after_task_id"] == "task-b"
        assert [row["task_id"] for row in page["tasks"]] == ["task-a", "task-b"]
        assert all(row["applied"] is False for row in page["tasks"])

        rows = []
        after = None
        while True:
            body = {"limit": 2} if after is None else {"limit": 2, "after_task_id": after}
            page = client.post("/api/tasks/scrub-errors", json=body).json()
            rows.extend(page["tasks"])
            after = page["next_after_task_id"]
            if after is None:
                break
        assert sorted(row["task_id"] for row in rows) == ["task-a", "task-b", "task-c"]
        assert all(row["applied"] for row in rows)
        listed = client.get("/api/tasks").text
        assert _TOKEN not in listed
        for task_id in ("task-a", "task-b", "task-c"):
            assert _TOKEN not in client.get(f"/api/tasks/{task_id}").text
        again = client.post("/api/tasks/scrub-errors").json()
        assert again["tasks"] == [] and again["examined"] == 4


def test_scrub_routes_answer_501_for_a_store_without_support():
    class Unsupported(InMemoryTaskStore):
        async def scrub_task_failure_text(self, task_id, scrub):
            raise NotImplementedError

        async def list_task_ids_with_failure_text(self, *, after_task_id=None, limit=100):
            raise NotImplementedError

    with _client(Unsupported()) as client:
        assert client.post("/api/tasks/task-a/scrub-error").status_code == 501
        assert client.post("/api/tasks/scrub-errors").status_code == 501


def test_scrub_routes_need_full_access():
    roles = ControlPlaneRoles()
    roles.add_api_path("/api")

    class Auth:
        control_plane_roles = roles

        def __call__(self, request):
            role = request.headers.get("x-test-role")
            if role is None:
                raise HTTPException(401, "Authentication required")
            context = AuthContext(subject=f"{role}-person", claims={"role": role})
            roles.require(request, context)
            return context

    app = CayuApp(task_store=_store_with_failures("task-a"), enable_logging=False)
    config = ServerConfig(access=AuthenticatedAccess(dependency=Auth()))
    with TestClient(create_server(app, config=config)) as client:
        for path in ("/api/tasks/task-a/scrub-error", "/api/tasks/scrub-errors"):
            assert client.post(path).status_code == 401
            assert client.post(path, headers={"x-test-role": "member"}).status_code == 403
        developer = client.post(
            "/api/tasks/task-a/scrub-error", headers={"x-test-role": "developer"}
        )
    assert developer.status_code == 200
    assert developer.json()["applied"] is True

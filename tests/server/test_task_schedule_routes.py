from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp, InMemoryTaskStore, TaskCreate, TaskSchedulePolicy, TaskStatus
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.server.config import AuthenticatedAccess
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.vaults.redaction import SecretRedactor


@pytest.fixture(params=["memory", "sqlite"])
def schedule_store(request, tmp_path):
    store = (
        SQLiteTaskStore(tmp_path / "http-scheduling.sqlite")
        if request.param == "sqlite"
        else InMemoryTaskStore()
    )
    try:
        yield store
    finally:
        if isinstance(store, SQLiteTaskStore):
            asyncio.run(store.close())


def test_authenticated_schedule_edit_cancel_and_public_projection(schedule_store):
    store = schedule_store
    app = CayuApp(
        task_store=store, enable_logging=False, secret_redactor=SecretRedactor("fire_once")
    )
    due = datetime.now(UTC) + timedelta(hours=1)
    asyncio.run(
        app.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
    )
    config = ServerConfig(
        access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="test-password")
        )
    )
    with TestClient(create_server(app, config=config)) as client:
        body = {
            "task_id": "followup",
            "operation_id": "move",
            "expected_revision": 1,
            "available_at": (due + timedelta(hours=1)).isoformat(),
            "policy": {"misfire_policy": "fire_once"},
        }
        assert client.post("/api/tasks/schedule/reschedule", json=body).status_code == 401
        client.auth = ("operator", "test-password")
        moved = client.post("/api/tasks/schedule/reschedule", json=body)
        assert moved.status_code == 200, moved.text
        receipt = moved.json()
        assert receipt["schedule"]["revision"] == 2
        assert receipt["schedule"]["policy"]["misfire_policy"] == "fire_once"
        assert "sha256" not in moved.text
        detail = client.get("/api/tasks/followup")
        listed = client.get("/api/tasks")
        assert detail.json()["schedule"] == receipt["schedule"]
        assert listed.json()[0]["schedule"] == receipt["schedule"]
        cancel = {"task_id": "followup", "operation_id": "cancel", "expected_revision": 1}
        assert client.post("/api/tasks/schedule/cancel", json=cancel).status_code == 409
        cancel["expected_revision"] = 2
        cancelled = client.post("/api/tasks/schedule/cancel", json=cancel)
        assert cancelled.status_code == 200, cancelled.text
        assert client.post("/api/tasks/schedule/cancel", json=cancel).json() == cancelled.json()
        assert client.post("/api/tasks/schedule/reschedule", json=body).json() == receipt
        assert client.get("/api/tasks/followup").json()["status"] == "cancelled"
        history = client.get("/api/tasks/followup/schedule/events", params={"limit": 2})
        assert history.status_code == 200, history.text
        assert [event["sequence"] for event in history.json()] == [1, 2]
        assert history.json()[0]["policy"]["misfire_policy"] == "fire_once"
        tail = client.get("/api/tasks/followup/schedule/events", params={"after_sequence": 2})
        assert tail.status_code == 200, tail.text
        assert [event["type"] for event in tail.json()] == ["task.schedule_cancelled"]
        assert "sha256" not in history.text + tail.text
    task = asyncio.run(store.load_task("followup"))
    assert task is not None and task.status is TaskStatus.CANCELLED


@pytest.mark.parametrize("invalid", [True, -1, 9007199254740992])
def test_schedule_http_rejects_invalid_revision_without_mutation(invalid):
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(
            "/api/tasks/schedule/cancel",
            json={
                "task_id": "followup",
                "operation_id": "cancel",
                "expected_revision": invalid,
            },
        )
        assert response.status_code == 422
    assert asyncio.run(store.list_tasks()) == []


@pytest.mark.parametrize("bad_field", ["available_at", "operation_id", "policy"])
def test_schedule_http_validation_is_secret_safe(bad_field, capsys, caplog, recwarn):
    canary = "private-http-scheduling-canary"
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False, secret_redactor=SecretRedactor(canary))
    body = {
        "task_id": "followup",
        "operation_id": "move",
        "expected_revision": 1,
        "available_at": "2026-09-20T12:00:00+00:00",
        "policy": {},
    }
    body[bad_field] = {"private": canary}
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post("/api/tasks/schedule/reschedule", json=body)
        assert response.status_code == 422
        assert canary not in response.text
    assert asyncio.run(store.list_tasks()) == []
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn

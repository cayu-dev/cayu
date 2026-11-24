"""`cayu task cancel --label` and `POST /api/tasks/cancellations`: cancel what filters match."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.fixture
def agent(monkeypatch):
    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    store = InMemoryTaskStore()

    async def seed() -> None:
        for task_id, labels in (
            ("ana-1", {"person": "ana"}),
            ("ana-2", {"person": "ana"}),
            ("ben-1", {"person": "ben"}),
        ):
            await store.create_task(TaskCreate(task_id=task_id, type="review", labels=labels))

    asyncio.run(seed())
    app = CayuApp(task_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        yield store, client


def _statuses(store: InMemoryTaskStore) -> dict[str, TaskStatus]:
    async def read():
        return {task.id: task.status for task in await store.list_tasks()}

    return asyncio.run(read())


def test_cancel_by_label_cancels_only_the_matching_tasks(agent, capsys) -> None:
    store, _client = agent
    code = main(
        ["task", "cancel", "--label", "person=ana", "--reason", "reset", "--json", "--wait", "5"]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    assert sorted(out["result"]["settled"]) == ["ana-1", "ana-2"]
    assert out["result"]["stopping"] == [] and out["result"]["has_more"] is False
    assert _statuses(store) == {
        "ana-1": TaskStatus.CANCELLED,
        "ana-2": TaskStatus.CANCELLED,
        "ben-1": TaskStatus.PENDING,
    }
    cancelled = asyncio.run(store.load_task("ana-1"))
    assert cancelled is not None and cancelled.error is not None
    assert cancelled.error["message"] == "reset"


def test_the_route_refuses_a_request_without_filters(agent) -> None:
    store, client = agent
    response = client.post("/api/tasks/cancellations", json={})
    assert response.status_code == 422
    assert all(status == TaskStatus.PENDING for status in _statuses(store).values())
    capped = client.post("/api/tasks/cancellations", json={"type": "review", "max_tasks": 2}).json()
    assert len(capped["settled"]) == 2 and capped["has_more"] is True


@pytest.mark.parametrize(
    "arguments", [[], ["--label", "person"], ["--label", "person=ana", "--wait", "61"]]
)
def test_the_command_refuses_bad_filters_before_sending(arguments, monkeypatch, capsys) -> None:
    from cayu.cli import _control_plane_actions

    def refuse(_args):
        raise AssertionError("nothing is sent")

    monkeypatch.setattr(_control_plane_actions, "open_action_api", refuse)
    assert main(["task", "cancel", *arguments, "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_the_request_timeout_outlasts_the_wait(monkeypatch, capsys) -> None:
    from cayu.cli import _control_plane_actions

    seen: list[float] = []

    def capture(args):
        seen.append(args.timeout_seconds)
        raise ValueError("stop here")

    monkeypatch.setattr(_control_plane_actions, "open_action_api", capture)
    main(["task", "cancel", "--label", "person=ana", "--wait", "60", "--json"])
    capsys.readouterr()
    assert seen == [90]

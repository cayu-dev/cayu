"""`cayu task list --label` and `GET /api/tasks?label=`: tasks by the labels they were created with."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApiError
from cayu.cli._task_readers import ApiTaskReader, StoreTaskReader, TaskListRequest
from cayu.storage import SQLiteTaskStore
from cayu.storage import migrations as schema
from cayu.tasks.creation import TaskCreate


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            for task_id, labels in (
                ("ana-close", {"person": "ana", "job": "close"}),
                ("ana-other", {"person": "ana"}),
                ("ben-close", {"person": "ben", "job": "close"}),
                ("plain", {}),
            ):
                await store.create_task(
                    TaskCreate(
                        task_id=task_id, type="review", labels=labels, metadata={"run": "r1"}
                    )
                )
        finally:
            await store.close()

    asyncio.run(seed())
    return path


def _list(capsys: pytest.CaptureFixture[str], database: Path, *arguments: str) -> list[str]:
    code = main(["task", "list", "--sqlite", str(database), "--json", *arguments])
    out = capsys.readouterr().out
    assert code == 0, out
    return sorted(task["id"] for task in json.loads(out)["tasks"])


def test_label_filters_select_tasks(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _list(capsys, database, "--label", "person=ana") == ["ana-close", "ana-other"]
    assert _list(capsys, database, "--label", "person=ana", "--label", "job=close") == ["ana-close"]
    assert _list(capsys, database, "--label", "job=close", "--input", "missing=1") == []
    # With an input or metadata filter the read takes the index's predicate path.
    assert _list(capsys, database, "--label", "person=ana", "--run", "r1") == [
        "ana-close",
        "ana-other",
    ]


@pytest.mark.parametrize(
    "arguments", [("--label", "person"), ("--label", "person=ana", "--label", "person=ben")]
)
def test_malformed_or_repeated_labels_are_refused(
    database: Path, capsys: pytest.CaptureFixture[str], arguments: tuple[str, ...]
) -> None:
    assert main(["task", "list", "--sqlite", str(database), *arguments]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_store_and_server_readers_select_the_same_tasks(database: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.server import create_server

    requests = [
        TaskListRequest(labels={"person": "ana"}),
        TaskListRequest(labels={"job": "close"}, limit=1, offset=1),
        TaskListRequest(labels={"job": "close"}, types=("review",)),
        TaskListRequest(labels={"person": "ben"}, metadata={"run": "r1"}),
    ]

    async def read() -> list[tuple[list[str], list[str]]]:
        local = StoreTaskReader(
            SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE, read_only=True),
            source={},
        )
        store = SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
                served = ApiTaskReader(CayuApi("http://testserver/", http=client), source={})
                return [
                    (
                        [row["id"] for row in await local.list_tasks(request)],
                        [row["id"] for row in await served.list_tasks(request)],
                    )
                    for request in requests
                ]
        finally:
            await local.close()
            await store.close()

    results = asyncio.run(read())

    assert [sorted(local) for local, _served in results] == [
        ["ana-close", "ana-other"],
        [results[1][0][0]],
        ["ana-close", "ben-close"],
        ["ben-close"],
    ]
    assert all(local == served for local, served in results)


class _OldServer:
    """A Cayu API whose server predates label filters: it would ignore them."""

    description = "the old server"

    def __init__(self) -> None:
        self.paths: list[str] = []

    def get(self, path: str, *, query: Any = None, resource: str) -> Any:
        self.paths.append(path)
        if path == "/contract":
            return {"capabilities": {"surfaces": {"tasks": {"read": {"enabled": True}}}}}
        return []


def test_a_server_without_label_filters_is_refused_before_it_lists() -> None:
    api = _OldServer()
    reader = ApiTaskReader(api, source={})  # type: ignore[arg-type]

    with pytest.raises(CayuApiError, match="can't filter tasks by label"):
        asyncio.run(reader.list_tasks(TaskListRequest(labels={"person": "ana"})))
    assert api.paths == ["/contract"]


def test_a_cloud_agent_reader_sends_label_filters_with_the_login() -> None:
    from cayu.storage.agent_readers import _CloudTaskReader

    class Reader:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[TaskListRequest] = []

        async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
            self.calls.append(request)
            return []

    through_cloud, login = Reader("cloud"), Reader("login")
    reader = _CloudTaskReader(through_cloud, login, source={})  # type: ignore[arg-type]
    asyncio.run(reader.list_tasks(TaskListRequest(labels={"person": "ana"})))
    asyncio.run(reader.list_tasks(TaskListRequest()))
    # The agent-read route doesn't forward the contract the label check reads.
    assert [len(login.calls), len(through_cloud.calls)] == [1, 1]
    assert login.calls[0].labels == {"person": "ana"}

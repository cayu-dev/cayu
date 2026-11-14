"""`cayu task list --input/--metadata/--run`: the filters `watchTasks` and `GET /api/tasks` take."""

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


def _tasks() -> list[TaskCreate]:
    return [
        TaskCreate(
            task_id="pass-cayu",
            type="shepherd_pass",
            input={"repository": "cayu-tech/cayu", "number": 217, "dry_run": False},
            metadata={"run": "round-1"},
        ),
        TaskCreate(
            task_id="pass-design",
            type="shepherd_pass",
            input={"repository": "cayu-tech/cayu-design", "number": 3},
            metadata={"run": "round-1"},
        ),
        TaskCreate(
            task_id="pass-later",
            type="shepherd_pass",
            input={"repository": "cayu-tech/cayu", "number": 218},
            metadata={"run": "round-2"},
        ),
        # A value under a sensitive key is hidden in the detail, so it selects nothing.
        TaskCreate(task_id="keyed", type="digest", input={"api_token": "token-canary"}),
    ]


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            for task in _tasks():
                await store.create_task(task)
        finally:
            await store.close()

    asyncio.run(seed())
    return path


def _list(capsys: pytest.CaptureFixture[str], database: Path, *arguments: str) -> list[str]:
    code = main(["task", "list", "--sqlite", str(database), "--json", *arguments])
    out = capsys.readouterr().out
    assert code == 0, out
    return sorted(task["id"] for task in json.loads(out)["tasks"])


def test_input_metadata_and_run_select_tasks_by_value(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _list(capsys, database, "--input", "repository=cayu-tech/cayu") == [
        "pass-cayu",
        "pass-later",
    ]
    assert _list(capsys, database, "--run", "round-1") == ["pass-cayu", "pass-design"]
    assert _list(capsys, database, "--metadata", "run=round-2") == ["pass-later"]
    # Every filter must match; numbers match by value and booleans as true or false.
    assert _list(
        capsys, database, "--input", "repository=cayu-tech/cayu", "--input", "number=217.0"
    ) == ["pass-cayu"]
    assert _list(capsys, database, "--input", "dry_run=false") == ["pass-cayu"]
    assert _list(capsys, database, "--input", "api_token=token-canary") == []


def test_value_filters_page_through_the_matching_tasks(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _list(capsys, database, "--input", "repository=cayu-tech/cayu", "--limit", "1")
    second = _list(
        capsys,
        database,
        *("--input", "repository=cayu-tech/cayu", "--limit", "1", "--offset", "1"),
    )

    assert sorted(first + second) == ["pass-cayu", "pass-later"]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--run", "round-1", "--metadata", "run=round-2"),
        ("--input", "a=1", "--input", "a=2"),
        tuple(item for index in range(9) for item in ("--input", f"key{index}=value")),
    ],
)
def test_conflicting_or_too_many_filters_are_refused(
    database: Path, capsys: pytest.CaptureFixture[str], arguments: tuple[str, ...]
) -> None:
    code = main(["task", "list", "--sqlite", str(database), *arguments])

    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_a_filter_without_a_value_separator_is_a_usage_error(database: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["task", "list", "--sqlite", str(database), "--input", "repository"])
    assert raised.value.code == 2


def test_a_request_refuses_a_key_the_server_would_split() -> None:
    with pytest.raises(ValueError, match='must not contain "="'):
        TaskListRequest(input={"a=b": "c"})


def test_store_and_server_readers_select_the_same_tasks(database: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.server import create_server

    requests = [
        TaskListRequest(input={"repository": "cayu-tech/cayu"}),
        TaskListRequest(metadata={"run": "round-1"}, types=("shepherd_pass",)),
        TaskListRequest(input={"number": "3"}, metadata={"run": "round-1"}),
        TaskListRequest(input={"repository": "cayu-tech/cayu"}, limit=1, offset=1),
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
        ["pass-cayu", "pass-later"],
        ["pass-cayu", "pass-design"],
        ["pass-design"],
        # The second of the two, in the store's order.
        [results[0][0][1]],
    ]
    assert all(local == served for local, served in results)


class _OldServer:
    """A Cayu API whose server predates the value filters: it would ignore them."""

    description = "the old server"

    def __init__(self) -> None:
        self.paths: list[str] = []

    def get(self, path: str, *, query: Any = None, resource: str) -> Any:
        self.paths.append(path)
        if path == "/contract":
            return {"capabilities": {"surfaces": {"tasks": {"read": {"enabled": True}}}}}
        return []


def test_a_server_without_value_filters_is_refused_before_it_lists() -> None:
    api = _OldServer()
    reader = ApiTaskReader(api, source={})  # type: ignore[arg-type]

    with pytest.raises(CayuApiError, match="can't filter tasks by input"):
        asyncio.run(reader.list_tasks(TaskListRequest(metadata={"run": "round-1"})))
    assert api.paths == ["/contract"]


def test_a_filtered_local_read_is_one_index_snapshot(database: Path) -> None:
    """Paging the store with offsets could shift a task into the next page and repeat it."""

    class OneSnapshot:
        def __init__(self, store: SQLiteTaskStore) -> None:
            self.store = store
            self.snapshots = 0

        async def list_tasks(self, query: Any) -> Any:
            raise AssertionError("a filtered read pages the store")

        async def query_task_index(self, query: Any, **kwargs: Any) -> Any:
            self.snapshots += 1
            return await self.store.query_task_index(query, **kwargs)

        async def close(self) -> None:
            await self.store.close()

    async def read() -> tuple[list[list[str]], int]:
        store = OneSnapshot(
            SQLiteTaskStore(database, schema_mode=schema.SchemaMode.VALIDATE, read_only=True)
        )
        reader = StoreTaskReader(store, source={})  # type: ignore[arg-type]
        try:
            pages = [
                [
                    row["id"]
                    for row in await reader.list_tasks(
                        TaskListRequest(metadata={"run": "round-1"}, **page)
                    )
                ]
                for page in (
                    {"type": "shepherd_pass"},
                    {"type": "shepherd_pass", "limit": 1},
                    {"type": "shepherd_pass", "limit": 1, "offset": 1},
                    {"type": "digest"},
                )
            ]
            return pages, store.snapshots
        finally:
            await reader.close()

    (both, first, second, digests), snapshots = asyncio.run(read())

    assert sorted(both) == ["pass-cayu", "pass-design"]
    assert first + second == both
    assert digests == []
    assert snapshots == 4

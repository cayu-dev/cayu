"""`cayu task scrub-error`: a running server, a Cloud Agent, and a local store."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate

_TOKEN = "gh" + "s_" + "Q7" * 18
_ERROR = {"code": "TimeoutExpired", "message": f"git fetch timed out with {_TOKEN}"}


async def _fail(store, *task_ids: str) -> None:
    for task_id in task_ids:
        await store.create_task(TaskCreate(task_id=task_id, type="scout_run"))
        await store.fail_task(task_id, dict(_ERROR))


@pytest.fixture
def served(monkeypatch):
    """A running agent's API, reached as ``--server-url`` and as ``--application``."""

    from cayu.cli import _cloud_agent_api
    from cayu.cli._cloud_agent_api import CayuApi

    store = InMemoryTaskStore()
    asyncio.run(_fail(store, "task-c", "task-a", "task-b"))
    app = CayuApp(task_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        original = CayuApi.__init__

        def through_test_client(self, base_url, *args, **kwargs):
            kwargs["http"] = client
            original(self, "http://testserver", *args, **kwargs)

        monkeypatch.setattr(CayuApi, "__init__", through_test_client)
        monkeypatch.setattr(
            _cloud_agent_api,
            "open_cloud_agent_api",
            lambda application, **kwargs: CayuApi("http://cloud.invalid"),
        )
        yield store


def _run(capsys, *args: str) -> dict:
    assert main(["task", "scrub-error", *args, "--json"]) == 0
    captured = capsys.readouterr()
    assert _TOKEN not in captured.out + captured.err
    return json.loads(captured.out)


def test_scrub_one_task_through_the_server(served, capsys):
    preview = _run(capsys, "task-a", "--dry-run", "--server-url", "http://127.0.0.1:8000")
    assert preview["source"]["kind"] == "server"
    assert preview["result"] == {
        "task_id": "task-a",
        "applied": False,
        "dry_run": True,
        "fields": ["error"],
        "masked_values": 1,
        "receipts": 0,
        "skipped_reason": None,
    }
    assert _TOKEN in asyncio.run(served.load_task("task-a")).error["message"]

    assert _run(capsys, "task-a", "--server-url", "http://127.0.0.1:8000")["result"]["applied"]
    assert _TOKEN not in asyncio.run(served.load_task("task-a")).error["message"]
    # Only task-a was scrubbed.
    assert _TOKEN in asyncio.run(served.load_task("task-b")).error["message"]


def test_scrub_all_through_a_cloud_agent_follows_pages(served, capsys, monkeypatch):
    from cayu.cli import _task_failure_scrub

    monkeypatch.setattr(_task_failure_scrub, "MAX_TASK_FAILURE_SCRUB_PAGE", 2)
    payload = _run(capsys, "--all", "--application", "scout")
    assert payload["source"] == {"kind": "cayu_cloud", "application": "scout"}
    assert payload["examined"] == 3 and payload["next_after"] is None
    assert [row["task_id"] for row in payload["tasks"]] == ["task-a", "task-b", "task-c"]
    for task_id in ("task-a", "task-b", "task-c"):
        assert _TOKEN not in asyncio.run(served.load_task(task_id)).error["message"]


def test_scrub_table_output_names_counts_not_text(served, capsys):
    assert main(["task", "scrub-error", "task-b", "--server-url", "http://127.0.0.1:8000"]) == 0
    out = capsys.readouterr().out
    assert "scrubbed" in out and "cayu task show task-b" in out
    assert _TOKEN not in out


def test_scrub_a_local_sqlite_store_directly(tmp_path, capsys):
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        await _fail(store, "task-a", "task-b")
        await store.close()

    asyncio.run(seed())
    preview = _run(capsys, "--all", "--dry-run", "--sqlite", str(path))
    assert preview["source"] == {"kind": "sqlite", "path": str(path)}
    assert [row["applied"] for row in preview["tasks"]] == [False, False]
    applied = _run(capsys, "--all", "--sqlite", str(path))
    assert [row["applied"] for row in applied["tasks"]] == [True, True]

    async def read() -> list:
        store = SQLiteTaskStore(path)
        try:
            return [await store.load_task(name) for name in ("task-a", "task-b")]
        finally:
            await store.close()

    for task in asyncio.run(read()):
        assert _TOKEN not in task.error["message"]
        assert task.error["cayu_scrub"]["scrubs"][0]["masked_values"] == 1


def test_scrub_refuses_a_task_id_with_all_and_after_without_all(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["task", "scrub-error", "task-a", "--all"])
    capsys.readouterr()
    missing = str(tmp_path / "missing.db")
    assert (
        main(["task", "scrub-error", "task-a", "--after", "x", "--sqlite", missing, "--json"]) == 1
    )
    assert "--after applies only with --all" in capsys.readouterr().out

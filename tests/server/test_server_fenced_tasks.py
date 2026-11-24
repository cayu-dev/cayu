"""`GET /api/fenced-tasks` and `POST /api/fenced-tasks/settle`, and `cayu task fenced`."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

import httpx

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import cloud_worker_exits as exits_module
from cayu.tasks.cloud_worker_exits import WorkerExit, WorkerExitRecord, worker_id_for_run
from cayu.tasks.creation import TaskCreate
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus

ARN_OLD = "arn:aws:ecs:region:account:task/agents/old"
ARN_LIVE = "arn:aws:ecs:region:account:task/agents/live"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


async def _fenced_store(path: Path, *, reclaim: bool = True) -> SQLiteTaskStore:
    clock = _Clock()
    store = SQLiteTaskStore(path, ownership_clock=clock)
    expiries = []
    for task_id, worker_id in (
        ("lost", worker_id_for_run("jobs", ARN_OLD)),
        ("live", worker_id_for_run("jobs", ARN_LIVE)),
        ("plain", "jobs"),
    ):
        await store.create_task(TaskCreate(task_id=task_id, type="scan"))
        claimed = await store.claim_task(worker_id, TaskQuery(type="scan"), lease_seconds=30)
        assert claimed is not None and claimed.lease_expires_at is not None
        await store.mark_claimed_task_execution_started(
            task_id, worker_id, claimed.lease_expires_at
        )
        expiries.append(claimed.lease_expires_at)
    clock.value = max(expiries) + timedelta(seconds=1)
    if reclaim:
        await store.reclaim_expired(query=TaskQuery(type="scan"))
    return store


def _stub_record(monkeypatch: pytest.MonkeyPatch, record: WorkerExitRecord | None) -> None:
    async def read(environ=None, *, transport=None):
        if record is None:
            raise exits_module.WorkerExitsUnsupported("CAYU_WORKER_EXITS_URL is not set.")
        return record

    monkeypatch.setattr(exits_module, "read_cloud_worker_exits", read)


def test_routes_list_and_settle_from_cloud_exits(tmp_path, monkeypatch):
    _stub_record(monkeypatch, WorkerExitRecord(exits=(WorkerExit(ARN_OLD, "stopped"),)))

    async def scenario():
        store = await _fenced_store(tmp_path / "tasks.sqlite")
        app = CayuApp(task_store=store, enable_logging=False)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            listed = await client.get("/api/fenced-tasks")
            assert listed.status_code == 200, listed.text
            body = listed.json()
            assert body["worker_exits"] == "read" and body["truncated"] is False
            assert {task["id"]: task["worker_run_stopped"] for task in body["tasks"]} == {
                "lost": True,
                "live": False,
                "plain": None,
            }
            assert {task["id"]: task["settleable"] for task in body["tasks"]} == {
                "lost": True,
                "live": False,
                "plain": False,
            }
            settled = await client.post("/api/fenced-tasks/settle")
            assert settled.status_code == 200, settled.text
            assert [task["id"] for task in settled.json()["settled"]] == ["lost"]
            again = (await client.get("/api/fenced-tasks")).json()
            assert sorted(task["id"] for task in again["tasks"]) == ["live", "plain"]
        await store.close()

    asyncio.run(scenario())


def test_routes_say_when_cloud_records_no_exits(tmp_path, monkeypatch):
    _stub_record(monkeypatch, None)

    async def scenario():
        store = await _fenced_store(tmp_path / "tasks.sqlite")
        app = CayuApp(task_store=store, enable_logging=False)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            body = (await client.get("/api/fenced-tasks")).json()
            assert body["worker_exits"] == "unsupported"
            assert "CAYU_WORKER_EXITS_URL" in body["worker_exits_message"]
            assert {task["worker_run_stopped"] for task in body["tasks"]} == {None}
            settled = (await client.post("/api/fenced-tasks/settle")).json()
            assert settled["settled"] == [] and settled["worker_exits"] == "unsupported"
        await store.close()

    asyncio.run(scenario())


def test_cli_lists_and_settles_a_local_store(tmp_path, monkeypatch, capsys):
    from cayu.cli import main

    _stub_record(monkeypatch, WorkerExitRecord(exits=(WorkerExit(ARN_OLD, "stopped"),)))
    path = tmp_path / "tasks.sqlite"

    async def prepare():
        await (await _fenced_store(path)).close()

    asyncio.run(prepare())
    assert main(["task", "fenced", "--sqlite", str(path), "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {task["id"] for task in listed["tasks"]} == {"lost", "live", "plain"}
    assert {task["id"]: task["worker_run_stopped"] for task in listed["tasks"]}["lost"] is True
    # Settling needs the Agent's own view of Cloud's record.
    assert main(["task", "fenced", "--sqlite", str(path), "--settle"]) == 1
    assert "--application" in capsys.readouterr().err


def test_fenced_warning_names_the_commands():
    from cayu.cli._cloud_task_health import fenced_tasks_warning

    warning = fenced_tasks_warning(
        {"task_health": {"fenced_after_worker_loss": 1, "complete": True}}, "invoices"
    )
    assert warning is not None
    assert "cayu task fenced --application invoices" in warning
    assert "--settle" in warning


def test_settle_records_fences_no_worker_recorded_yet(tmp_path, monkeypatch):
    _stub_record(monkeypatch, WorkerExitRecord(exits=(WorkerExit(ARN_OLD, "stopped"),)))

    async def scenario():
        # The leases expired, but no worker has reclaimed them: no fence is recorded.
        store = await _fenced_store(tmp_path / "tasks.sqlite", reclaim=False)
        app = CayuApp(task_store=store, enable_logging=False)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            settled = await client.post("/api/fenced-tasks/settle")
            assert [task["id"] for task in settled.json()["settled"]] == ["lost"]
        await store.close()

    asyncio.run(scenario())


def test_a_retry_series_task_is_stopped_but_not_settleable():
    from cayu.tasks.cloud_worker_exits import _report_item
    from cayu.tasks.records import Task

    task = Task.model_construct(
        id="series-1",
        type="scan",
        status=TaskStatus.CLAIMED,
        worker_id="jobs",
        lease_expires_at=None,
        session_id=None,
        retry_series="series",
    )
    item = _report_item(task, True)
    assert item["worker_run_stopped"] is True and item["settleable"] is False

"""Task progress reported from a handler and read through the API, the CLI and activity."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._task_lease_authority import TaskLeaseAuthority, bind_task_lease_authority
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.progress import InMemoryTaskProgressStore, report_task_progress
from cayu.tasks.queries import TaskQuery


def _in_task(task_id: str):
    return bind_task_lease_authority(
        TaskLeaseAuthority(
            datetime.now(UTC) + timedelta(minutes=5),
            claim_generation=1,
            task_id=task_id,
            worker_id="worker",
        )
    )


async def _in_handler(task_id: str, work) -> object:
    # A worker runs the handler in an asyncio task created under the task's binding.
    with _in_task(task_id):
        running = asyncio.ensure_future(work())
    return await running


def _app(**stores) -> CayuApp:
    return CayuApp(enable_logging=False, **stores)


def test_a_handler_reports_progress_for_its_own_task() -> None:
    async def scenario() -> None:
        progress = InMemoryTaskProgressStore()
        app = _app(task_store=InMemoryTaskStore(), task_progress_store=progress)
        try:

            async def handler() -> None:
                await report_task_progress(app, total=3, done=1, counts={"approved": 1})
                await report_task_progress(app, done=1, counts={"failed": 1})

            await _in_handler("review-1", handler)
            stored = await progress.load_progress("review-1")
            assert stored is not None
            assert (stored.done, stored.total, stored.counts) == (
                2,
                3,
                {"approved": 1, "failed": 1},
            )
            with pytest.raises(RuntimeError, match="inside a task handler"):
                await report_task_progress(app, done=1)
            bare = _app(task_store=InMemoryTaskStore())
            with pytest.raises(RuntimeError, match="task_progress_store"):
                await _in_handler("review-1", lambda: report_task_progress(bare, done=1))
            await bare.aclose()
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_the_app_refuses_something_that_is_not_a_progress_store() -> None:
    with pytest.raises(TypeError, match="task_progress_store"):
        _app(task_progress_store=object())  # type: ignore[arg-type]


def test_the_task_api_serves_progress() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._task_readers import ApiTaskReader
    from cayu.server import ServerConfig, create_server

    async def seed(app: CayuApp) -> None:
        assert app.task_store is not None and app.task_progress_store is not None
        await app.task_store.create_task(TaskCreate(task_id="review-1", type="review"))
        await app.task_store.create_task(TaskCreate(task_id="quiet", type="review"))
        await app.task_progress_store.set_progress("review-1", total=40, done=12)

    app = _app(task_store=InMemoryTaskStore(), task_progress_store=InMemoryTaskProgressStore())
    asyncio.run(seed(app))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        body = client.get("/api/tasks/review-1/progress").json()
        assert (body["done"], body["total"], body["counts"]) == (12, 40, {})
        assert client.get("/api/tasks/quiet/progress").status_code == 404
        assert client.get("/api/tasks/missing/progress").status_code == 404
        reader = ApiTaskReader(CayuApi("http://testserver/", http=client), source={})
        assert asyncio.run(reader.task_progress("quiet")) is None
        assert asyncio.run(reader.task_progress("review-1"))["done"] == 12  # type: ignore[index]


def test_task_show_prints_progress_from_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cayu.cli import main
    from cayu.storage import SQLiteTaskStore
    from cayu.storage.task_progress_sqlite import SQLiteTaskProgressStore

    path = tmp_path / "cayu.db"

    async def seed() -> None:
        tasks, progress = SQLiteTaskStore(path), SQLiteTaskProgressStore(path)
        try:
            await tasks.create_task(TaskCreate(task_id="review-1", type="review"))
            await progress.set_progress("review-1", total=40)
            await progress.add_progress("review-1", done=12, counts={"approved": 10, "failed": 2})
        finally:
            await progress.close()
            await tasks.close()

    asyncio.run(seed())
    assert main(["task", "show", "review-1", "--sqlite", str(path), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)["progress"]
    assert (shown["done"], shown["total"]) == (12, 40)
    assert main(["task", "show", "review-1", "--sqlite", str(path)]) == 0
    assert "Progress: 12 of 40 done (approved 10, failed 2)" in capsys.readouterr().out


def test_activity_names_a_runs_task_and_carries_running_tasks_progress() -> None:
    from cayu.cloud_reporting import read_cloud_activity

    async def scenario() -> None:
        sessions, tasks, progress = (
            InMemorySessionStore(),
            InMemoryTaskStore(),
            InMemoryTaskProgressStore(),
        )
        app = _app(session_store=sessions, task_store=tasks, task_progress_store=progress)
        try:

            async def start(session_id: str) -> None:
                await sessions.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
                await sessions.append_event(
                    session_id, Event(type=EventType.SESSION_STARTED, session_id=session_id)
                )

            await tasks.create_task(TaskCreate(task_id="review-1", type="review"))
            await tasks.create_task(TaskCreate(task_id="queued", type="review"))
            claimed = await tasks.claim_task("worker", TaskQuery(type="review"), lease_seconds=300)
            assert claimed is not None and claimed.id == "review-1"
            await _in_handler("review-1", lambda: start("in-task"))
            await start("alone")
            await progress.set_progress("review-1", total=40, done=12)
            await progress.set_progress("queued", total=1)
            first = await read_cloud_activity(app)
            runs = {run.session_id: run for run in first.runs}
            assert runs[app.project_session_id_for_exposure("in-task")].task_id == "review-1"
            assert runs[app.project_session_id_for_exposure("alone")].task_id is None
            # Only running tasks: the queued one isn't listed.
            assert [(item.task_id, item.done) for item in first.task_progress] == [("review-1", 12)]
            # A progress report alone shows on the next page, with no run changed.
            await progress.add_progress("review-1", done=1)
            later = await read_cloud_activity(app, cursor=first.next_cursor)
            assert later.runs == ()
            assert [(item.task_id, item.done) for item in later.task_progress] == [("review-1", 13)]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_read_only_sqlite_snapshot_is_still_inspected(tmp_path: Path) -> None:
    import sqlite3

    from cayu.cli._task_readers import SQLiteProgressReader
    from cayu.storage.task_progress_sqlite import SQLiteTaskProgressStore

    path = tmp_path / "snapshot.db"

    async def seed() -> None:
        store = SQLiteTaskProgressStore(path)
        await store.set_progress("review-1", done=3)
        await store.close()

    asyncio.run(seed())
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
    path.chmod(0o444)
    try:
        reader = SQLiteProgressReader(path)
        progress = asyncio.run(reader.load_progress("review-1"))
        assert progress is not None and progress.done == 3
        empty = tmp_path / "empty.db"
        sqlite3.connect(empty).close()
        assert asyncio.run(SQLiteProgressReader(empty).load_progress("review-1")) is None
    finally:
        path.chmod(0o644)


def test_a_server_without_the_progress_route_shows_no_progress() -> None:
    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli._task_readers import ApiTaskReader

    class OldServer:
        description = "the old server"

        def get(self, path, **kwargs):
            raise CayuApiError("unsupported", "no such route")

    reader = ApiTaskReader(OldServer(), source={})  # type: ignore[arg-type]
    assert asyncio.run(reader.task_progress("review-1")) is None


def test_a_cloud_agent_reader_reads_progress_with_the_login() -> None:
    from cayu.storage.agent_readers import _CloudTaskReader

    class Reader:
        def __init__(self, progress) -> None:
            self.progress, self.calls = progress, 0

        async def task_progress(self, task_id: str):
            self.calls += 1
            return self.progress

    through_cloud, login = Reader(None), Reader({"done": 12})
    reader = _CloudTaskReader(through_cloud, login, source={})  # type: ignore[arg-type]
    assert asyncio.run(reader.task_progress("review-1")) == {"done": 12}
    assert (through_cloud.calls, login.calls) == (0, 1)

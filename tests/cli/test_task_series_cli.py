"""`cayu task series` and the series id every occurrence keeps in its metadata."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.storage import SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.tasks import TaskCreate, TaskRecurrence, TaskSeries
from cayu.tasks.queries import TaskQuery
from cayu.tasks.store import TaskStore

START = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)


def _series(series_id: str) -> TaskSeries:
    return TaskSeries(
        series_id=series_id,
        recurrence=TaskRecurrence(every_seconds=3600, start_at=START),
        task=TaskCreate(type="review", labels={"owner": "42"}),
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.save_task_series(_series("review-a"))
            await store.save_task_series(_series("review-b"))
        finally:
            await store.close()

    asyncio.run(seed())
    return path


def _json(capsys: pytest.CaptureFixture[str], *arguments: str, code: int = 0) -> dict:
    assert main(["task", *arguments, "--json"]) == code
    return json.loads(capsys.readouterr().out)


def test_series_commands_list_show_pause_resume_and_delete(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = ("--sqlite", str(database))
    listed = _json(capsys, "series", "list", *db)
    assert [item["series"]["series_id"] for item in listed["series"]] == ["review-a", "review-b"]
    shown = _json(capsys, "series", "show", "review-a", *db)["series"]
    assert shown["paused"] is False and shown["next_task_id"].startswith("review-a@")

    assert _json(capsys, "series", "pause", "review-a", *db)["series"]["paused"] is True
    assert _json(capsys, "series", "resume", "review-a", *db)["series"]["paused"] is False

    # Each occurrence names its series in its metadata, so `task list --metadata` finds them.
    tasks = _json(capsys, "list", "--metadata", "cayu_series_id=review-a", *db)["tasks"]
    assert [task["id"] for task in tasks] == [shown["next_task_id"]]

    assert _json(capsys, "series", "delete", "review-b", *db)["deleted"] is True
    assert _json(capsys, "series", "delete", "review-b", *db, code=1)["deleted"] is False
    missing = _json(capsys, "series", "show", "review-b", *db, code=1)
    assert "No task series review-b" in missing["error"]["message"]

    assert main(["task", "series", "list", *db]) == 0
    table = capsys.readouterr().out
    assert "review-a" in table and "every 3600s" in table


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_occurrences_name_their_series_and_series_list_pages(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        store: TaskStore
        if backend == "memory":
            from cayu.tasks.memory import InMemoryTaskStore

            store = InMemoryTaskStore()
        elif backend == "sqlite":
            store = SQLiteTaskStore(tmp_path / "tasks.db")
        else:
            from cayu.storage import PostgresTaskStore

            store = PostgresTaskStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.MIGRATE
            )
        prefix = f"label-{backend}"
        for index in range(3):
            await store.save_task_series(_series(f"{prefix}-{index}"))
        page = await store.list_task_series(limit=2, offset=1)
        assert [state.series.series_id for state in page][:2] == [f"{prefix}-1", f"{prefix}-2"]
        tasks = [
            task
            for task in await store.list_tasks(TaskQuery(type="review"))
            if task.metadata.get("cayu_series_id") == f"{prefix}-0"
        ]
        assert len(tasks) == 1
        # The series id is not a label, so the labels the task runs as are the template's.
        assert tasks[0].invocation.access_labels == {"owner": "42"}
        with pytest.raises(ValueError):
            await store.list_task_series(limit=0)
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_the_series_id_metadata_replaces_a_template_value() -> None:
    from cayu.tasks.series import occurrence_request

    series = TaskSeries(
        series_id="x",
        recurrence=TaskRecurrence(every_seconds=3600, start_at=START),
        task=TaskCreate(type="review", metadata={"cayu_series_id": "other", "keep": "1"}),
    )
    request = occurrence_request(series, START)
    assert request.metadata["cayu_series_id"] == "x" and request.metadata["keep"] == "1"


def test_series_output_masks_credentials_in_the_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "secret.db"
    secret = "sk-" + "live" * 6

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.save_task_series(
                TaskSeries(
                    series_id="keyed",
                    recurrence=TaskRecurrence(every_seconds=3600, start_at=START),
                    task=TaskCreate(type="review", input={"api_key": secret}),
                )
            )
        finally:
            await store.close()

    asyncio.run(seed())
    db = ("--sqlite", str(path))
    for command in (("list",), ("show", "keyed"), ("pause", "keyed"), ("resume", "keyed")):
        assert main(["task", "series", *command, *db, "--json"]) == 0
        out = capsys.readouterr().out
        assert secret not in out and "[REDACTED]" in out
        assert main(["task", "series", *command, *db]) == 0
        assert secret not in capsys.readouterr().out

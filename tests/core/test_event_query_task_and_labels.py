"""``EventQuery.task_id`` and ``EventQuery.labels``: events of the sessions a task or labels select."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._task_lease_authority import TaskLeaseAuthority, bind_task_lease_authority
from cayu.sessions.event_queries import EventQuery, copy_event_query
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

_SESSIONS = (
    ("ana-review", {"person": "ana", "job": "close"}, "task-1"),
    ("ana-chat", {"person": "ana"}, None),
    ("ben-review", {"person": "ben", "job": "close"}, "task-2"),
)


async def _create(store, session_id: str, labels: dict[str, str]) -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "hello")],
            labels=labels,
        ),
        identity=SessionIdentity(provider_name="provider", model="model"),
    )


async def _check(store) -> None:
    for session_id, labels, task_id in _SESSIONS:
        if task_id is None:
            await _create(store, session_id, labels)
        else:
            # A session created while the task's handler runs is marked as run under it.
            authority = TaskLeaseAuthority(
                datetime.now(UTC) + timedelta(minutes=5),
                claim_generation=1,
                task_id=task_id,
                worker_id="worker",
            )
            with bind_task_lease_authority(authority):
                running = asyncio.ensure_future(_create(store, session_id, labels))
            await running
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"{session_id}-{index}",
                    session_id=session_id,
                    type=EventType.SESSION_STARTED,
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                )
                for index in range(2)
            ],
        )

    async def sessions(**fields) -> list[str]:
        records = await store.query_events(EventQuery(**fields, limit=100))
        return sorted({record.event.session_id for record in records})

    assert await sessions(task_id="task-1") == ["ana-review"]
    assert await sessions(labels={"person": "ana"}) == ["ana-chat", "ana-review"]
    assert await sessions(labels={"job": "close"}) == ["ana-review", "ben-review"]
    assert await sessions(labels={"job": "close"}, task_id="task-2") == ["ben-review"]
    assert await sessions(labels={"person": "ana"}, task_id="task-2") == []
    assert await sessions(task_id="missing") == []
    assert len(await store.query_events(EventQuery(task_id="task-1", limit=100))) == 2
    assert await store.event_exists(EventQuery(labels={"person": "ben"}))
    assert not await store.event_exists(EventQuery(labels={"person": "nobody"}))


def test_memory_store_reads_events_by_task_and_labels() -> None:
    asyncio.run(_check(InMemorySessionStore()))


def test_sqlite_store_reads_events_by_task_and_labels(tmp_path: Path) -> None:
    from cayu.storage.sqlite import SQLiteSessionStore

    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            await _check(store)
        finally:
            await store.close()

    asyncio.run(scenario())


def test_postgres_store_reads_events_by_task_and_labels(postgres_test_dsn: str) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def scenario() -> None:
        store = PostgresSessionStore(postgres_test_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _check(store)
        finally:
            await store.close()

    asyncio.run(scenario())


def test_task_and_label_filters_are_validated_and_copied() -> None:
    query = EventQuery(task_id="task-1", labels={"person": "ana"})
    assert copy_event_query(query) == query
    dumped = EventQuery().model_dump()
    assert "task_id" not in dumped and "labels" not in dumped
    with pytest.raises(ValidationError):
        EventQuery(task_id=" ")
    with pytest.raises(ValidationError):
        EventQuery(labels={"person": 1})  # type: ignore[dict-item]


def test_cli_pages_the_events_of_a_task_or_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from cayu.cli import main
    from cayu.storage.sqlite import SQLiteSessionStore

    path = tmp_path / "sessions.sqlite"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            await _check(store)
        finally:
            await store.close()

    asyncio.run(seed())

    def sessions(*arguments: str) -> list[str]:
        code = main(["session", "events", "--sqlite", str(path), "--json", *arguments])
        out = capsys.readouterr().out
        assert code == 0, out
        return sorted({row["session_id"] for row in json.loads(out)["events"]})

    assert sessions("--task", "task-2") == ["ben-review"]
    assert sessions("--label", "person=ana") == ["ana-chat", "ana-review"]
    assert sessions("--label", "job=close", "--task", "task-1") == ["ana-review"]
    assert main(["session", "events", "--sqlite", str(path)]) == 1
    captured = capsys.readouterr()
    assert "Give a session id" in captured.out + captured.err


def test_remote_readers_refuse_task_and_label_filters() -> None:
    from cayu.cli._agent_session_reader import AgentSessionReader

    reader = AgentSessionReader(object())  # type: ignore[arg-type]
    for query in (
        EventQuery(session_id="s1", task_id="task-1"),
        EventQuery(session_id="s1", labels={"person": "ana"}),
    ):
        with pytest.raises(ValueError, match="local session store"):
            asyncio.run(reader.query_events(query))

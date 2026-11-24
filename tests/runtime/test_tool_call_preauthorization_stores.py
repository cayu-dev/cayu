"""One contract suite for in-memory, SQLite, and PostgreSQL preauthorization stores."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu.approvals import (
    InMemoryToolCallPreauthorizationStore,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantNotFound,
    ToolCallPreauthorization,
    ToolCallPreauthorizationUse,
)
from cayu.approvals.authorization_grants import tool_arguments_digest
from cayu.storage import SQLiteToolCallPreauthorizationStore
from cayu.storage import migrations as schema

pytestmark = pytest.mark.anyio

T0 = datetime(2026, 10, 1, 12, 0, 0, 123000, tzinfo=UTC)
OPERATOR = ToolAuthorizationGrantActor(subject="operator", tenant="acme")
ARGUMENTS = {"email_id": "e-1", "to": ["ap@example.com"]}


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def open_store(request, tmp_path):
    """Return a factory; each call opens a new store instance on one database."""

    backend = request.param
    address = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "preauthorizations.sqlite")
    )
    memory = InMemoryToolCallPreauthorizationStore()
    opened = []

    def factory():
        if backend == "memory":
            return memory
        if backend == "sqlite":
            store = SQLiteToolCallPreauthorizationStore(address)
        else:
            from cayu.storage import PostgresToolCallPreauthorizationStore

            store = PostgresToolCallPreauthorizationStore(
                address, schema_mode=schema.SchemaMode.CREATE
            )
        opened.append(store)
        return store

    yield factory
    for store in opened:
        await store.close()


def record(task_id: str, *, suffix: str = "1", created_at: datetime = T0, **overrides):
    return ToolCallPreauthorization.for_call(
        preauthorization_id=f"{task_id}-{suffix}",
        task_id=task_id,
        tool_name="send_email",
        arguments=overrides.pop("arguments", ARGUMENTS),
        authorized_by=OPERATOR,
        created_at=created_at,
        **overrides,
    )


def call(task_id: str, call_id: str, at: datetime = T0) -> dict:
    return {
        "task_id": task_id,
        "tool_name": "send_email",
        "arguments_digest": tool_arguments_digest(ARGUMENTS),
        "agent_name": "mailer",
        "use": ToolCallPreauthorizationUse(
            session_id="s",
            session_instance_id="i",
            tool_round_id="r",
            tool_call_id=call_id,
            used_at=at + timedelta(seconds=1),
        ),
    }


def fresh_task() -> str:
    # PostgreSQL databases are shared per module, so every test binds a fresh task.
    return f"task-{uuid4().hex}"


async def test_issue_is_idempotent_and_durable(open_store) -> None:
    task = fresh_task()
    store = open_store()
    first = await store.issue(record(task))
    assert await store.issue(record(task)) == first
    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.issue(record(task, arguments={"to": ["other@example.com"]}))
    with pytest.raises(ValueError, match="unused and unrevoked"):
        await store.issue(record(task, suffix="2").model_copy(update={"revoked_at": T0}))
    reopened = open_store()
    assert await reopened.get(first.preauthorization_id) == first
    with pytest.raises(ToolAuthorizationGrantNotFound):
        await reopened.get("missing")


async def test_list_for_task_is_ordered_and_scoped(open_store) -> None:
    task = fresh_task()
    store = open_store()
    later = await store.issue(record(task, suffix="b", created_at=T0 + timedelta(seconds=5)))
    earlier = await store.issue(record(task, suffix="a"))
    await store.issue(record(fresh_task()))
    assert await store.list_for_task(task) == (earlier, later)


async def test_use_is_once_and_a_retry_of_the_same_call_repeats(open_store) -> None:
    task = fresh_task()
    store = open_store()
    await store.issue(record(task))
    used = await store.use(**call(task, "call_1"))
    assert used is not None and used.use is not None
    assert await store.use(**call(task, "call_1")) == used
    assert await open_store().use(**call(task, "call_2")) is None
    assert (await open_store().get(used.preauthorization_id)).use == used.use


async def test_concurrent_calls_use_a_record_once(open_store) -> None:
    task = fresh_task()
    await open_store().issue(record(task))
    stores = [open_store() for _ in range(4)]
    results = await asyncio.gather(
        *(store.use(**call(task, f"call_{index}")) for index, store in enumerate(stores))
    )
    assert sum(result is not None for result in results) == 1


async def test_revoke_and_expiry(open_store) -> None:
    task = fresh_task()
    store = open_store()
    await store.issue(record(task, suffix="revoked"))
    revoked = await store.revoke(f"{task}-revoked", revoked_by=OPERATOR, now=T0)
    assert revoked.revoked_by == OPERATOR
    assert await store.revoke(f"{task}-revoked", revoked_by=None, now=T0) == revoked
    assert await store.use(**call(task, "call_1")) is None
    with pytest.raises(ToolAuthorizationGrantNotFound):
        await store.revoke("missing", revoked_by=None, now=T0)

    await store.issue(record(task, suffix="expiring", lifetime_seconds=1))
    assert await store.use(**call(task, "call_2", at=T0 + timedelta(seconds=5))) is None

    await store.issue(record(task, suffix="used"))
    assert await store.use(**call(task, "call_3")) is not None
    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.revoke(f"{task}-used", revoked_by=None, now=T0)


async def test_concurrent_issues_of_one_id_for_two_tasks_conflict(open_store) -> None:
    first, second = fresh_task(), fresh_task()
    issued = record(first, suffix="shared")
    other = issued.model_copy(update={"task_id": second})
    results = await asyncio.gather(
        open_store().issue(issued), open_store().issue(other), return_exceptions=True
    )
    assert sum(isinstance(result, ToolAuthorizationGrantConflict) for result in results) == 1
    kept = next(result for result in results if isinstance(result, ToolCallPreauthorization))
    store = open_store()
    assert await store.get(kept.preauthorization_id) == kept
    assert [r.preauthorization_id for r in await store.list_for_task(kept.task_id)] == [
        kept.preauthorization_id
    ]


async def test_concurrent_identical_issues_return_one_record(open_store) -> None:
    task = fresh_task()
    results = await asyncio.gather(
        *(
            open_store().issue(record(task, created_at=T0 + timedelta(milliseconds=index)))
            for index in range(4)
        )
    )
    assert len({result.model_dump_json() for result in results}) == 1

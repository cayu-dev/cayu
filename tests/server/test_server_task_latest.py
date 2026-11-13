"""`GET /api/tasks/latest` and several task types on `GET /api/tasks`."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import asyncio

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.server import create_server
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery


def _client() -> TestClient:
    store = InMemoryTaskStore()

    async def seed() -> None:
        for index, repository in enumerate(("o/a", "o/b", "o/a")):
            await store.create_task(
                TaskCreate(task_id=f"pass-{index}", type="pass", input={"repository": repository})
            )
            claimed = await store.claim_task("worker", TaskQuery(type="pass"))
            assert claimed is not None
            await store.complete_task(
                claimed.id,
                {"pass": index},
                worker_id="worker",
                claim_generation=claimed.claim_generation,
            )
        await store.create_task(TaskCreate(task_id="recheck-1", type="recheck"))
        await store.create_task(TaskCreate(task_id="other-1", type="other"))

    asyncio.run(seed())
    return TestClient(create_server(CayuApp(task_store=store), config=_LOCAL_SERVER_CONFIG))


def test_latest_returns_the_newest_finished_tasks_with_their_results() -> None:
    client = _client()

    latest = client.get("/api/tasks/latest", params={"type": "pass", "status": "completed"})
    assert latest.status_code == 200
    assert latest.headers["Cache-Control"] == "private, no-store"
    assert [(task["id"], task["result"]) for task in latest.json()] == [
        ("pass-2", {"pass": 2}),
        ("pass-1", {"pass": 1}),
        ("pass-0", {"pass": 0}),
    ]

    grouped = client.get("/api/tasks/latest", params={"type": "pass", "by": "input.repository"})
    assert [task["id"] for task in grouped.json()] == ["pass-2", "pass-1"]

    one = client.get("/api/tasks/latest", params={"types": ["pass", "recheck"], "limit": 1})
    assert [task["id"] for task in one.json()] == ["pass-2"]

    assert client.get("/api/tasks/latest", params={"by": "result.x"}).status_code == 422
    assert client.get("/api/tasks/latest", params={"type": "a", "types": ["b"]}).status_code == 422
    # A task named `latest` is still readable by id elsewhere; this route answers first.
    assert client.get("/api/tasks/pass-1").json()["id"] == "pass-1"


def test_task_list_filters_by_several_types() -> None:
    client = _client()

    assert client.get("/api/tasks", params={"type": "a", "types": ["b"]}).status_code == 422
    listed = client.get("/api/tasks", params={"types": ["recheck", "other"]})
    assert listed.status_code == 200
    assert sorted(task["id"] for task in listed.json()) == ["other-1", "recheck-1"]

    counted = client.get("/api/tasks", params={"types": ["recheck", "other"], "include": "counts"})
    assert counted.status_code == 200
    assert sorted(task["id"] for task in counted.json()["items"]) == ["other-1", "recheck-1"]


def test_latest_selectors_match_the_public_projection_not_hidden_values() -> None:
    from cayu.vaults.redaction import SecretRedactor

    secret = "acct-secret-9f3a"
    store = InMemoryTaskStore()

    async def seed() -> None:
        await store.create_task(TaskCreate(task_id="t1", type="pass", assigned_agent_name=secret))
        claimed = await store.claim_task("worker", TaskQuery(type="pass"))
        assert claimed is not None
        await store.complete_task(
            claimed.id, {}, worker_id="worker", claim_generation=claimed.claim_generation
        )

    asyncio.run(seed())
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secret))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    # Guessing the hidden name confirms nothing, as on `GET /api/tasks`.
    for guess in (secret, "acct-secret-wrong"):
        assert client.get("/api/tasks/latest", params={"assigned_agent_name": guess}).json() == []
    assert [task["id"] for task in client.get("/api/tasks/latest").json()] == ["t1"]


def test_latest_by_input_key_forms_no_group_from_a_hidden_value() -> None:
    from cayu.vaults.redaction import SecretRedactor

    secrets = ("tok-secret-1a2b", "tok-secret-3c4d", "tok-secret-5e6f")
    tokens = (secrets[0], "plain-a", secrets[1], secrets[0], "plain-b", secrets[2])
    store = InMemoryTaskStore()

    async def seed() -> None:
        for index, token in enumerate(tokens):
            await store.create_task(
                TaskCreate(task_id=f"t{index}", type="pass", input={"token": token})
            )
            claimed = await store.claim_task("worker", TaskQuery(type="pass"))
            assert claimed is not None
            await store.complete_task(
                claimed.id, {}, worker_id="worker", claim_generation=claimed.claim_generation
            )

    asyncio.run(seed())
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secrets))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    def grouped(**paging: int) -> list[str]:
        params = {"by": "input.token", **paging}
        return [task["id"] for task in client.get("/api/tasks/latest", params=params).json()]

    # One row per hidden value would count the hidden values and pair the tasks that
    # share one. Only the keys the response shows form groups, and paging is over those.
    assert grouped() == ["t4", "t1"]
    assert grouped(limit=1) == ["t4"]
    assert grouped(limit=1, offset=1) == ["t1"]
    assert grouped(limit=1, offset=2) == []
    # Without grouping every task is still listed, its hidden value masked.
    assert len(client.get("/api/tasks/latest").json()) == len(tokens)

    # One request reads the store a bounded number of times: a page past the keys
    # those reads reach is refused, and a full page needs no further read.
    from cayu.server import _task_latest_routes as route

    reads = []
    read_store = store.latest_tasks

    async def counted(query: TaskQuery, by: str | None = None) -> list:
        reads.append(query.offset)
        return await read_store(query, by=by)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(route, "_GROUP_READ_SIZE", 1)
        patch.setattr(route, "_MAX_GROUP_READS", 2)
        patch.setattr(store, "latest_tasks", counted)
        assert grouped(limit=1) == ["t4"]
        assert reads == [0, 1]
        reads.clear()
        refused = client.get("/api/tasks/latest", params={"by": "input.token", "offset": 1})
        assert refused.status_code == 422 and "at most" in refused.json()["detail"]
        assert reads == [0, 1]

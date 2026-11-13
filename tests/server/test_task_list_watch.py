"""`GET /api/tasks` filters by a task's input and metadata values and answers unchanged lists with 304."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskCreate
from cayu.vaults import SecretRedactor


def _passes():
    return [
        TaskCreate(
            task_id="pass-cayu",
            type="shepherd_pass",
            input={
                "repository": "cayu-tech/cayu",
                "number": 217,
                "dry_run": False,
                "threshold": 0.00001,
                "ratio": 2.0,
            },
            metadata={"run": "round-1"},
        ),
        TaskCreate(
            task_id="pass-design",
            type="shepherd_pass",
            input={"repository": "cayu-tech/cayu-design"},
            metadata={"run": "round-1"},
        ),
        TaskCreate(
            task_id="pass-secret",
            type="shepherd_pass",
            input={"repository": "private-repository-canary"},
            metadata={"run": "round-2"},
        ),
        TaskCreate(task_id="digest", type="digest", input={"repository": "cayu-tech/cayu"}),
    ]


@pytest.fixture(params=["memory", "sqlite"])
def client(request, tmp_path):
    if request.param == "sqlite":
        from cayu.storage import SQLiteTaskStore

        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    else:
        store = InMemoryTaskStore()

    async def seed():
        for task in _passes():
            await store.create_task(task)

    asyncio.run(seed())
    app = CayuApp(
        task_store=store,
        secret_redactor=SecretRedactor("private-repository-canary"),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as test_client:
        test_client.store = store
        yield test_client
    if request.param == "sqlite":
        asyncio.run(store.close())


def _ids(response) -> set[str]:
    assert response.status_code == 200, response.text
    body = response.json()
    items = body["items"] if isinstance(body, dict) else body
    return {item["id"] for item in items}


def test_input_and_metadata_filters_match_top_level_values(client):
    get = client.get
    assert _ids(get("/api/tasks", params={"input": "repository=cayu-tech/cayu"})) == {
        "pass-cayu",
        "digest",
    }
    assert _ids(
        get("/api/tasks", params={"type": "shepherd_pass", "input": "repository=cayu-tech/cayu"})
    ) == {"pass-cayu"}
    # Numbers and booleans match their JSON text; several filters must all match.
    assert _ids(
        get("/api/tasks", params=[("input", "number=217"), ("input", "dry_run=false")])
    ) == {"pass-cayu"}
    assert _ids(get("/api/tasks", params={"input": "number=218"})) == set()
    # Numbers compare by value, however a client spells them.
    for spelling in ("threshold=0.00001", "threshold=1e-5", "ratio=2", "ratio=2.0", "number=217.0"):
        assert _ids(get("/api/tasks", params={"input": spelling})) == {"pass-cayu"}, spelling
    assert _ids(get("/api/tasks", params={"input": "ratio=two"})) == set()
    assert _ids(get("/api/tasks", params={"metadata": "run=round-1"})) == {
        "pass-cayu",
        "pass-design",
    }
    assert _ids(
        get(
            "/api/tasks",
            params={"metadata": "run=round-1", "statuses": ["pending"], "include": "counts"},
        )
    ) == {"pass-cayu", "pass-design"}
    # A value with "=" in it keeps everything after the first "=".
    assert _ids(get("/api/tasks", params={"input": "repository=a=b"})) == set()


def test_filters_match_keys_as_the_detail_shows_them(client):
    asyncio.run(
        client.store.create_task(
            TaskCreate(
                task_id="secret-key",
                type="shepherd_pass",
                input={"private-repository-canary": "repo"},
            )
        )
    )
    shown = client.get("/api/tasks/secret-key").json()["input"]
    (key,) = shown
    assert key != "private-repository-canary"
    assert _ids(client.get("/api/tasks", params={"input": f"{key}=repo"})) == {"secret-key"}
    assert (
        _ids(client.get("/api/tasks", params={"input": "private-repository-canary=repo"})) == set()
    )


def test_filters_match_the_redacted_value_the_detail_shows(client):
    shown = client.get("/api/tasks/pass-secret").json()["input"]["repository"]
    assert shown != "private-repository-canary"
    assert (
        _ids(client.get("/api/tasks", params={"input": "repository=private-repository-canary"}))
        == set()
    )
    assert _ids(client.get("/api/tasks", params={"input": f"repository={shown}"})) == {
        "pass-secret"
    }


@pytest.mark.parametrize("value", ["repository", "=cayu", " repository=x"])
def test_a_filter_without_a_key_is_refused(client, value):
    assert client.get("/api/tasks", params={"input": value}).status_code == 422
    assert client.get("/api/tasks", params={"metadata": value}).status_code == 422


def test_too_many_filters_are_refused(client):
    params = [("input", f"k{index}=v") for index in range(9)]
    assert client.get("/api/tasks", params=params).status_code == 422


def test_a_cursor_stays_bound_to_its_value_filters(client):
    params = {"include": "counts", "limit": 1, "metadata": "run=round-1"}
    first = client.get("/api/tasks", params=params).json()
    assert first["next_cursor"]
    other = client.get(
        "/api/tasks", params={**params, "metadata": "run=round-2", "cursor": first["next_cursor"]}
    )
    assert other.status_code >= 400
    same = client.get("/api/tasks", params={**params, "cursor": first["next_cursor"]})
    assert same.status_code == 200


def test_an_unchanged_list_answers_304(client):
    params = {"type": "shepherd_pass", "metadata": "run=round-1"}
    first = client.get("/api/tasks", params=params)
    etag = first.headers["ETag"]
    assert etag.startswith('W/"')
    again = client.get("/api/tasks", params=params, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["ETag"] == etag

    asyncio.run(
        client.store.create_task(
            TaskCreate(task_id="pass-new", type="shepherd_pass", metadata={"run": "round-1"})
        )
    )
    changed = client.get("/api/tasks", params=params, headers={"If-None-Match": etag})
    assert changed.status_code == 200
    assert changed.headers["ETag"] != etag
    assert "pass-new" in _ids(changed)
    # The unfiltered list has its own tag.
    assert client.get("/api/tasks").headers["ETag"] not in {etag, changed.headers["ETag"]}

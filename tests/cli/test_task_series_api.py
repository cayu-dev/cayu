"""`cayu task series` through a running server and a Cloud Agent, and the `/task-series` routes."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskRecurrence, TaskSeries

START = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=5)
SERVER = ("--server-url", "http://127.0.0.1:8000")


SECRET = "sk-" + "series" * 4


def _series(series_id: str, **task) -> TaskSeries:
    return TaskSeries(
        series_id=series_id,
        recurrence=TaskRecurrence(every_seconds=3600, start_at=START),
        task=TaskCreate(type="review", **task),
    )


@pytest.fixture
def served(monkeypatch):
    from cayu.cli import _cloud_agent_api
    from cayu.cli._cloud_agent_api import CayuApi

    store = InMemoryTaskStore()

    async def seed() -> None:
        await store.save_task_series(_series("review-a", input={"token": SECRET}))
        await store.save_task_series(_series("review-b"))
        await store.save_task_series(_series("reports/daily"))

    asyncio.run(seed())
    from cayu.vaults.redaction import SecretRedactor

    app = CayuApp(task_store=store, enable_logging=False, secret_redactor=SecretRedactor([SECRET]))
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
        yield client


def _json(capsys, *arguments: str, code: int = 0) -> dict:
    assert main(["task", "series", *arguments, "--json"]) == code
    return json.loads(capsys.readouterr().out)


def test_series_commands_through_a_server_and_a_cloud_agent(served, capsys) -> None:
    listed = _json(capsys, "list", *SERVER)
    assert listed["source"]["kind"] == "server"
    assert [item["series"]["series_id"] for item in listed["series"]] == [
        "reports/daily",
        "review-a",
        "review-b",
    ]
    # A series id with a slash still reaches its own routes.
    assert _json(capsys, "pause", "reports/daily", *SERVER)["series"]["paused"] is True
    shown = _json(capsys, "show", "review-a", "--application", "invoices")["series"]
    assert shown["series"]["series_id"] == "review-a" and shown["paused"] is False

    assert _json(capsys, "pause", "review-a", *SERVER)["series"]["paused"] is True
    assert (
        _json(capsys, "resume", "review-a", "--application", "invoices")["series"]["paused"]
        is False
    )
    assert _json(capsys, "delete", "review-b", *SERVER)["deleted"] is True
    assert _json(capsys, "delete", "review-b", *SERVER, code=1)["deleted"] is False
    missing = _json(capsys, "show", "review-b", *SERVER, code=1)
    assert "No task series review-b" in missing["error"]["message"]
    assert "No task series" in _json(capsys, "pause", "nope", *SERVER, code=1)["error"]["message"]


def test_the_routes_answer_as_documented(served) -> None:
    page = served.get("/api/task-series", params={"limit": 1, "offset": 2})
    assert page.status_code == 200
    [item] = page.json()["series"]
    assert (item["series_id"], item["paused"]) == ("review-b", False)
    assert item["state"]["series"]["series_id"] == "review-b"
    # A configured secret in a series' template is redacted as in task details.
    shown = served.get("/api/task-series/review-a")
    assert shown.status_code == 200 and SECRET not in shown.text
    assert served.get("/api/task-series/reports%2Fdaily").json()["series_id"] == "reports/daily"
    assert served.get("/api/task-series/none").status_code == 404
    assert served.post("/api/task-series/none/pause").status_code == 404
    assert served.post("/api/task-series/none/delete").status_code == 404
    assert served.get("/api/task-series", params={"limit": 0}).status_code == 422


class _OlderAgent:
    """An agent whose Cayu serves its API but not the series routes."""

    def get(self, path, **kwargs):
        from cayu.cli._cloud_agent_api import CayuApiError

        if path == "/app":
            return {}
        raise CayuApiError("unsupported", "The agent runs a Cayu that doesn't serve them yet.")

    def close(self) -> None:
        pass


def test_an_agent_without_the_routes_gets_a_clear_error(monkeypatch, capsys) -> None:
    from cayu.cli import _task_series

    monkeypatch.setattr(
        _task_series,
        "_open_api",
        lambda target: (_task_series._ApiSeriesStore(_OlderAgent()), {"kind": "cayu_cloud"}),
    )
    message = _json(capsys, "list", "--application", "invoices", code=1)["error"]["message"]
    assert "doesn't serve them yet" in message and "--metadata cayu_series_id=ID" in message


def test_an_older_cayu_answering_404_reads_as_unsupported(capsys) -> None:
    """A real server without the routes: the CLI finds its API first, then reports the gap."""

    from fastapi import FastAPI

    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._task_series import _ApiSeriesStore

    older = FastAPI()

    @older.get("/api/app")
    def app_info():
        return {"name": "older"}

    with TestClient(older) as client:
        store = _ApiSeriesStore(CayuApi("http://testserver", http=client))
        with pytest.raises(ValueError, match="--metadata cayu_series_id=ID"):
            asyncio.run(store.list_task_series(limit=10, offset=0))


def test_a_server_without_a_task_store_says_so() -> None:
    with TestClient(
        create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    ) as client:
        assert client.get("/api/task-series/x").status_code == 501

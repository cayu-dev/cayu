"""`cayu session show` reports the model call a session's run is making.

A worker that died inside a model call leaves the call's stage active. `show` prints
it by identity and state, locally and from a Cloud Agent, so an operator can tell a run
stopped inside a model call (and pass its stage and run epoch to
`resolve-provider-operation`) without reading the store.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.runtime.test_orphaned_session_reconciliation import _orphan

from cayu import CayuApp
from cayu.cli import main
from cayu.runtime._model_stage_inspection import _stage_inspection, inspect_model_stage
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.sessions._model_stage import ModelCompletionStage
from cayu.storage import SQLiteSessionStore

_STAGE_FIELDS = {
    "stage_id",
    "purpose",
    "state",
    "dispatched",
    "background",
    "model_step_id",
    "model_attempt_id",
    "dispatch_ordinal",
    "provider_name",
    "requested_model",
    "source_run_epoch",
    "reservation_count",
    "prepared_at",
    "completed_at",
}


def _show(capsys, *argv: str) -> dict:
    assert main(["session", "show", "scan", *argv, "--json"]) == 0, capsys.readouterr()
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def agent(monkeypatch):
    """The store served as a Cloud Agent, and a CLI signed in to Cayu Cloud."""

    def serve(database: Path):
        monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
        server = create_server(
            CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False),
            config=ServerConfig.protected(
                BasicAuth(username="operator", password="s3cret-password")
            ),
        )
        client = TestClient(server, base_url="http://127.0.0.1")
        client.__enter__()
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        return client

    clients = []
    yield lambda database: clients.append(serve(database)) or clients[-1]
    for client in clients:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("crash", ["model", "prepared"])
def test_show_reports_the_model_call_a_lost_run_stopped_inside(
    tmp_path, capsys, agent, crash
) -> None:
    database = tmp_path / "cayu.db"
    # The worker reserved budget and died inside the provider's stream, or after it
    # prepared the call and before it dispatched it.
    _orphan(database, "scan", crash=crash, database_url=f"sqlite:///{database}")

    local = _show(capsys, "--sqlite", str(database))

    stage = local["model_stage"]
    assert set(stage) == _STAGE_FIELDS
    assert stage["purpose"] == "assistant-turn"
    assert stage["state"] == "in_flight"
    assert stage["dispatched"] is (crash == "model")
    assert stage["background"] is False
    assert stage["provider_name"] == "fake"
    assert stage["requested_model"] == "fake-model"
    assert stage["reservation_count"] >= 1
    assert stage["stage_id"] == f"{stage['model_step_id']}:dispatch:{stage['dispatch_ordinal']}"
    assert stage["model_attempt_id"]
    assert stage["completed_at"] is None
    # What `resolve-provider-operation --run-epoch` takes.
    assert type(local["session"]["run_epoch"]) is int
    assert stage["source_run_epoch"] == local["session"]["run_epoch"]
    assert local["provider_operation"]["status"] == "synchronous"
    # Identity and state only: nothing the model was sent.
    assert '"go"' not in json.dumps(local)

    # A Cloud Agent's session state reports the same call; the model step and attempt
    # ids stay private, as its call listings keep them.
    agent(database)
    remote = _show(capsys, "--application", "bug-hunter-agent")
    assert remote["model_stage"] == {**stage, "model_step_id": None, "model_attempt_id": None}
    assert remote["session"]["run_epoch"] == local["session"]["run_epoch"]
    assert remote["provider_operation"] == local["provider_operation"]

    assert main(["session", "show", "scan", "--sqlite", str(database), "--table"]) == 0
    rows = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1].strip()
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("model_stage.")
    }
    assert rows["model_stage.stage_id"] == stage["stage_id"]
    assert rows["model_stage.dispatched"] == str(crash == "model")


def test_show_reports_no_model_stage_when_the_run_isnt_in_a_model_call(tmp_path, capsys) -> None:
    database = tmp_path / "cayu.db"
    # The worker died inside a tool call, after the model answered.
    _orphan(database, "scan", crash="tool")

    payload = _show(capsys, "--sqlite", str(database))

    assert payload["model_stage"] is None
    assert payload["session"]["status"] == "running"


def test_the_state_api_reports_the_active_model_stage(tmp_path) -> None:
    database = tmp_path / "cayu.db"
    _orphan(database, "scan", crash="model")

    async def inspected():
        store = SQLiteSessionStore(database)
        try:
            return await inspect_model_stage(store, "scan")
        finally:
            await store.close()

    stage = asyncio.run(inspected())
    assert stage is not None and stage.dispatched
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/sessions/scan/state")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_stage"] == {
        **stage.model_dump(mode="json"),
        "model_step_id": None,
        "model_attempt_id": None,
    }
    assert body["execution"]["run_epoch"] == stage.source_run_epoch


def test_resolve_provider_operation_help_names_where_show_prints_its_inputs(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["session", "resolve-provider-operation", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "model_stage.stage_id" in help_text
    assert "session.run_epoch" in help_text
    assert "provider_operation.allowed_resolutions" in help_text


def test_an_auxiliary_call_reports_the_model_step_its_events_name() -> None:
    """An auxiliary stage's logical step is its publication identity, not its model step."""

    class Store:
        async def load_model_completion_stage_dispatch(self, session_id, stage_id):
            return None

    stage = ModelCompletionStage.model_construct(
        session_id="scan",
        stage_id="auxiliary:abc",
        logical_step_id="auxiliary:abc",
        dispatch_ordinal=0,
        purpose="auxiliary-inference",
        state="in_flight",
        intent={
            "model_step_id": "mstep_0123",
            "model_attempt_id": "matt_0123",
            "provider_name": "fake",
            "requested_model": "fake-model",
            "request": {"api_key": "sk-secret"},
        },
        reservation_ids=(),
        source_run_epoch=2,
        prepared_at=datetime(2026, 11, 7, tzinfo=UTC),
        completed_at=None,
    )

    inspected = asyncio.run(_stage_inspection(Store(), stage))

    assert inspected.model_step_id == "mstep_0123"
    assert inspected.stage_id == "auxiliary:abc"
    assert inspected.purpose == "auxiliary-inference"
    assert "sk-secret" not in inspected.model_dump_json()

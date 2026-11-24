"""`cayu cloud data-move` runs a deployed Agent's declared data moves on the Agent's server."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.cli.serve import _include_data_moves
from cayu.data_move_runs import MemoryDataMoveRunStore
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult
from cayu.server import AuthenticatedAccess, BasicAuth, ServerConfig, create_server
from cayu.server.data_moves import data_move_router

_PASSWORD = "operator-password"

_MODULE = """
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult

FAIL = {"at": None}
SOURCE = ["a", "b", "c", "d", "e"]
TARGET = set()


def _read(cursor, limit):
    if cursor is not None and cursor == FAIL["at"]:
        FAIL["at"] = None
        raise ConnectionError("the old store went away")
    start = int(cursor or 0)
    end = start + limit
    return SourcePage(SOURCE[start:end], str(end) if end < len(SOURCE) else None)


def _apply(item, *, dry_run):
    if item in TARGET:
        return StepResult(StepOutcome.UNCHANGED)
    if not dry_run:
        TARGET.add(item)
    return StepResult(StepOutcome.CREATED)


MOVE = DataMove("letters", _read, str, _apply)
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0"\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n\n'
        '[tool.cayu.data_moves]\nletters = "letters_cloud:MOVE"\n',
        encoding="utf-8",
    )
    (tmp_path / "letters_cloud.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_DATA_VOLUME", str(tmp_path / "volume"))
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    yield tmp_path
    sys.modules.pop("letters_cloud", None)


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A fake Cayu Cloud whose Agent `acme-orders` serves Cayu at `/cayu/`."""

    state: dict[str, Any] = {"agent": None}

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            target = urlsplit(self.path)
            origin = f"http://{self.headers['Host']}"
            cloud_answers = {
                "/v1/applications": {"items": [{"id": "acme-orders", "name": "Acme Orders"}]},
                "/v1/applications/acme-orders/service": {
                    "status": "running",
                    "cayu_url": f"{origin}/cayu/",
                },
                "/v1/applications/acme-orders/operator-credentials": {
                    "username": "operator",
                    "password": _PASSWORD,
                    "env": {"username": "U", "password": "P"},
                },
            }
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            if target.path in cloud_answers:
                status, answer = 200, cloud_answers[target.path]
            elif target.path.startswith("/cayu/"):
                status, answer = state["agent"](
                    method, self.headers.get("Authorization"), self.path.removeprefix("/cayu"), body
                )
            else:
                status, answer = 404, {"detail": "Not Found"}
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._answer("GET")

        def do_POST(self) -> None:
            self._answer("POST")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def served(cloud: dict[str, Any], project: Path) -> Iterator[dict[str, Any]]:
    """The Agent is a real Cayu server, its data moves mounted as `cayu serve` mounts them."""

    auth = BasicAuth(username="operator", password=_PASSWORD)
    server = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig(access=AuthenticatedAccess(dependency=auth)),
    )
    _include_data_moves(server, project, "/api", auth=auth)
    with TestClient(server) as client:

        def forward(method: str, authorization: str | None, target: str, body: Any):
            headers = {} if authorization is None else {"Authorization": authorization}
            response = client.request(method, target, headers=headers, json=body)
            return response.status_code, response.json()

        cloud["agent"] = forward
        yield cloud


def _cloud(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, Any]:
    code = main(["cloud", "data-move", *arguments, "--application", "acme-orders", "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_a_run_on_the_agent_is_waited_for_and_recorded_in_its_data_volume(
    served: dict[str, Any], project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _cloud(capsys, "run", "letters", "--page-size", "2")

    run = output["result"]
    assert code == 0
    assert run["status"] == "finished"
    assert run["report"]["counts"]["created"] == 5 and run["report"]["clean"] is True
    assert (project / "volume" / "data-moves" / f"{run['run_id']}.json").is_file()

    code, listing = _cloud(capsys, "runs")
    assert listing["result"]["moves"] == [{"name": "letters", "running": False}]
    assert [item["run_id"] for item in listing["result"]["runs"]] == [run["run_id"]]


def test_a_run_that_stopped_on_the_agent_resumes_from_its_record(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sys.modules.pop("letters_cloud", None)
    import letters_cloud

    letters_cloud.FAIL["at"] = "2"

    first_code, first = _cloud(capsys, "run", "letters", "--page-size", "2")
    resumed_code, resumed = _cloud(capsys, "resume", "letters")

    # As `cayu data-move run` does: a run that stopped exits 3, a clean one 0.
    assert (first_code, resumed_code) == (3, 0)
    assert first["result"]["status"] == "stopped"
    assert first["result"]["error"] == "ConnectionError: the old store went away"
    assert resumed["result"]["run_id"] == first["result"]["run_id"]
    assert resumed["result"]["status"] == "finished"
    assert resumed["result"]["report"]["counts"]["created"] == 5


def test_a_start_without_waiting_exits_zero(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sys.modules.pop("letters_cloud", None)
    import letters_cloud

    # The run will stop on an error, but the command only started it.
    letters_cloud.FAIL["at"] = "2"

    code, output = _cloud(capsys, "run", "letters", "--page-size", "2", "--no-wait")

    assert code == 0
    assert output["result"]["status"] == "running"
    assert "_exit_status" not in output


def test_a_run_that_was_not_clean_exits_three(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    sys.modules.pop("letters_cloud", None)
    import letters_cloud

    letters_cloud.TARGET.update({"a", "b"})
    letters_cloud._apply_original = letters_cloud._apply

    def conflicting(item, *, dry_run):
        from cayu.data_moves import StepOutcome, StepResult

        if item == "c":
            return StepResult(StepOutcome.CONFLICT, "edited in the app")
        return letters_cloud._apply_original(item, dry_run=dry_run)

    letters_cloud.MOVE = letters_cloud.DataMove("letters", letters_cloud._read, str, conflicting)

    code, output = _cloud(capsys, "run", "letters", "--page-size", "2")

    assert code == 3
    assert output["result"]["status"] == "finished"
    assert output["result"]["report"]["clean"] is False


def test_an_undeclared_move_is_refused(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _cloud(capsys, "run", "invoices")

    assert code == 2
    assert "No data move 'invoices' is declared" in output["error"]["message"]


def test_an_agent_on_an_older_cayu_is_unsupported(
    cloud: dict[str, Any], project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def older(method: str, authorization: str | None, target: str, body: Any):
        if target == "/api/contract":
            return 200, {"server_contract_version": "84"}
        return 404, {"detail": "Not Found"}

    cloud["agent"] = older
    code, output = _cloud(capsys, "run", "letters")

    assert code == 2
    assert output["error"]["category"] == "unsupported"
    assert "doesn't serve data moves" in output["error"]["message"]
    assert "[tool.cayu.data_moves]" in output["error"]["message"]


class _Gate:
    """A move whose first page waits until the test opens the gate."""

    def __init__(self) -> None:
        self.open = threading.Event()

    def read(self, cursor: str | None, limit: int) -> SourcePage[str]:
        self.open.wait(10)
        return SourcePage(["x"])

    def move(self) -> DataMove[str]:
        return DataMove(
            "gated", self.read, str, lambda item, *, dry_run: StepResult(StepOutcome.CREATED)
        )


def _router_client(gate: _Gate, *, auth: Any) -> tuple[TestClient, MemoryDataMoveRunStore]:
    store = MemoryDataMoveRunStore()
    app = FastAPI()
    app.include_router(data_move_router({"gated": gate.move()}, auth=auth, store=store))
    return TestClient(app), store


def test_one_run_of_a_move_goes_at_a_time_and_ids_are_not_reused() -> None:
    gate = _Gate()
    client, store = _router_client(gate, auth=lambda: None)

    first = client.post("/data-moves/gated/runs", json={"run_id": "one"})
    second = client.post("/data-moves/gated/runs", json={})
    gate.open.set()
    for _ in range(100):
        if store.get("one").status != "running":
            break
        threading.Event().wait(0.05)
    reused = client.post("/data-moves/gated/runs", json={"run_id": "one"})

    assert first.status_code == 202 and first.json()["status"] == "running"
    assert second.status_code == 409 and second.json()["detail"]["code"] == "data_move_running"
    assert store.get("one").status == "finished"
    assert reused.status_code == 409 and reused.json()["detail"]["code"] == "data_move_run_exists"


def test_a_server_without_authentication_runs_no_moves() -> None:
    client, store = _router_client(_Gate(), auth=None)

    answer = client.post("/data-moves/gated/runs", json={})

    assert answer.status_code == 409
    assert answer.json()["detail"]["code"] == "data_moves_require_authentication"
    assert store.runs == {}


def test_the_local_command_runs_a_declared_move_by_name(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data-move", "run", "letters", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report"]["counts"]["created"] == 5

    assert main(["data-move", "run", "invoices"]) == 2
    assert "no data move 'invoices' is declared" in capsys.readouterr().err


def test_a_run_its_process_left_running_reads_as_stopped(project: Path) -> None:
    from cayu.data_move_runs import DataMoveRun, open_run

    store = MemoryDataMoveRunStore()
    gate = _Gate()
    orphan = open_run(gate.move(), store, run_id="orphan")
    app = FastAPI()
    app.include_router(data_move_router({"gated": gate.move()}, auth=lambda: None, store=store))
    client = TestClient(app)

    answer = client.get("/data-moves/runs/orphan").json()

    assert answer["status"] == "stopped"
    assert answer["error"].startswith("interrupted")
    assert store.get("orphan") == orphan  # the record itself is left as it was
    assert isinstance(orphan, DataMoveRun)


def test_records_know_a_move_by_its_declared_name(project: Path) -> None:
    store = MemoryDataMoveRunStore()
    gate = _Gate()
    gate.open.set()
    app = FastAPI()
    app.include_router(data_move_router({"backfill": gate.move()}, auth=lambda: None, store=store))

    started = TestClient(app).post("/data-moves/backfill/runs", json={}).json()

    assert started["move"] == "backfill"


def test_a_declared_move_runs_from_a_project_subdirectory(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "sub").mkdir()
    monkeypatch.chdir(project / "sub")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(project)])
    sys.modules.pop("letters_cloud", None)

    assert main(["data-move", "run", "letters", "--no-record", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["created"] == 5


def test_a_service_project_is_told_how_to_serve_its_moves(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cayu.cli.serve import _data_moves_service_notice
    from cayu.server.data_moves import declared_data_move_sources

    _data_moves_service_notice(project, FastAPI())
    notice = capsys.readouterr().err
    assert "Data moves: not served" in notice
    assert "data_moves=declared_data_move_sources()" in notice

    served = FastAPI()
    served.include_router(
        data_move_router(declared_data_move_sources(project), auth=lambda: None),
        prefix="/cayu/api",
    )
    _data_moves_service_notice(project, served)
    assert "Data moves: /cayu/api/data-moves (letters)" in capsys.readouterr().out

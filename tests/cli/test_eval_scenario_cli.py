"""`cayu eval scenario` checks, saves and runs production scenarios through the server API.

The tests serve the real scenario routes, a durable eval store and an eval worker, and drive
the commands end to end: against a running server (`--server-url`) and against a Cloud Agent
(`--application`) reached through a fake Cayu Cloud, as `cayu cloud eval` is.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from tests.cli.test_cloud_evals import _AGENT, _PASSWORD
from tests.cli.test_eval_curation_cli import (
    _CURATION_AUTHORIZATION,
    _LOCAL,
    _authenticate_curation,
    _command,
    _ok,
    _refused,
    _served,
)
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.server.test_server_eval_scenarios import _scenario, _target
from tests.server.test_server_eval_session_additions import (
    addition_environment as addition_environment,
)
from tests.server.test_server_evaluation_promotion import _SESSION_ID

from cayu.cli import _eval_scenarios, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_agent_api import CayuApi, basic_authorization
from cayu.evals.scenario import EvalScenarioDocumentV2, eval_scenario_to_json
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvalsConfig,
    ServerConfig,
    _scenario_routes,
    create_server,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_BEARER = "Bearer scenario-cli-fixture-8b51c0e27d94"
_HEADERS = {"Authorization": _BEARER}
_BASIC = basic_authorization("operator", _PASSWORD)
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
# The release `tests.server.test_server_eval_scenarios._target` publishes.
_RELEASE = "release-current"
_WAIT_SECONDS = 10.0


def _authenticate(request: Request) -> AuthContext:
    """The operator by a bearer token locally, or by the login Cloud hands the CLI."""

    if request.headers.get("Authorization") in {_BEARER, _BASIC}:
        return AuthContext(subject="scenario-operator")
    raise HTTPException(status_code=401, detail="unauthorized")


@contextmanager
def _scenario_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, runs: int, release: str = _RELEASE
) -> Iterator[tuple[TestClient, ScriptedModelProvider]]:
    """A server with an eval worker whose agent answers ``runs`` scenario trials.

    Its store lives under ``tmp_path``, so a later server with another ``release`` is the
    same server after a deploy.
    """

    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("current scenario result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
            for _ in range(runs)
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    target = target.model_copy(update={"application_release_id": release})
    store = SQLiteEvalStore(tmp_path / "scenarios.db")
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                poll_interval_seconds=0.02,
                lease_seconds=5,
                shutdown_grace_seconds=2,
            ),
        ),
    )
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _BEARER)
    try:
        with _served(server, monkeypatch) as client:
            yield client, provider
    finally:
        asyncio.run(store.close())


def _scenario_file(tmp_path: Path, scenario: EvalScenarioDocumentV2) -> Path:
    path = tmp_path / "scenario.json"
    path.write_text(eval_scenario_to_json(scenario), encoding="utf-8")
    return path


def _renamed(scenario: EvalScenarioDocumentV2, name: str) -> EvalScenarioDocumentV2:
    return EvalScenarioDocumentV2.create(
        id=scenario.id, target_key=scenario.target_key, name=name, events=scenario.events
    )


def _finished(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        run = client.get(f"/api/evals/runs/{run_id}", headers=_HEADERS).json()
        if run["status"] in _TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError(f"Scenario run {run_id} did not finish.")


def test_scenario_check_then_run_admits_the_checked_revision_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario = _scenario()
    path = _scenario_file(tmp_path, scenario)
    with _scenario_server(tmp_path, monkeypatch, runs=2) as (client, provider):
        checked = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
        assert checked["scenario"] == scenario.model_dump(mode="json")
        assert checked["preflight"]["ready"] is True
        assert checked["execution_profile_revision"].startswith("sha256:")
        assert checked["entry"] is None
        # Checking saves and runs nothing.
        assert _ok(capsys, "scenario", "list", *_LOCAL)["items"] == []
        assert provider.requests == []

        # The run command pins the checked revision, binding and execution profile.
        run = _command(checked["commands"]["run"])
        assert run[:6] == (
            *("scenario", "run", "--file", str(path)),
            *("--expected-scenario-revision", scenario.revision),
        )
        assert run[-6:] == (
            *("--expected-binding-revision", checked["preflight"]["binding"]["revision"]),
            *("--expected-execution-profile-revision", checked["execution_profile_revision"]),
            *_LOCAL,
        )
        launched = _ok(capsys, *run)
        run_id = launched["run"]["id"]
        assert launched["scenario_revision"] == scenario.revision
        assert launched["operation_id"] == run[run.index("--operation-id") + 1]
        assert launched["binding_revision"] == launched["preflight"]["binding"]["revision"]
        assert _command(launched["commands"]["show"]) == ("runs", "show", run_id, *_LOCAL)
        # Running saved the checked revision to the library.
        [entry] = _ok(capsys, "scenario", "list", *_LOCAL)["items"]
        assert entry["revision"] == scenario.revision
        shown = _ok(capsys, "scenario", "show", scenario.revision, *_LOCAL)
        assert shown["scenario"] == checked["scenario"]

        # The same run command admits nothing new: it answers with the run it admitted.
        assert _ok(capsys, *run)["run"]["id"] == run_id
        assert _finished(client, run_id)["status"] == "completed"
        assert _ok(capsys, "runs", "show", run_id, *_LOCAL)["run"]["status"] == "completed"
        assert len(provider.requests) == 1
        assert _ok(capsys, *_command(launched["commands"]["continue"]))["run"]["id"] == run_id

        # The key can't admit another run of the scenario.
        error = _refused(
            capsys,
            *("scenario", "run", "--scenario", scenario.revision),
            *("--trial-timeout-seconds", "30", "--operation-id", launched["operation_id"]),
            *_LOCAL,
        )
        assert (error["category"], error["operation_id"]) == (
            "operation_id_conflict",
            launched["operation_id"],
        )

        # A scenario the server doesn't hold is saved from the file only while the file holds
        # the checked revision.
        second = _renamed(scenario, "Second request")
        _scenario_file(tmp_path, second)
        recheck = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
        _scenario_file(tmp_path, _renamed(scenario, "Edited request"))
        error = _refused(capsys, *_command(recheck["commands"]["run"]))
        assert error["category"] == "scenario_changed"
        assert error["expected_scenario_revision"] == second.revision
        assert len(_ok(capsys, "scenario", "list", *_LOCAL)["items"]) == 1
        assert len(_ok(capsys, "runs", "list", *_LOCAL)["items"]) == 1

        edited = _renamed(scenario, "Edited request").revision
        assert main(["eval", "scenario", "check", "--file", str(path), *_LOCAL, "--table"]) == 0
        assert main(["eval", "scenario", "list", *_LOCAL, "--table"]) == 0
        continuation = _command(launched["commands"]["continue"])
        assert main(["eval", *continuation, "--table"]) == 0
        text = capsys.readouterr().out
        assert f"Scenario Edited request ({edited}) is ready to run." in text
        assert "Run it: cayu eval scenario run --file" in text
        assert f"Retained request  1       0          {scenario.revision}" in text
        assert f"Scenario run {run_id}: completed." in text

        # An admitted run's continuation names its saved revision, so it needs no file.
        saved = _ok(capsys, "scenario", "check", "--file", str(path), "--save", *_LOCAL)
        admitted = _ok(capsys, *_command(saved["commands"]["run"]))
        assert _finished(client, admitted["run"]["id"])["status"] == "completed"
        path.unlink()
        continued = _ok(capsys, *_command(admitted["commands"]["continue"]))
        assert continued["run"]["id"] == admitted["run"]["id"]
        assert len(provider.requests) == 2


def test_a_lost_scenario_run_answer_is_continued_without_admitting_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        return httpx.Response(502) if method == "POST" and url.endswith("/runs") else response

    scenario = _scenario()
    path = _scenario_file(tmp_path, scenario)
    with _scenario_server(tmp_path, monkeypatch, runs=1) as (client, provider):
        run = ("scenario", "run", "--file", str(path), "--trial-timeout-seconds", "60", *_LOCAL)
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, *run)
        assert error["category"] == "outcome_unknown"
        continuation = _command(error["commands"]["continue"])
        # The continuation names the saved revision, the settings and the reviewed binding.
        assert continuation[:6] == (
            "scenario",
            "run",
            "--scenario",
            scenario.revision,
            "--trial-timeout-seconds",
            "60",
        )
        assert continuation[6:8] == ("--operation-id", error["operation_id"])
        assert continuation[-2:] == _LOCAL

        launched = _ok(capsys, *continuation)
        [admitted] = _ok(capsys, "runs", "list", *_LOCAL)["items"]
        assert launched["run"]["id"] == admitted["id"]
        assert launched["scenario"] is None
        assert _finished(client, admitted["id"])["status"] == "completed"
        assert len(provider.requests) == 1


def test_a_lost_answer_before_the_scenario_is_saved_keeps_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        return httpx.Response(502) if method == "POST" and url.endswith("/runs") else response

    scenario = _scenario()
    path = _scenario_file(tmp_path, scenario)
    with _scenario_server(tmp_path, monkeypatch, runs=1) as (client, provider):
        checked = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, *_command(checked["commands"]["run"]))
        assert error["category"] == "outcome_unknown"
        # The scenario isn't known to be saved, so the continuation still names its source.
        continuation = _command(error["commands"]["continue"])
        assert continuation[:6] == (
            *("scenario", "run", "--file", str(path)),
            *("--expected-scenario-revision", scenario.revision),
        )
        assert _ok(capsys, "scenario", "list", *_LOCAL)["items"] == []

        launched = _ok(capsys, *continuation)
        assert launched["scenario"]["revision"] == scenario.revision
        # Saved now, the scenario is named by its revision.
        assert _command(launched["commands"]["continue"])[:4] == (
            *("scenario", "run", "--scenario", scenario.revision),
        )
        assert _finished(client, launched["run"]["id"])["status"] == "completed"
        assert len(provider.requests) == 1


def test_scenarios_reach_a_cloud_agent_with_the_cli_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud = _FakeCloud()
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    on_cloud = ("--application", _AGENT)
    scenario = _scenario()
    path = _scenario_file(tmp_path, scenario)

    with _scenario_server(tmp_path, monkeypatch, runs=1) as (client, _):
        saved = _ok(capsys, "scenario", "check", "--file", str(path), "--save", *on_cloud)
        assert saved["application_id"] == _AGENT
        assert saved["entry"]["revision"] == scenario.revision
        listed = _ok(capsys, "scenario", "list", *on_cloud)
        assert [item["revision"] for item in listed["items"]] == [scenario.revision]

        checked = _ok(capsys, "scenario", "check", "--scenario", scenario.revision, *on_cloud)
        run = _command(checked["commands"]["run"])
        assert run[-2:] == on_cloud
        launched = _ok(capsys, *run)
        assert launched["application_id"] == _AGENT
        assert _finished(client, launched["run"]["id"])["status"] == "completed"
        # The local server is the same Agent: one run, one saved revision.
        assert _ok(capsys, *run[:-2], *_LOCAL)["run"]["id"] == launched["run"]["id"]
    assert "GET /v1/applications/bug-hunter-agent/operator-credentials" in cloud.requests


def test_a_checked_run_command_answers_with_its_run_after_a_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _scenario_file(tmp_path, _scenario())
    with _scenario_server(tmp_path, monkeypatch, runs=1) as (client, _):
        checked = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
        run = _command(checked["commands"]["run"])
        run_id = _ok(capsys, *run)["run"]["id"]
        assert _finished(client, run_id)["status"] == "completed"

    # The same server and store after a deploy, whose release changes the binding a check
    # reviews.
    with _scenario_server(tmp_path, monkeypatch, runs=0, release="release-next") as (_, agent):
        rechecked = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
        assert rechecked["preflight"]["binding"] != checked["preflight"]["binding"]
        # The checked command still answers with the run it admitted.
        assert _ok(capsys, *run)["run"]["id"] == run_id
        # A new key under the reviewed binding admits nothing.
        index = run.index("--operation-id") + 1
        error = _refused(capsys, *run[:index], "another-key", *run[index + 1 :])
        assert error["category"] == "scenario_changed"
        assert len(_ok(capsys, "runs", "list", *_LOCAL)["items"]) == 1
        assert agent.requests == []


def test_scenario_check_reads_the_scenario_a_finished_session_proposes(
    addition_environment, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, _ = addition_environment(auth=_authenticate_curation)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _CURATION_AUTHORIZATION)

    with _served(server, monkeypatch) as client:
        proposed = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers={"Authorization": _CURATION_AUTHORIZATION},
            json={},
        ).json()["scenario_conversion"]["scenario"]
        checked = _ok(capsys, "scenario", "check", "--session", _SESSION_ID, *_LOCAL)
        assert checked["scenario"] == proposed
        error = _refused(capsys, "scenario", "check", "--session", "missing-session", *_LOCAL)
        assert error["category"] == "not_found"


def test_scenario_options_and_files_are_checked_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    revision = "sha256:" + "a" * 64
    path = tmp_path / "scenario.json"
    for content, argv, message in (
        (
            None,
            ("check", "--file", str(path)),
            f"Could not read --file {path}: No such file or directory.",
        ),
        (
            "[]",
            ("check", "--file", str(path)),
            f"--file {path} must hold one scenario JSON object.",
        ),
        ("{", ("check", "--file", str(path)), f"--file {path} is not JSON: "),
        (
            None,
            ("check", "--scenario", revision, "--currency", "EUR"),
            "--currency is the currency of --max-cost; pass it with --max-cost.",
        ),
        (
            None,
            ("run", "--scenario", revision, "--expected-binding-revision", revision),
            "--expected-binding-revision and --expected-execution-profile-revision name a "
            "reviewed binding together; pass both, from a run's continue command.",
        ),
        (
            None,
            (
                *("run", "--session", "session-1", "--expected-binding-revision", revision),
                *("--expected-execution-profile-revision", revision),
            ),
            "A reviewed binding is of one scenario revision: pass --scenario REVISION or "
            "--expected-scenario-revision REV with it, from the printed command.",
        ),
        (
            None,
            ("run", "--scenario", revision, "--expected-scenario-revision", revision),
            "--scenario names the revision already; leave out --expected-scenario-revision.",
        ),
    ):
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content)
        error = _refused(capsys, "scenario", *argv)
        assert error["category"] == "invalid_input"
        assert error["message"].startswith(message)


def test_scenario_refusals_the_cli_tells_apart_are_the_servers_own() -> None:
    tree = ast.parse(inspect.getsource(_scenario_routes))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert _eval_scenarios._BINDING_CHANGED in literals
    assert _eval_scenarios._LAUNCH_NOT_READY in literals

"""`cayu eval capture preview|save|run` and `cayu eval baseline approve --result` evaluate a
finished session's captured evidence through the server API the dashboard's captured
evaluation details call.

The tests serve the real routes, a durable eval store and an eval worker over a seeded
finished session, and drive the commands end to end: against a running server
(`--server-url`) and against a Cloud Agent (`--application`) reached through a fake Cayu
Cloud.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException, Request
from tests.cli.test_cloud_evals import _AGENT, _PASSWORD
from tests.cli.test_eval_curation_cli import (
    _CURATION_AUTHORIZATION,
    _LOCAL,
    _command,
    _ok,
    _refused,
    _served,
)
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.server.test_server_eval_session_additions import (
    addition_environment as addition_environment,
)
from tests.server.test_server_evaluation_promotion import _SESSION_ID

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi, basic_authorization
from cayu.evals.corpus import EvalSuiteSpec, TrialRequestSpec, eval_suite_trial_policy
from cayu.server.auth import AuthContext

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_HEADERS = {"Authorization": _CURATION_AUTHORIZATION}
_CHECK = {"id": "done", "kind": "root_status", "expected": "completed"}


def _operator(request: Request) -> AuthContext:
    """The operator by a bearer token locally, or by the login Cloud hands the CLI."""

    allowed = {_CURATION_AUTHORIZATION, basic_authorization("operator", _PASSWORD)}
    if request.headers.get("Authorization") in allowed:
        return AuthContext(subject="eval-operator")
    raise HTTPException(status_code=401, detail="unauthorized")


def _finished(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = client.get(f"/api/evals/runs/{run_id}", headers=_HEADERS).json()
        if run["status"] in _TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError(f"Eval run {run_id} did not finish.")


def test_a_session_is_scored_saved_and_approved_as_the_baseline(
    addition_environment, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, _ = addition_environment(auth=_operator)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _CURATION_AUTHORIZATION)
    preview = f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview"

    with _served(server, monkeypatch) as client:
        # Preview score: the capture's own proposal, as the dashboard opens it.
        proposed = _ok(capsys, "capture", "preview", _SESSION_ID, *_LOCAL)
        served = client.post(preview, headers=_HEADERS, json={}).json()
        for key in ("baseline_revision", "candidate", "captured_score", "runnable_conversion"):
            assert proposed[key] == served[key]
        assert proposed["candidate"]["evidence"]["revision"].startswith("sha256:")
        revision = proposed["candidate"]["revision"]
        assert _command(proposed["commands"]["save"]) == (
            *("capture", "save", _SESSION_ID, "--expected-candidate-revision", revision),
            *_LOCAL,
        )

        # The checks and names a person edits, scored over the same evidence.
        edits = (
            "--case-name",
            "Refund finished",
            "--assertion",
            json.dumps(_CHECK, sort_keys=True),
        )
        edited = _ok(capsys, "capture", "preview", _SESSION_ID, *edits, *_LOCAL)
        case = edited["candidate"]["case"]
        assert (case["name"], [check["id"] for check in case["assertions"]]) == (
            "Refund finished",
            ["done"],
        )
        assert [item["assertion_id"] for item in edited["captured_score"]["assertions"]] == ["done"]
        assert edited["candidate"]["revision"] != revision
        # Previewing saves nothing.
        assert _ok(capsys, "suites", "list", *_LOCAL)["items"] == []

        # Save evaluation records that exact result; saving it again changes nothing.
        save = _command(edited["commands"]["save"])
        assert save[3:7] == edits
        saved = _ok(capsys, *save)
        record = saved["record"]
        assert saved["candidate_revision"] == edited["candidate"]["revision"]
        assert record["origin"] == "captured_session"
        assert _ok(capsys, *save)["record"] == record
        shown = _ok(capsys, *_command(saved["commands"]["show"]))
        assert [run["result_revision"] for run in shown["runs"]["items"]] == [record["revision"]]
        # A reviewed candidate the options no longer describe saves nothing.
        error = _refused(
            capsys,
            "capture",
            "save",
            _SESSION_ID,
            "--expected-candidate-revision",
            revision,
            *edits,
            *_LOCAL,
        )
        assert (error["category"], error["expected_candidate_revision"]) == (
            "capture_changed",
            revision,
        )

        # Approve baseline: the saved captured result, by its revision.
        approve = _command(saved["commands"]["baseline"])
        assert approve == ("baseline", "approve", "--result", record["revision"], *_LOCAL)
        approved = _ok(capsys, *approve)
        assert (approved["changed"], approved["baseline"]["generation"]) == (True, 1)
        assert approved["baseline"]["result_revision"] == record["revision"]
        assert _ok(capsys, *approve)["changed"] is False
        detail = client.get(f"/api/evals/results/{record['revision']}", headers=_HEADERS)
        assert detail.json()["baseline"] == approved["baseline"]

        assert main(["eval", "capture", "preview", _SESSION_ID, *edits, *_LOCAL, "--table"]) == 0
        assert main(["eval", *approve, "--table"]) == 0
        text = capsys.readouterr().out
        assert "Case " in text and "  done: passed" in text
        assert (
            f"Result {record['revision']} was already the baseline of suite "
            f"{record['suite_id']} (generation 1)." in text
        )


def test_a_session_runs_on_the_current_app_once_and_on_a_cloud_agent(
    addition_environment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server, _, _ = addition_environment(auth=_operator)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _CURATION_AUTHORIZATION)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _FakeCloud())
    on_cloud = ("--application", _AGENT)
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        lost = method == "POST" and url.endswith("/evaluation/launch")
        return httpx.Response(502) if lost else response

    with _served(server, monkeypatch) as client:
        proposed = _ok(capsys, "capture", "preview", _SESSION_ID, *on_cloud)
        assert proposed["application_id"] == _AGENT
        run = _command(proposed["commands"]["run"])
        assert run[:3] == ("capture", "run", _SESSION_ID)
        assert run[-2:] == on_cloud

        launched = _ok(capsys, *run, "--max-steps", "3")
        run_id = launched["run"]["id"]
        assert launched["record"]["origin"] == "captured_session"
        continuation = _command(launched["commands"]["continue"])
        assert continuation[3:5] == ("--max-steps", "3")
        assert continuation[continuation.index("--operation-id") + 1] == launched["operation_id"]
        assert (
            continuation[continuation.index("--expected-execution-profile-revision") + 1]
            == (launched["execution_profile_revision"])
        )
        # The continue command answers with the run its key admitted.
        assert _ok(capsys, *continuation)["run"]["id"] == run_id
        assert _command(launched["commands"]["show"]) == ("runs", "show", run_id, *on_cloud)
        assert _finished(client, run_id)["status"] == "completed"

        # A launch whose answer is lost is continued, admitting one run.
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, "capture", "run", _SESSION_ID, *_LOCAL)
        assert error["category"] == "outcome_unknown"
        resumed = _ok(capsys, *_command(error["commands"]["continue"]))
        assert resumed["operation_id"] == error["operation_id"]
        again = _ok(capsys, *_command(error["commands"]["continue"]))
        assert again["run"]["id"] == resumed["run"]["id"]
        listed = _ok(capsys, "runs", "list", *_LOCAL)["items"]
        assert sorted(item["id"] for item in listed) == sorted({run_id, resumed["run"]["id"]})
        assert _finished(client, resumed["run"]["id"])["status"] == "completed"

        # A continue command whose candidate changed since (a deploy changes it too) can't
        # repeat the launch, and says the run it may have admitted is listed, not that none ran.
        changed = (*continuation[:-2], "--case-name", "Renamed", *on_cloud)
        error = _refused(capsys, *changed)
        assert (error["category"], error["operation_id"]) == (
            "capture_changed",
            launched["operation_id"],
        )
        assert "whether it admitted a run is unknown" in error["message"]
        assert _command(error["commands"]["list"])[:3] == ("runs", "list", "--target-key")

        # Concurrency is the suite trial policy's: the launch carries the one asked for, as the
        # server's own concurrency check reads it. This target runs one trial, so it refuses.
        bodies: list[dict] = []

        def recorded(self: CayuApi, method: str, url: str, *args, **kwargs):
            if method == "POST" and url.endswith("/evaluation/launch"):
                bodies.append(args[1])
            return send(self, method, url, *args, **kwargs)

        with monkeypatch.context() as recording:
            recording.setattr(CayuApi, "_bounded_request", recorded)
            _refused(capsys, *run, "--trials", "2", "--max-concurrency", "2")
        trial_request = TrialRequestSpec.model_validate(bodies[-1]["trial_request"])
        suite = EvalSuiteSpec.create(id="refunds", name="Refunds", trial_request=trial_request)
        assert eval_suite_trial_policy(suite).max_concurrency == 2

        assert main(["eval", *continuation, "--table"]) == 0
        assert f"Run {run_id} on the current app:" in capsys.readouterr().out


def test_capture_options_are_checked_before_any_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    revision = "sha256:" + "a" * 64
    for argv, message in (
        (
            ("capture", "run", "session-1", "--currency", "EUR"),
            "--currency is the currency of --max-cost; pass it with --max-cost.",
        ),
        (
            ("capture", "run", "session-1", "--expected-execution-profile-revision", revision),
            "--expected-execution-profile-revision repeats an earlier admission; pass it with "
            "that admission's --operation-id and --expected-candidate-revision.",
        ),
        (
            ("baseline", "approve", "--result", revision, "--operation-id", revision),
            "--operation-id repeats an earlier approval; pass it with that approval's "
            "--expected-generation.",
        ),
    ):
        error = _refused(capsys, *argv)
        assert error["message"] == message

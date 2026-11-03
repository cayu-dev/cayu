"""`cayu eval scenario approve|deny|prepare-fixture` and `cayu eval runs cancel|report` act on
scenarios and runs through the server API the run page and scenario editor call.

The tests serve the real routes, a durable eval store and an eval worker, and drive the
commands end to end: against a running server (`--server-url`) and against a Cloud Agent
(`--application`) reached through a fake Cayu Cloud.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.cli.test_cloud_evals import _AGENT
from tests.cli.test_eval_curation_cli import _LOCAL, _command, _ok, _refused, _served
from tests.cli.test_eval_scenario_cli import (
    _BEARER,
    _HEADERS,
    _authenticate,
    _finished,
    _scenario_file,
)
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.server.test_server_eval_scenarios import (
    _approval_scenario,
    _ReviewTool,
    _scenario,
    _target,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.base import ArtifactScope
from cayu.cli import _eval_curation, _eval_scenarios, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_agent_api import CayuApi
from cayu.evals.execution import CorpusTarget
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.scenario import ScenarioArtifactRequirementV2
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import (
    DashboardConfig,
    EvalsConfig,
    ServerConfig,
    _evaluation_run_routes,
    create_server,
)
from cayu.sessions.requests import RunRequest
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_CHECKPOINT = "review-approval"
_WAIT_SECONDS = 20.0


class _CheckpointProvider(ModelProvider):
    """Each trial's agent calls `review_action` once, then answers."""

    name = "cli-scenario-checkpoint-provider"

    def __init__(self) -> None:
        self.request_count = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:cli-scenario-checkpoint-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count % 2:
            yield ModelStreamEvent.tool_call(
                id=f"call-review-{self.request_count}", name="review_action", arguments={}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("reviewed scenario result")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


@contextmanager
def _checkpoint_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: _ReviewTool
) -> Iterator[TestClient]:
    """A server whose agent asks a person to approve each `review_action` call."""

    app = CayuApp(enable_logging=False)
    app.register_provider(_CheckpointProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scenario-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    target = CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[], max_steps=8),
        application_release_id="release-current",
    )
    store = SQLiteEvalStore(tmp_path / "checkpoints.db")
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                # Each trial of a run starts from the application's own reset, so a scenario
                # may run two.
                execution_profile_policy=EvalExecutionProfilePolicyV1(
                    reset_strategy="application_managed",
                    isolation_revision="sha256:" + "a" * 64,
                    max_trials=2,
                ),
                poll_interval_seconds=0.02,
                lease_seconds=5,
                shutdown_grace_seconds=2,
            ),
        ),
    )
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _BEARER)
    try:
        with _served(server, monkeypatch) as client:
            yield client
    finally:
        asyncio.run(store.close())


def _awaiting_approval(client: TestClient, run_id: str, *, trial: int = 1) -> dict:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        run = client.get(f"/api/evals/runs/{run_id}", headers=_HEADERS).json()
        progress = run.get("scenario_progress")
        awaiting = progress is not None and progress["trials"][trial - 1]["phase"]
        if awaiting == "awaiting_approval":
            return run
        time.sleep(0.02)
    raise AssertionError(f"Trial {trial} of scenario run {run_id} did not pause for approval.")


def _on_cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _FakeCloud())
    return ("--application", _AGENT)


def test_a_scenario_checkpoint_takes_one_decision_and_the_run_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _ReviewTool()
    path = _scenario_file(tmp_path, _approval_scenario("assistant.default"))
    reports = (tmp_path / "result.eval-result.json", tmp_path / "report.html")
    with _checkpoint_server(tmp_path, monkeypatch, tool) as client:
        run = ("scenario", "run", "--file", str(path), "--trial-timeout-seconds", "30")
        run_id = _ok(capsys, *run, *_LOCAL)["run"]["id"]
        _awaiting_approval(client, run_id)

        # `runs show` names the checkpoint the trial awaits and its tool.
        shown = _ok(capsys, "runs", "show", run_id, *_LOCAL)
        [trial] = shown["scenario_progress"]["trials"]
        assert (trial["phase"], trial["pending_event_id"], trial["pending_tool_name"]) == (
            "awaiting_approval",
            _CHECKPOINT,
            "review_action",
        )
        assert main(["eval", "runs", "show", run_id, *_LOCAL, "--table"]) == 0
        assert (
            f"Trial 1: awaiting_approval, a fresh approval of review_action (event {_CHECKPOINT})"
            in capsys.readouterr().out
        )
        # A run still awaiting a decision has no report.
        error = _refused(
            capsys, "runs", "report", run_id, "--json-output", str(reports[0]), *_LOCAL
        )
        assert error["category"] == "eval_run_unfinished"
        assert _command(error["commands"]["show"]) == ("runs", "show", run_id, *_LOCAL)
        assert not reports[0].exists()

        # Another checkpoint, or a trial that doesn't exist, decides nothing.
        error = _refused(capsys, "scenario", "approve", run_id, "--event", "other", *_LOCAL)
        assert (error["category"], error["event_id"]) == ("scenario_approval_changed", _CHECKPOINT)
        error = _refused(capsys, "scenario", "approve", run_id, "--trial", "2", *_LOCAL)
        assert (error["category"], error["message"]) == (
            "scenario_approval_not_pending",
            f"Trial 2 of eval run {run_id} doesn't await a decision now; the run has no such "
            "trial.",
        )
        assert tool.run_count == 0

        approved = _ok(
            capsys,
            *("scenario", "approve", run_id, "--event", _CHECKPOINT),
            *("--reason", "The action is the reviewed one.", *_LOCAL),
        )
        assert (approved["decision"], approved["trial_number"], approved["event_id"]) == (
            "approve",
            1,
            _CHECKPOINT,
        )
        assert approved["tool_name"] == "review_action"
        assert approved["approval"]["actor_id"] == "scenario-operator"
        assert approved["approval"]["reason"] == "The action is the reviewed one."
        assert _command(approved["commands"]["show"]) == ("runs", "show", run_id, *_LOCAL)
        # The checkpoint has its decision: another records nothing.
        error = _refused(capsys, "scenario", "deny", run_id, "--trial", "1", *_LOCAL)
        assert error["category"] == "scenario_approval_not_pending"
        assert _finished(client, run_id)["status"] == "completed"
        assert tool.run_count == 1

        # The report is the server's: its JSON report and the HTML it renders.
        downloaded = _ok(
            capsys,
            *("runs", "report", run_id, "--json-output", str(reports[0])),
            *("--html-output", str(reports[1]), *_LOCAL),
        )
        assert downloaded["files"] == {"json": str(reports[0]), "html": str(reports[1])}
        assert downloaded["summary"]["status"] == "passed"
        for file, kind in zip(reports, ("json", "html"), strict=True):
            served = client.get(f"/api/evals/runs/{run_id}/report.{kind}", headers=_HEADERS)
            assert file.read_text(encoding="utf-8") == served.text
        assert main(["eval", "report", str(reports[0]), "--json"]) == 0
        assert capsys.readouterr().out.strip() == reports[0].read_text(encoding="utf-8").strip()

        rerun = _ok(capsys, *run, *_LOCAL)["run"]["id"]
        _awaiting_approval(client, rerun)
        assert main(["eval", "scenario", "approve", run_id, *_LOCAL, "--table"]) == 2
        assert main(["eval", "scenario", "deny", rerun, *_LOCAL, "--table"]) == 0
        captured = capsys.readouterr()
        assert (
            f"Error: Eval run {run_id} is completed, so none of its trials awaits a decision."
            in captured.err
        )
        assert (
            f"Denied review_action for trial 1 of run {rerun} (checkpoint {_CHECKPOINT})."
            in captured.out
        )
        assert _finished(client, rerun)["status"] == "completed"
        assert tool.run_count == 1


def test_decisions_and_cancellation_reach_a_cloud_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _ReviewTool()
    on_cloud = _on_cloud(tmp_path, monkeypatch)
    path = _scenario_file(tmp_path, _approval_scenario("assistant.default"))
    run = ("scenario", "run", "--file", str(path), "--trial-timeout-seconds", "30", *on_cloud)
    send = CayuApi._bounded_request
    decisions: list[str] = []

    def moved_once(self: CayuApi, method: str, url: str, *args, **kwargs):
        # Another trial's progress moves the run's revision before the first decision lands.
        if method == "POST" and url.endswith("/scenario-approval"):
            decisions.append(url)
            if len(decisions) == 1:
                return httpx.Response(409, json={"detail": _eval_scenarios._APPROVAL_CHANGED})
        return send(self, method, url, *args, **kwargs)

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        lost = method == "POST" and url.endswith("/scenario-approval")
        return httpx.Response(502) if lost else response

    with _checkpoint_server(tmp_path, monkeypatch, tool) as client:
        denied_run = _ok(capsys, *run)["run"]["id"]
        _awaiting_approval(client, denied_run)
        with monkeypatch.context() as racing:
            racing.setattr(CayuApi, "_bounded_request", moved_once)
            denied = _ok(capsys, "scenario", "deny", denied_run, *on_cloud)
        assert len(decisions) == 2
        assert (denied["application_id"], denied["decision"]) == (_AGENT, "deny")
        assert _finished(client, denied_run)["status"] == "completed"
        assert tool.run_count == 0

        # A lost answer's continue command decides only the checkpoint of the trial session
        # it was sent for. Every trial has the same checkpoints: once trial 1 has its decision
        # and finishes, trial 2 reaches the same checkpoint, and the continuation leaves it.
        lost_run = _ok(capsys, *run, "--trials", "2")["run"]["id"]
        session = _awaiting_approval(client, lost_run)["scenario_progress"]["trials"][0]
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, "scenario", "approve", lost_run, *on_cloud)
        assert (error["category"], error["event_id"]) == ("outcome_unknown", _CHECKPOINT)
        assert error["trial_session_id"] == session["session_id"]
        assert _command(error["commands"]["show"]) == ("runs", "show", lost_run, *on_cloud)
        continuation = _command(error["commands"]["continue"])
        assert continuation == (
            *("scenario", "approve", lost_run, "--trial", "1", "--event", _CHECKPOINT),
            *("--trial-session", session["session_id"], *on_cloud),
        )
        second = _awaiting_approval(client, lost_run, trial=2)["scenario_progress"]["trials"][1]
        assert second["pending_event_id"] == _CHECKPOINT
        error = _refused(capsys, *continuation)
        assert (error["category"], error["message"]) == (
            "scenario_approval_not_pending",
            f"Trial 1 of eval run {lost_run} doesn't await a decision now; it is completed.",
        )
        shown = _ok(capsys, "runs", "show", lost_run, *on_cloud)["scenario_progress"]
        assert shown["trials"][1]["approval"] is None
        assert tool.run_count == 1
        # A trial session that started over is not the one decided.
        error = _refused(
            capsys,
            *("scenario", "approve", lost_run, "--trial", "2", "--event", _CHECKPOINT),
            *("--trial-session", session["session_id"], *on_cloud),
        )
        assert error["category"] == "scenario_approval_changed"
        assert _ok(capsys, "scenario", "approve", lost_run, *on_cloud)["trial_number"] == 2
        assert _finished(client, lost_run)["status"] == "completed"
        assert tool.run_count == 2

        # A run waiting on a person is cancelled from the terminal, locally or on the Agent,
        # and `runs cancel` answers as `cayu cloud eval runs cancel` does.
        waiting = _ok(capsys, *run)["run"]["id"]
        _awaiting_approval(client, waiting)
        cancelled = _ok(capsys, "runs", "cancel", waiting, *_LOCAL)
        assert cancelled["run"]["status"] in {"cancelling", "cancelled"}
        # A run that is no longer running takes no decision, whatever its trials last showed.
        error = _refused(capsys, "scenario", "approve", waiting, *on_cloud)
        assert error["category"] == "scenario_approval_not_pending"
        assert error["message"].startswith(f"Eval run {waiting} is cancel")
        assert _finished(client, waiting)["status"] == "cancelled"
        assert main(["eval", "runs", "cancel", waiting, *on_cloud]) == 0
        answer = json.loads(capsys.readouterr().out)
        assert main(["cloud", "eval", "runs", "cancel", *on_cloud, waiting, "--json"]) == 0
        cloud_answer = json.loads(capsys.readouterr().out)
        # `cayu cloud` also says which Organization the change acted in.
        assert cloud_answer.pop("organization")["id"] == "org_test"
        assert cloud_answer == answer
        assert answer["result"]["run"]["status"] == "cancelled"
        assert tool.run_count == 2


def test_a_scenario_file_fixture_is_prepared_saved_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("reviewed the attachment"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
        ]
    )
    target, artifacts, _ = _target(tmp_path, provider)
    # A file the runtime attaches to a model request.
    content = b"%PDF-1.4\n% retained production attachment\n"

    async def retain():
        return await artifacts.put_bytes(
            content,
            filename="request.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.SESSION,
            session_id="production-session",
            environment_name="files",
        )

    artifact = asyncio.run(retain())
    requirement = ScenarioArtifactRequirementV2(
        id="request-file",
        source="artifact_reference",
        reference=artifact.id,
        content_sha256=hashlib.sha256(content).hexdigest(),
        filename=artifact.filename,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
    )
    scenario = _scenario(requirement)
    path = _scenario_file(tmp_path, scenario)
    store = SQLiteEvalStore(tmp_path / "fixtures.db")
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
    prepare = ("scenario", "prepare-fixture", "request-file", "--file", str(path))
    try:
        with _served(server, monkeypatch) as client:
            checked = _ok(capsys, "scenario", "check", "--file", str(path), *_LOCAL)
            assert [item["code"] for item in checked["preflight"]["diagnostics"]] == [
                "artifact_binding_required"
            ]
            assert checked["commands"] == {}

            # The retained file named by --artifact becomes the fixture; the run command
            # names the saved revision that uses it, without the spent --artifact.
            prepared = _ok(capsys, *prepare, "--artifact", f"request-file={artifact.id}", *_LOCAL)
            updated = prepared["scenario"]
            assert prepared["source_scenario_revision"] == scenario.revision
            assert updated["revision"] != scenario.revision
            assert updated["artifact_requirements"][0]["reference"] == prepared["artifact_id"]
            fixture = asyncio.run(artifacts.read_bytes(prepared["artifact_id"]))
            assert fixture.content == content
            assert fixture.metadata.scope is ArtifactScope.ENVIRONMENT
            assert prepared["entry"]["revision"] == updated["revision"]
            assert prepared["preflight"]["ready"] is True
            listed = _ok(capsys, "scenario", "list", *_LOCAL)["items"]
            assert [item["revision"] for item in listed] == [updated["revision"]]
            run = _command(prepared["commands"]["run"])
            assert run[:4] == ("scenario", "run", "--scenario", updated["revision"])
            assert "--artifact" not in run

            # Preparing again, from the file or the saved revision, changes nothing.
            for source in (("--file", str(path)), ("--scenario", updated["revision"])):
                again = _ok(capsys, *prepare[:3], *source, *_LOCAL)
                assert again["artifact_id"] == prepared["artifact_id"]
                assert again["scenario"]["revision"] == updated["revision"]
            assert len(_ok(capsys, "scenario", "list", *_LOCAL)["items"]) == 1

            launched = _ok(capsys, *run)
            assert _finished(client, launched["run"]["id"])["status"] == "completed"
            assert len(provider.requests) == 1

            assert main(["eval", *prepare, *_LOCAL, "--table"]) == 0
            text = capsys.readouterr().out
            assert (
                f"Prepared fixture {prepared['artifact_id']} for request-file; saved the "
                "scenario revision that uses it." in text
            )
            assert "Run it: cayu eval scenario run --scenario" in text

            error = _refused(capsys, *prepare[:2], "attachment", *prepare[3:], *_LOCAL)
            assert (error["category"], error["message"]) == (
                "requirement_not_found",
                "The scenario has no file requirement attachment; its file requirements are "
                "request-file.",
            )
            gone = requirement.model_copy(update={"reference": "art_" + "0" * 32})
            _scenario_file(tmp_path, _scenario(gone))
            error = _refused(capsys, *prepare, *_LOCAL)
            assert (error["category"], error["message"]) == (
                "fixture_not_prepared",
                "Preparing a fixture for request-file failed: A bound scenario artifact is no "
                "longer retained.",
            )
            assert len(_ok(capsys, "scenario", "list", *_LOCAL)["items"]) == 1
    finally:
        asyncio.run(store.close())


def test_report_files_are_checked_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report, link = tmp_path / "report.json", tmp_path / "link.json"
    report.write_text("kept", encoding="utf-8")
    link.hardlink_to(report)
    error = _refused(capsys, "runs", "report", "run-1")
    assert (error["category"], error["message"]) == (
        "invalid_input",
        "Name the report's file: --json-output FILE, --html-output FILE or both.",
    )
    # The report files and the answer are different files, also through a link to one, and
    # that is checked before --output is opened.
    for argv, message in (
        (
            ("--json-output", str(report), "--html-output", str(link)),
            "--json-output and --html-output name the same file.",
        ),
        (
            ("--json-output", str(link), "--output", str(report)),
            "--json-output and --output name the same file.",
        ),
    ):
        error = _refused(capsys, "runs", "report", "run-1", *argv)
        assert (error["category"], error["message"]) == ("invalid_input", message)
    assert report.read_text(encoding="utf-8") == "kept"
    # Nor is a report the Cloud context the command signs in with, however it is selected.
    context = tmp_path / "context.json"
    context.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(context))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    error = _refused(
        capsys, "runs", "report", "run-1", "--html-output", str(context), "--application", "a"
    )
    assert (error["category"], error["message"]) == (
        "invalid_input",
        "--html-output and the Cloud context name the same file.",
    )
    assert context.read_text(encoding="utf-8") == "{}"
    # Nor the API key file it signs in with. (The key is a placeholder.)
    key = tmp_path / "cloud.key"
    key.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("CAYU_CLOUD_API_KEY_FILE", str(key))
    error = _refused(
        capsys, "runs", "report", "run-1", "--json-output", str(key), "--application", "a"
    )
    assert error["message"] == "--json-output and the Cloud API key file name the same file."
    assert key.read_text(encoding="utf-8") == "placeholder"


def test_refusals_the_cli_tells_apart_are_the_servers_own() -> None:
    tree = ast.parse(inspect.getsource(_evaluation_run_routes))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert _eval_scenarios._APPROVAL_CHANGED in literals
    assert _eval_curation._NO_RESULT in literals

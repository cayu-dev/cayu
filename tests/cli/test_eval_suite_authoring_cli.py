"""`cayu eval targets list` and `cayu eval suites draft|check|run` author and run suites through
the server API the dashboard's suite editor and launch use.

The tests serve the real routes, a durable eval store and an eval worker, and drive the
commands end to end: against a running server (`--server-url`) and against a Cloud Agent
(`--application`) reached through a fake Cayu Cloud.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.cli.test_cloud_evals import _AGENT, _agent
from tests.cli.test_eval_curation_cli import _LOCAL, _command, _ok, _refused, _served
from tests.cli.test_eval_scenario_cli import _BASIC, _BEARER, _authenticate
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.evals.test_corpus_execution import _corpus
from tests.server.test_server_eval_scenarios import _target

from cayu.cli import _eval_suite_authoring as authoring
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.server import (
    DashboardConfig,
    EvalsConfig,
    ServerConfig,
    _suite_authoring_routes,
    _suite_launch_routes,
    create_server,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_WAIT_SECONDS = 20.0


@contextmanager
def _authoring_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, answers: int
) -> Iterator[tuple[TestClient, ScriptedModelProvider]]:
    """A server whose agent answers ``answers`` trials, with an execution profile an authored
    suite can launch on."""

    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("Refund approved."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
            for _ in range(answers)
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    store = SQLiteEvalStore(tmp_path / "suites.db")
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                execution_profile_policy=EvalExecutionProfilePolicyV1(
                    reset_strategy="application_managed",
                    isolation_revision="sha256:" + "a" * 64,
                    max_concurrency=2,
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
            yield client, provider
    finally:
        asyncio.run(store.close())


def _finished(client: TestClient, run_id: str, authorization: str = _BEARER) -> dict:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        run = client.get(
            f"/api/evals/runs/{run_id}", headers={"Authorization": authorization}
        ).json()
        if run["status"] in _TERMINAL:
            return run
        time.sleep(0.02)
    raise AssertionError(f"Eval run {run_id} did not finish.")


def _on_cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _FakeCloud())
    return ("--application", _AGENT)


def test_a_new_suite_is_drafted_checked_saved_and_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "suite.json"
    with _authoring_server(tmp_path, monkeypatch, answers=3) as (client, provider):
        targets = _ok(capsys, "targets", "list", *_LOCAL)
        [entry] = targets["items"]
        assert targets["default_target_key"] == entry["target_key"] == "assistant.default"
        assert entry["execution_profile_ready"] is True

        # New suite's draft: one sample case with the completed-session check.
        drafted = _ok(
            capsys,
            *("suites", "draft", "--new", "refund-regressions", "--name", "Refunds"),
            *("--file", str(path), *_LOCAL),
        )
        draft = json.loads(path.read_text(encoding="utf-8"))
        assert draft == drafted["draft"]
        assert (draft["id"], draft["name"], draft["target_key"]) == (
            "refund-regressions",
            "Refunds",
            "assistant.default",
        )
        assert [check["kind"] for check in draft["cases"][0]["assertions"]] == ["root_status"]
        assert drafted["parent_suite_revision"] is None
        error = _refused(capsys, "suites", "draft", "--new", "other", "--file", str(path), *_LOCAL)
        assert error["category"] == "invalid_input"
        assert json.loads(path.read_text(encoding="utf-8")) == draft

        # The edits a person makes in the editor: the case's input, and a duplicated case.
        draft["cases"][0]["stimulus"]["input"]["messages"][0]["text"] = "Refund order 42."
        duplicate = copy.deepcopy(draft["cases"][0])
        duplicate.update(id="case-2", name="Case 2")
        draft["cases"].append(duplicate)
        path.write_text(json.dumps(draft), encoding="utf-8")

        checked = _ok(capsys, "suites", "check", "--file", str(path), *_LOCAL)
        assert (checked["ready"], checked["diagnostics"], checked["entry"]) == (True, [], None)
        assert [case["id"] for case in checked["selection"]["cases"]] == ["case-1", "case-2"]
        save = _command(checked["commands"]["save"])
        assert save == ("suites", "check", "--file", str(path), "--save", *_LOCAL)
        # Checking saves nothing.
        assert _ok(capsys, "suites", "list", *_LOCAL)["items"] == []

        saved = _ok(capsys, *save)
        revision = checked["suite"]["revision"]
        assert (saved["suite"]["revision"], saved["entry"]["revision"]) == (revision, revision)
        suite = saved["suite_id"]
        assert [item["id"] for item in _ok(capsys, "suites", "list", *_LOCAL)["items"]] == [suite]
        # A revision is its content: saving it again changes nothing.
        assert _ok(capsys, *save)["entry"] == saved["entry"]

        # Check launch states the work and runs nothing.
        launch_check = _command(saved["commands"]["check_launch"])
        assert launch_check == ("suites", "run", suite, "--preview", *_LOCAL)
        previewed = _ok(capsys, *launch_check)
        assert previewed["runs"] == []
        assert previewed["preview"]["exposure"]["candidate_trials"] == 2
        assert provider.requests == []
        run = _command(previewed["commands"]["run"])
        assert run[:3] == ("suites", "run", suite)
        assert "--expected-exposure-revision" in run
        [group] = previewed["preview"]["launches"]
        assert run[run.index("--expected-execution-profile") + 1] == (
            f"case-1,case-2={group['execution_profile_revision']}"
        )

        launched = _ok(capsys, *run)
        [admitted] = launched["runs"]
        assert admitted["case_ids"] == ["case-1", "case-2"]
        run_id = admitted["run"]["id"]
        assert _command(launched["commands"]["list"]) == (
            *("runs", "list", "--target-key", "assistant.default", *_LOCAL),
        )
        # The run command answers with the runs its key admitted.
        assert [item["run"]["id"] for item in _ok(capsys, *run)["runs"]] == [run_id]
        assert _finished(client, run_id)["status"] == "completed"

        # A selection runs only its cases; its key can't admit another selection.
        subset = _ok(capsys, "suites", "run", suite, "--case", "case-2", *_LOCAL)
        [only] = subset["runs"]
        assert only["case_ids"] == ["case-2"]
        assert _finished(client, only["run"]["id"])["status"] == "completed"
        assert len(provider.requests) == 3
        error = _refused(
            capsys,
            *("suites", "run", suite, "--case", "case-1"),
            *("--operation-id", launched["operation_id"], *_LOCAL),
        )
        assert (error["category"], error["operation_id"]) == (
            "operation_id_conflict",
            launched["operation_id"],
        )
        assert len(_ok(capsys, "runs", "list", *_LOCAL)["items"]) == 2

        assert main(["eval", "targets", "list", *_LOCAL, "--table"]) == 0
        assert main(["eval", "suites", "check", "--file", str(path), *_LOCAL, "--table"]) == 0
        assert main(["eval", *launch_check, "--table"]) == 0
        text = capsys.readouterr().out
        assert "assistant.default (default)" in text
        assert f"Suite Refunds ({revision}) is ready to save." in text
        assert "At most 2 candidate trials and 0 judge evaluations, 1 at a time." in text
        assert "Run it: cayu eval suites run " in text


def test_a_saved_suite_is_edited_from_its_draft_and_run_on_a_cloud_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    on_cloud = _on_cloud(tmp_path, monkeypatch)
    first, edit = tmp_path / "suite.json", tmp_path / "edit.json"
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        return httpx.Response(502) if method == "POST" and url.endswith("/runs") else response

    with _authoring_server(tmp_path, monkeypatch, answers=1) as (client, provider):
        _ok(capsys, "suites", "draft", "--new", "refunds", "--file", str(first), *on_cloud)
        saved = _ok(capsys, "suites", "check", "--file", str(first), "--save", *on_cloud)
        assert saved["application_id"] == _AGENT
        suite, revision = saved["suite_id"], saved["suite"]["revision"]

        # Edit suite: the saved revision's draft, saved as its child.
        drafted = _ok(capsys, "suites", "draft", suite, "--file", str(edit), *on_cloud)
        assert drafted["parent_suite_revision"] == revision
        check = _command(drafted["commands"]["check"])
        assert check == (
            *("suites", "check", "--file", str(edit), "--save", "--parent", revision, *on_cloud),
        )
        # Unedited, the draft is the saved revision, and so is the revision's own JSON. The
        # save command a check prints keeps the parent.
        unedited = _ok(
            capsys, "suites", "check", "--file", str(edit), "--parent", revision, *_LOCAL
        )
        assert unedited["suite"]["revision"] == revision
        assert _command(unedited["commands"]["save"]) == (*check[:-2], *_LOCAL)
        document = tmp_path / "revision.json"
        document.write_text(
            client.get(f"/api/evals/suites/{revision}", headers={"Authorization": _BEARER}).text,
            encoding="utf-8",
        )
        from_document = _ok(capsys, "suites", "check", "--file", str(document), *_LOCAL)
        assert from_document["suite"]["revision"] == revision
        draft = json.loads(edit.read_text(encoding="utf-8"))
        draft["cases"][0]["name"] = "Refund approval"
        edit.write_text(json.dumps(draft), encoding="utf-8")
        child = _ok(capsys, *check)
        assert child["suite"]["revision"] != revision
        listed = _ok(capsys, "suites", "list", *on_cloud)["items"]
        assert {item["id"] for item in listed} == {suite, child["suite_id"]}
        # A saved revision keeps the parent it was saved with.
        error = _refused(capsys, "suites", "check", "--file", str(edit), "--save", *_LOCAL)
        assert (error["category"], error["message"]) == (
            "suite_conflict",
            f"The server didn't save revision {child['suite']['revision']}: it is already saved "
            "under another parent (a saved revision keeps its first).",
        )
        # A parent is a revision of the same suite.
        other = tmp_path / "other.json"
        _ok(capsys, "suites", "draft", "--new", "other-suite", "--file", str(other), *_LOCAL)
        error = _refused(
            capsys,
            "suites",
            "check",
            "--file",
            str(other),
            "--save",
            "--parent",
            revision,
            *_LOCAL,
        )
        assert error["category"] == "suite_conflict"
        assert error["message"].endswith(
            f", or --parent {revision} is not a saved revision of this suite here."
        )
        # So is a parent this server doesn't hold, such as one copied from another server's.
        unknown = "sha256:" + "0" * 64
        error = _refused(
            capsys, "suites", "check", "--file", str(other), "--save", "--parent", unknown, *_LOCAL
        )
        assert error["category"] == "suite_conflict"
        assert error["message"].endswith(
            f", or --parent {unknown} is not a saved revision of this suite here."
        )

        # A launch whose answer is lost is continued, admitting nothing twice.
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, "suites", "run", child["suite_id"], *on_cloud)
        assert error["category"] == "outcome_unknown"
        continuation = _command(error["commands"]["continue"])
        assert continuation[-2:] == on_cloud
        assert continuation[continuation.index("--operation-id") + 1] == error["operation_id"]
        [admitted] = _ok(capsys, *continuation)["runs"]
        [listed_run] = _ok(capsys, "runs", "list", *_LOCAL)["items"]
        assert admitted["run"]["id"] == listed_run["id"]
        assert _finished(client, listed_run["id"])["status"] == "completed"
        assert len(provider.requests) == 1


def test_a_suite_is_named_by_its_id_however_large_its_cases_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, edit = tmp_path / "suite.json", tmp_path / "edit.json"
    with _authoring_server(tmp_path, monkeypatch, answers=0):
        _ok(capsys, "suites", "draft", "--new", "long-answers", "--file", str(path), *_LOCAL)
        draft = json.loads(path.read_text(encoding="utf-8"))
        # One case larger than a suite view's default byte limit (1 MiB).
        draft["cases"][0]["assertions"] += [
            {"id": f"fact-{index}", "kind": "final_output_contains", "expected": "x" * 65_536}
            for index in range(17)
        ]
        path.write_text(json.dumps(draft), encoding="utf-8")
        suite = _ok(capsys, "suites", "check", "--file", str(path), "--save", *_LOCAL)["suite_id"]

        assert _ok(capsys, "suites", "run", suite, "--preview", *_LOCAL)["preview"]["ready"]
        drafted = _ok(capsys, "suites", "draft", suite, "--file", str(edit), *_LOCAL)
        assert json.loads(edit.read_text(encoding="utf-8")) == drafted["draft"]
        assert len(drafted["draft"]["cases"][0]["assertions"]) == 18

        error = _refused(capsys, "suites", "run", "not-a-suite", *_LOCAL)
        assert (error["category"], error["message"]) == (
            "invalid_input",
            "not-a-suite is not a suite id from `cayu eval suites list`.",
        )


def test_a_stored_corpus_suite_is_run_from_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _BASIC)
    with _agent(tmp_path, monkeypatch) as client:
        imported = client.post(
            "/api/evals/corpora",
            json=corpus.model_dump(mode="json"),
            headers={"Authorization": _BASIC},
        )
        assert imported.status_code in {200, 201}, imported.text
        [suite] = _ok(capsys, "suites", "list", *_LOCAL)["items"]
        assert suite["source"] == "corpus"

        previewed = _ok(capsys, "suites", "run", suite["id"], "--preview", *_LOCAL)
        assert previewed["bound"]["total_trials"] == 1
        assert previewed["runs"] == []
        run = _command(previewed["commands"]["run"])
        assert run[3:5] == ("--max-concurrency", "1")
        assert "--expected-execution-profile-revision" in run
        assert _ok(capsys, "runs", "list", *_LOCAL)["items"] == []

        # A second reviewed launch, not run before the deploy below.
        reviewed = _command(
            _ok(capsys, "suites", "run", suite["id"], "--preview", *_LOCAL)["commands"]["run"]
        )
        [admitted] = _ok(capsys, *run)["runs"]
        assert (admitted["kind"], admitted["case_ids"]) == ("corpus", None)
        run_id = admitted["run"]["id"]
        assert [item["run"]["id"] for item in _ok(capsys, *run)["runs"]] == [run_id]
        assert _finished(client, run_id, _BASIC)["status"] == "completed"
        assert [item["id"] for item in _ok(capsys, "runs", "list", *_LOCAL)["items"]] == [run_id]

        for argv, message in (
            (
                ("suites", "run", suite["id"], "--case", "refund-approval"),
                f"Suite {suite['id']} is a corpus suite, which runs whole; --case selects "
                "cases of an authored suite.",
            ),
            (
                ("suites", "draft", suite["id"], "--file", str(tmp_path / "draft.json")),
                f"Suite {suite['id']} is a suite of a stored corpus, edited as its corpus file; "
                "only an authored suite has a draft.",
            ),
        ):
            error = _refused(capsys, *argv, *_LOCAL)
            assert (error["category"], error["message"]) == ("invalid_input", message)

    # The same Agent after a deploy: a reviewed launch whose key admitted nothing is refused
    # under the profile it was reviewed with, and names the pin to leave out.
    with _agent(tmp_path, monkeypatch, release="release-2026-09-01"):
        error = _refused(capsys, *reviewed)
        assert error["category"] == "eval_release_changed"
        assert error["message"].endswith(
            "Start a new run without --operation-id and --expected-execution-profile-revision."
        )
        assert len(_ok(capsys, "runs", "list", *_LOCAL)["items"]) == 1


def test_suite_options_are_checked_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    revision = "sha256:" + "a" * 64
    draft = tmp_path / "suite.json"
    draft.write_text("{}", encoding="utf-8")
    for argv, message in (
        (
            ("draft", "suite-1", "--name", "Name", "--file", str(tmp_path / "new.json")),
            "--target-key and --name describe a new suite; pass them with --new.",
        ),
        (
            ("draft", "--new", "refunds", "--file", str(draft)),
            f"--file {draft} exists; name a new file, so no edits are overwritten.",
        ),
        (
            ("run", "suite-1", "--currency", "EUR"),
            "--currency is the currency of --max-cost; pass it with --max-cost.",
        ),
        (
            ("run", "suite-1", "--operation-id", "key", "--expected-exposure-revision", revision),
            "--expected-exposure-revision and --expected-execution-profile name a reviewed "
            "launch together; pass both, from a printed run command.",
        ),
        (
            ("run", "suite-1", "--expected-execution-profile-revision", revision),
            "A reviewed launch is repeated under its key; pass --operation-id with it, from the "
            "printed run command.",
        ),
        (
            ("run", "suite-1", "--preview", "--operation-id", "key"),
            "--preview reviews a new launch; leave out --operation-id and the expected revisions.",
        ),
        (
            ("run", "suite-1", "--case", "a", "--case", "a"),
            "Each --case names a different case.",
        ),
    ):
        error = _refused(capsys, "suites", *argv)
        assert (error["category"], error["message"]) == ("invalid_input", message)
    for value in ("case-1", f"={revision}", "case-1=sha256:short", f"a,,b={revision}"):
        with pytest.raises(SystemExit):
            main(["eval", "suites", "run", "s", "--expected-execution-profile", value])
        capsys.readouterr()


def test_suite_refusals_the_cli_tells_apart_are_the_servers_own() -> None:
    # Adjacent string literals are one constant once parsed, as the server sends them.
    literals = {
        node.value
        for module in (_suite_launch_routes, _suite_authoring_routes)
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert authoring._LAUNCH_NOT_READY in literals
    assert literals >= authoring._LAUNCH_CHANGED
    assert {authoring._SAVE_CONFLICT, authoring._SCENARIOS_CHANGED} <= literals

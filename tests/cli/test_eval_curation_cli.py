"""`cayu eval suites|runs|add-session|baseline` curate evals through the dashboard's server API.

Each test serves the real eval routes and a durable eval store, and drives the commands end to
end: against a running server (`--server-url`), and against a Cloud Agent (`--application`)
reached through a fake Cayu Cloud, as `cayu cloud eval` is.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.cli.test_cloud_evals import (
    _AGENT,
    _FAST,
    _PASSWORD,
    _agent,
    _corpus_file,
    _run_arguments,
)
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.evals.test_corpus_execution import _corpus, _provider, _target
from tests.server.test_server_eval_reporting import _REPORT_HEADERS, _configure, _payload
from tests.server.test_server_eval_session_additions import _destination
from tests.server.test_server_eval_session_additions import (
    addition_environment as addition_environment,
)
from tests.server.test_server_evals import _server
from tests.server.test_server_evaluation_promotion import _AUTH_HEADERS, _SESSION_ID

from cayu import Message, RunRequest
from cayu.cli import _eval_curation as curation
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi, basic_authorization
from cayu.server import _eval_addition_routes
from cayu.server.auth import BasicAuth
from cayu.storage.evals_sqlite import SQLiteEvalStore

_URL = "http://127.0.0.1"
_LOCAL = ("--server-url", _URL)


@contextmanager
def _served(server, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """``server`` as the CLI reaches it, at ``_URL``."""

    with TestClient(server, base_url=_URL) as client:
        # Each command closes the client it opened; keep this shared one for the next.
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield client


def _eval(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["eval", *argv])
    captured = capsys.readouterr()
    assert _PASSWORD not in captured.out + captured.err
    return code, json.loads(captured.out)


def _ok(capsys: pytest.CaptureFixture[str], *argv: str) -> dict:
    code, answer = _eval(capsys, *argv)
    assert code == 0 and answer["ok"] is True, answer
    return answer["result"]


def _refused(capsys: pytest.CaptureFixture[str], *argv: str) -> dict:
    code, answer = _eval(capsys, *argv)
    assert code == 2 and answer["ok"] is False, answer
    return answer["error"]


def _command(text: str) -> tuple[str, ...]:
    """A printed `cayu eval ...` command's arguments after `cayu eval`."""

    words = shlex.split(text)
    assert words[:2] == ["cayu", "eval"], text
    return tuple(words[2:])


def test_add_session_previews_then_publishes_the_reviewed_plan_once(
    addition_environment, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, _ = addition_environment()
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _AUTH_HEADERS["Authorization"])

    with _served(server, monkeypatch) as client:
        selector, parent = _destination(client, "authored")
        listed = _ok(capsys, "suites", "list", *_LOCAL)
        assert listed["server_url"] == _URL
        assert selector in [item["id"] for item in listed["items"]]

        add = ("add-session", _SESSION_ID, "--suite", selector, "--mode", "case")
        previewed = _ok(capsys, *add, "--preview", *_LOCAL)
        # Left out, the case takes the id and name the session's capture proposes, as in the
        # dashboard; the server keeps its proposed checks.
        capture = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers=_AUTH_HEADERS,
            json={},
        ).json()["candidate"]["case"]
        assert previewed["request"] == {
            "mode": "case",
            "destination": {"kind": "existing", "selector": selector},
            "case_id": capture["id"],
            "case_name": capture["name"],
        }
        assert previewed["receipt"] is None
        assert previewed["plan"]["membership_changed"] is True
        assert previewed["plan"]["destination_case_count"] == parent["cases"]["total"] + 1
        # A preview publishes nothing.
        assert _ok(capsys, "suites", "show", selector, *_LOCAL)["cases"] == parent["cases"]

        publish = _command(previewed["commands"]["publish"])
        assert publish[-6:] == (
            "--operation-id",
            previewed["operation_id"],
            "--expected-plan-revision",
            previewed["plan"]["revision"],
            "--expected-source-revision",
            previewed["plan"]["source_revision"],
        )
        published = _ok(capsys, *publish)
        assert published["plan"] == published["receipt"]["plan"] == previewed["plan"]
        destination = published["destination"]
        assert destination != selector
        shown = _ok(capsys, "suites", "show", destination, *_LOCAL)
        assert shown["suite"]["id"] == destination
        assert shown["cases"]["total"] == parent["cases"]["total"] + 1
        assert capture["id"] in [case["id"] for case in shown["cases"]["items"]]

        # Publishing again with the same key replays the receipt and publishes nothing new.
        assert _ok(capsys, *publish)["receipt"] == published["receipt"]
        recovered = client.get(
            f"/api/evals/additions/{published['receipt']['operation_id']}",
            headers=_AUTH_HEADERS,
        )
        assert recovered.json() == published["receipt"]
        assert _ok(capsys, "suites", "show", selector, *_LOCAL)["cases"] == parent["cases"]

        assert main(["eval", *add, "--preview", *_LOCAL, "--table"]) == 0
        assert main(["eval", "suites", "show", destination, *_LOCAL, "--table"]) == 0
        text = capsys.readouterr().out
        assert f"Preview of case {capture['id']}: a new revision with" in text
        assert f"Publish it: cayu eval add-session {_SESSION_ID}" in text
        assert f"Suite {shown['suite']['name']} (authored," in text
        assert "Baseline: one for each run's execution group" in text


def test_add_session_refuses_a_stale_plan_and_a_key_bound_to_another_addition(
    addition_environment, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, app, _ = addition_environment()
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _AUTH_HEADERS["Authorization"])

    async def identical_session() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="identical-session",
                messages=[Message.text("user", "promote this completed run")],
            )
        ):
            pass

    asyncio.run(identical_session())
    resolve = type(app)._resolve_public_session_id

    async def resolve_to_identical_session(self, public_session_id: str) -> str:
        if self is app and public_session_id == _SESSION_ID:
            return "identical-session"
        return await resolve(self, public_session_id)

    with _served(server, monkeypatch) as client:
        selector, parent = _destination(client, "authored")
        add = ("add-session", _SESSION_ID, "--suite", selector, "--mode", "case", *_LOCAL)
        previewed = _ok(capsys, *add, "--preview")

        # The session ID now names another session with the same evidence: the reviewed plan
        # no longer describes it, so publishing it publishes nothing.
        with monkeypatch.context() as moved:
            moved.setattr(type(app), "_resolve_public_session_id", resolve_to_identical_session)
            error = _refused(capsys, *_command(previewed["commands"]["publish"]))
        assert error["category"] == "eval_addition_changed"
        assert error["operation_id"] == previewed["operation_id"]
        assert error["message"].endswith("so nothing was published. Preview it again.")
        assert _ok(capsys, "suites", "show", selector, *_LOCAL)["cases"] == parent["cases"]

        # Without --preview the command previews and publishes in one go; the key it is given
        # can't publish a different addition afterwards.
        published = _ok(capsys, *add, "--operation-id", "add-regression")
        assert published["receipt"]["plan"] == published["plan"]
        error = _refused(
            capsys, *add, "--case-name", "Another name", "--operation-id", "add-regression"
        )
        assert (error["category"], error["operation_id"]) == (
            "operation_id_conflict",
            "add-regression",
        )


def test_a_lost_publication_answer_is_continued_without_publishing_twice(
    addition_environment, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, _ = addition_environment()
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _AUTH_HEADERS["Authorization"])
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        return httpx.Response(502) if method == "POST" and url.endswith("/additions") else response

    with _served(server, monkeypatch):
        add = ("add-session", _SESSION_ID, "--new-suite", "regressions", "--mode", "score")
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, *add, *_LOCAL)
        assert error["category"] == "outcome_unknown"
        publish = _command(error["commands"]["publish"])
        # The continuation names every default the first command took.
        assert publish[:6] == (
            "add-session",
            _SESSION_ID,
            "--new-suite",
            "regressions",
            "--suite-name",
            publish[5],
        )
        assert publish[-8:-6] == _LOCAL

        published = _ok(capsys, *publish)
        plan = published["plan"]
        assert (plan["mode"], plan["membership_changed"], plan["execution_case_count"]) == (
            "score",
            True,
            1,
        )
        assert published["receipt"]["operation_id"].startswith("sha256:")
        assert _ok(capsys, *publish)["receipt"] == published["receipt"]
        # One new captured suite, scored once.
        listed = _ok(capsys, "suites", "list", *_LOCAL)["items"]
        created = [item for item in listed if item["identity"].get("suite_id") == "regressions"]
        assert [item["id"] for item in created] == [published["destination"]]
        runs = _ok(capsys, "suites", "show", published["destination"], *_LOCAL)["runs"]
        assert [run["result_revision"] for run in runs["items"]] == [
            plan["captured_result_revision"]
        ]


def test_add_session_reaches_a_cloud_agent_with_the_cli_login(
    addition_environment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server, _, _ = addition_environment(auth=BasicAuth(username="operator", password=_PASSWORD))
    cloud = _FakeCloud()
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    on_cloud = ("--application", _AGENT)

    with _served(server, monkeypatch):
        created = _ok(
            capsys,
            *("add-session", _SESSION_ID, "--new-suite", "regressions"),
            *("--suite-name", "Regressions", "--mode", "case", "--case-id", "first", *on_cloud),
        )
        assert created["application_id"] == _AGENT
        assert created["plan"]["destination_case_count"] == 1
        assert _command(created["commands"]["publish"])[-8:-6] == on_cloud
        listed = _ok(capsys, "suites", "list", *on_cloud)
        assert [item["name"] for item in listed["items"]] == ["Regressions"]
        assert listed["items"][0]["id"] == created["destination"]

        grown = _ok(
            capsys,
            *("add-session", _SESSION_ID, "--suite", created["destination"]),
            *("--mode", "case", "--case-id", "second", *on_cloud),
        )
        assert grown["plan"]["destination_case_count"] == 2
        shown = _ok(capsys, "suites", "show", grown["destination"], *on_cloud)
        assert [case["id"] for case in shown["cases"]["items"]] == ["first", "second"]
    assert "GET /v1/applications/bug-hunter-agent/operator-credentials" in cloud.requests


def test_runs_and_baselines_answer_as_cayu_cloud_eval_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    suite_id = corpus.suites[0].id
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", basic_authorization("operator", _PASSWORD))
    on_cloud = ("--application", _AGENT)

    def cloud(*argv: str) -> dict:
        assert main(["cloud", "eval", *argv, "--json"]) == 0
        return json.loads(capsys.readouterr().out)

    with _agent(tmp_path, monkeypatch):
        first = cloud(*run, *_FAST)["result"]["run"]["id"]
        second = cloud(*run, *_FAST)["result"]["run"]["id"]

        # With --application, `cayu eval runs` is `cayu cloud eval runs`.
        for argv in (("runs", "list"), ("runs", "show", first)):
            assert main(["eval", *argv, *on_cloud]) == 0
            assert json.loads(capsys.readouterr().out) == cloud(*argv, *on_cloud)
        # A running server answers the same, named by its address.
        listed = _ok(capsys, "runs", "list", "--status", "completed", *_LOCAL)
        assert listed.pop("server_url") == _URL
        on_agent = cloud("runs", "list", "--status", "completed", *on_cloud)["result"]
        assert on_agent.pop("application_id") == _AGENT
        assert listed == on_agent
        assert [item["id"] for item in listed["items"]] == [second, first]
        shown = _ok(capsys, "runs", "show", first, *_LOCAL)
        assert shown["run"]["result"]["status"] == "passed"
        assert shown["progress"]["completed_trials"] == 1
        error = _refused(capsys, "runs", "show", "missing-run", *_LOCAL)
        assert (error["category"], error["message"]) == ("not_found", "Eval run not found.")

        suites = _ok(capsys, "suites", "list", "--limit", "1", *_LOCAL)
        assert suites["has_more"] is False
        suite = suites["items"][0]
        assert (suite["source"], suite["identity"]["suite_id"]) == ("corpus", suite_id)
        assert suite["last_run"]["native_run_id"] == second
        page = _ok(capsys, "suites", "show", suite["id"], "--runs-limit", "1", *_LOCAL)
        assert [item["native_run_id"] for item in page["runs"]["items"]] == [second]
        assert page["runs"]["has_more"] is True
        page = _ok(
            capsys,
            *("suites", "show", suite["id"], "--runs-cursor", page["runs"]["next_cursor"]),
            *_LOCAL,
        )
        assert [item["native_run_id"] for item in page["runs"]["items"]] == [first]

        approved = _ok(capsys, "baseline", "approve", first, *_LOCAL)
        assert (approved["changed"], approved["baseline"]["generation"]) == (True, 1)
        assert approved["baseline"]["result_revision"] == approved["run"]["result"]["revision"]
        again = _ok(capsys, "baseline", "approve", first, *on_cloud)
        assert (again["changed"], again["application_id"]) == (False, _AGENT)
        replaced = _ok(capsys, "baseline", "approve", second, *on_cloud)
        assert replaced["baseline"]["generation"] == 2
        assert (
            _ok(capsys, "suites", "show", suite["id"], *_LOCAL)["suite"]["baseline"]
            == (replaced["baseline"])
        )

        # A decision made over generation 1 is refused now the baseline is at generation 2.
        error = _refused(
            capsys,
            *("baseline", "approve", first, "--expected-generation", "1"),
            *("--operation-id", "sha256:" + "b" * 64, *_LOCAL),
        )
        assert error["category"] == "eval_baseline_conflict"
        assert error["message"].startswith(
            f"The baseline of suite {suite_id} moved to generation 2 after this approval was "
            "decided over generation 1, so it was not applied."
        )

        # An approval whose answer is lost is repeated by its continuation, as the same decision.
        send = CayuApi._bounded_request

        def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
            response = send(self, method, url, *args, **kwargs)
            return (
                httpx.Response(502) if method == "POST" and url.endswith("/baseline") else response
            )

        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, "baseline", "approve", first, *_LOCAL)
        assert (error["category"], error["expected_generation"]) == ("outcome_unknown", 2)
        continuation = _command(error["commands"]["continue"])
        assert continuation == (
            *("baseline", "approve", first, *_LOCAL),
            *("--operation-id", error["operation_id"], "--expected-generation", "2"),
        )
        replayed = _ok(capsys, *continuation)
        assert replayed["mutation"]["operation_id"] == error["operation_id"]
        assert replayed["baseline"]["generation"] == 3

        assert main(["eval", "runs", "list", *_LOCAL, "--table"]) == 0
        assert main(["eval", "suites", "list", *_LOCAL, "--table"]) == 0
        assert main(["eval", "baseline", "approve", first, *_LOCAL, "--table"]) == 0
        text = capsys.readouterr().out
        assert f"{second}  completed  {suite_id}" in text
        assert f"{suite['name']}  corpus  1      completed  {suite['id']}" in text
        assert f"Run {first} was already the baseline of suite {suite_id} (generation 3)." in text


def test_suites_show_a_reported_series_with_its_reported_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _target(_provider())
    _configure(monkeypatch, target.key)
    payload = asyncio.run(_payload(target.key))
    store = SQLiteEvalStore(tmp_path / "reports.db")
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _AUTH_HEADERS["Authorization"])
    try:
        with _served(_server(target, store), monkeypatch) as client:
            published = client.post("/api/evals/reports", json=payload, headers=_REPORT_HEADERS)
            assert published.status_code == 200, published.text
            run_key = published.json()["run_key"]

            # Without --server-url the commands reach `cayu serve`'s address, and name it.
            listed = _ok(capsys, "suites", "list")
            assert listed["server_url"] == "http://127.0.0.1:8000"
            [series] = listed["items"]
            assert series["source"] == "reported"
            shown = _ok(capsys, "suites", "show", series["id"])
            assert shown["suite"]["source"] == "reported"
            assert shown["suite"]["id"] == series["id"]
            # A reported series stores no cases; its runs are the reports.
            assert shown["cases"] is None
            assert [run["id"] for run in shown["runs"]["items"]] == [run_key]

            assert main(["eval", "suites", "show", series["id"], "--table"]) == 0
            text = capsys.readouterr().out
            assert f"Suite {series['name']} (reported," in text
            assert "Baseline: unavailable; reported runs can't be baselines" in text
            assert "Cases: none stored; a reported suite has only its reported runs." in text
            assert "\nCases\n" not in text
            assert run_key in text
    finally:
        asyncio.run(store.close())


def test_options_are_checked_before_any_request(capsys: pytest.CaptureFixture[str]) -> None:
    add = ("add-session", "session-1", "--mode", "case")
    revision = "sha256:" + "a" * 64
    for argv, message in (
        (
            (*add, "--suite", "suite-1", "--suite-name", "Name"),
            "--suite-name names a new suite; pass it with --new-suite.",
        ),
        (
            (*add, "--new-suite", "regressions", "--expected-plan-revision", revision),
            "--expected-plan-revision and --expected-source-revision name a previewed plan "
            "together; pass both, from its publish command.",
        ),
        (
            (
                *(*add, "--new-suite", "regressions", "--preview"),
                *("--expected-plan-revision", revision, "--expected-source-revision", revision),
            ),
            "--preview previews a new plan; leave out the expected revisions.",
        ),
        (
            ("runs", "list", "--context", "cloud-context.json"),
            "--context applies only with --application.",
        ),
    ):
        error = _refused(capsys, *argv)
        assert (error["category"], error["message"]) == ("invalid_input", message)


def test_publication_refusals_the_cli_tells_apart_are_the_servers_own() -> None:
    tree = ast.parse(inspect.getsource(_eval_addition_routes))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert curation._ADDITION_CHANGED in literals
    assert curation._ADDITION_KEY_CONFLICT in literals

"""`cayu cloud eval` runs and reads a Cloud Agent's evals through the Agent's own eval API.

Each test serves a real eval target and durable eval store as the Agent, behind the Agent's
Developer dashboard login, and drives the commands end to end with a fake Cayu Cloud.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import shlex
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.evals.test_corpus_execution import (
    _corpus,
    _model_judge_corpus,
    _model_judge_target,
    _price_book,
    _provider,
    _target,
)
from tests.server.test_server_evals import (
    _BlockingProvider,
    _evals_config,
    _repeatable_execution_policy,
)

from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.cli import _cloud_evals as cloud_evals
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.evals.corpus import EvalCorpusDocument, eval_corpus_to_json
from cayu.evals.execution import CorpusExecutionLimits, ModelJudgeTarget
from cayu.evals.store import EVAL_STORE_MAX_PAGE_SIZE, TERMINAL_EVAL_RUN_STATUSES, EvalRunStatus
from cayu.providers import ModelProvider
from cayu.server import (
    DashboardConfig,
    ServerConfig,
    _corpus_routes,
    _eval_run_admission,
    create_server,
)
from cayu.server.auth import BasicAuth
from cayu.storage.evals_sqlite import SQLiteEvalStore

_AGENT = "bug-hunter-agent"
_PASSWORD = "s3cret-password"
_FAST = ("--poll-seconds", "0.05")


@contextmanager
def _agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: ModelProvider | None = None,
    repeatable: bool = True,
    priced: bool = False,
    release: str = "release-2026-08-06",
    limits: CorpusExecutionLimits | None = None,
    model_judges: tuple[ModelJudgeTarget, ...] = (),
) -> Iterator[TestClient]:
    """The Agent's `cayu serve` with one eval target, reached as `cayu cloud` reaches it.

    Its eval store lives under ``tmp_path``, so a later `_agent` with another ``release`` is
    the same Agent after a deploy.
    """

    target = _target(
        provider or _provider(trials=12),
        price_book=_price_book() if priced else None,
        application_release_id=release,
        limits=limits,
        model_judges=model_judges,
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")
    policy = (
        {"execution_profile_policy": _repeatable_execution_policy(target)} if repeatable else {}
    )
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password=_PASSWORD),
            dashboard=DashboardConfig(enabled=False),
            evals=_evals_config(target, store, **policy),
        ),
    )
    cloud = _FakeCloud()
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    try:
        with TestClient(server, base_url="http://127.0.0.1") as client:
            # Each command closes the client it opened; keep this shared one for the next.
            monkeypatch.setattr(client, "close", lambda: None)
            monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
            yield client
    finally:
        asyncio.run(store.close())


def _corpus_file(tmp_path: Path, corpus: EvalCorpusDocument) -> Path:
    path = tmp_path / "evals.json"
    path.write_text(eval_corpus_to_json(corpus), encoding="utf-8")
    return path


def _run_arguments(path: Path, corpus: EvalCorpusDocument) -> tuple[str, ...]:
    return ("run", "--application", _AGENT, "--corpus", str(path), "--suite", corpus.suites[0].id)


def _cloud(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict, str]:
    code = main(["cloud", "eval", *argv, "--json"])
    captured = capsys.readouterr()
    assert _PASSWORD not in captured.out + captured.err
    return code, json.loads(captured.out), captured.err


def _continuation(command: str) -> tuple[str, ...]:
    """A printed `cayu cloud eval ...` command's arguments after `cayu cloud eval`."""

    words = shlex.split(command)
    assert words[:3] == ["cayu", "cloud", "eval"], command
    return tuple(words[3:])


def _ok(capsys: pytest.CaptureFixture[str], *argv: str) -> dict:
    code, answer, _ = _cloud(capsys, *argv)
    assert code == 0, answer
    return answer["result"]


def _refused(capsys: pytest.CaptureFixture[str], *argv: str) -> dict:
    code, answer, _ = _cloud(capsys, *argv)
    assert code == 2 and answer["ok"] is False, answer
    return answer["error"]


def test_run_waits_for_the_result_and_writes_the_report_eval_compare_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=2)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    first_report, second_report = tmp_path / "first.json", tmp_path / "second.json"

    with _agent(tmp_path, monkeypatch) as agent:
        code, answer, progress = _cloud(
            capsys, *run, *_FAST, "--operation-id", "release-check", "--output", str(first_report)
        )
        assert code == 0, answer
        first = answer["result"]
        run_id = first["run"]["id"]
        assert answer["operation"] == "eval.run"
        assert first["bound"] == {
            "target_key": "refund-agent",
            "suite_id": corpus.suites[0].id,
            "cases": 1,
            "trials_per_case": 2,
            "total_trials": 2,
            "max_concurrency": 1,
            "max_steps_per_trial": 1,
            "timeout_seconds_per_trial": 30,
            "candidate_cost_threshold": None,
            "judges": [],
        }
        assert (first["run"]["status"], first["output"]) == ("completed", str(first_report))
        assert first["summary"]["status"] == "passed"
        assert first["summary"]["trials"] == {
            "passed": 2,
            "failed": 0,
            "error": 0,
            "unavailable": 0,
        }
        # The bound and the operation ID come before admission, then the progress.
        assert progress.index("1 case x 2 trials = 2 trials") < progress.index("release-check")
        assert "No candidate cost threshold" in progress
        assert f"Run {run_id}: completed, 2/2 trials finished" in progress
        assert (
            first["commands"]["show"]
            == f"cayu cloud eval runs show --application {_AGENT} {run_id}"
        )
        report = json.loads(first_report.read_text())
        assert report["record_type"] == "cayu.eval-result-report"
        assert report["result"]["revision"] == first["summary"]["result_revision"]
        assert (
            first_report.read_bytes()
            == agent.get(
                f"/api/evals/runs/{run_id}/report.json", auth=("operator", _PASSWORD)
            ).content
        )

        # The same operation ID admits nothing new and goes on with the same run.
        assert _ok(capsys, *run, *_FAST, "--operation-id", "release-check")["run"]["id"] == run_id
        admitted = _ok(capsys, *run, "--no-wait", "--operation-id", "release-check")
        assert (admitted["run"]["id"], admitted["summary"]) == (run_id, None)
        second = _ok(capsys, *run, *_FAST, "--output", str(second_report))
        assert second["run"]["id"] != run_id
        assert second["operation_id"].startswith("cayu-cli-eval-")

        compared = _ok(capsys, "compare", "--application", _AGENT, run_id, second["run"]["id"])
        assert compared["comparison"]["compatibility"]["comparable"] is True
        assert compared["comparison"]["regressions"] == []
        assert [compared["baseline"]["id"], compared["current"]["id"]] == [
            run_id,
            second["run"]["id"],
        ]

    # The downloaded reports are what `cayu eval compare` and `cayu eval report` read.
    assert main(["eval", "compare", str(first_report), str(second_report)]) == 0
    assert json.loads(capsys.readouterr().out)["compatibility"]["comparable"] is True
    html = tmp_path / "first.html"
    assert (
        main(["eval", "report", str(first_report), "--format", "html", "--output", str(html)]) == 0
    )
    assert "Cayu Eval Report" in html.read_text()


def test_runs_are_listed_shown_cancelled_and_approved_as_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch):
        first = _ok(capsys, *run, *_FAST)["run"]["id"]
        second = _ok(capsys, *run, *_FAST)["run"]["id"]

        listed = _ok(capsys, "runs", "list", "--application", _AGENT, "--limit", "1")
        assert [item["id"] for item in listed["items"]] == [second]
        assert listed["has_more"] is True
        listed = _ok(capsys, "runs", "list", "--application", _AGENT, "--status", "completed")
        assert [item["id"] for item in listed["items"]] == [second, first]
        assert listed["items"][0]["suite_id"] == corpus.suites[0].id
        assert (
            _ok(
                capsys,
                "runs",
                "list",
                "--application",
                _AGENT,
                "--target-key",
                "refund-agent",
                "--status",
                "failed",
            )["items"]
            == []
        )

        shown = _ok(capsys, "runs", "show", "--application", _AGENT, first)
        assert shown["run"]["result"]["status"] == "passed"
        assert shown["progress"]["completed_trials"] == shown["progress"]["total_trials"] == 1
        # A finished run stays finished; cancelling it changes nothing.
        assert _ok(capsys, "runs", "cancel", "--application", _AGENT, first)["run"]["status"] == (
            "completed"
        )

        approved = _ok(capsys, "baseline", "--application", _AGENT, first)
        assert approved["changed"] is True and approved["baseline"]["generation"] == 1
        assert approved["operation_id"].startswith("sha256:")
        # Approving the same run again finds it already approved.
        again = _ok(capsys, "baseline", "--application", _AGENT, first)
        assert again["changed"] is False and again["baseline"]["generation"] == 1
        replaced = _ok(capsys, "baseline", "--application", _AGENT, second)
        assert replaced["baseline"]["generation"] == 2
        assert replaced["baseline"]["result_revision"] == replaced["run"]["result"]["revision"]

        for argv in (
            ("runs", "show", "--application", _AGENT, first),
            ("runs", "cancel", "--application", _AGENT, first),
            ("compare", "--application", _AGENT, first, second),
            ("baseline", "--application", _AGENT, second),
        ):
            assert main(["cloud", "eval", *argv]) == 0, argv
        text = capsys.readouterr().out
        assert f"Run {first}: completed, 1/1 trials finished" in text
        assert f"Run {first} already ended completed; cancelling it changed nothing." in text
        assert f"Baseline {first}: passed 1" in text and "Comparable; 0 regressions." in text
        assert (
            f"Run {second} was already the baseline of suite {corpus.suites[0].id} (generation 2)."
        ) in text

        error = _refused(capsys, "runs", "show", "--application", _AGENT, "missing-run")
        assert (error["category"], error["message"]) == ("not_found", "Eval run not found.")


def test_run_prints_a_summary_and_sends_the_candidate_cost_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=2)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch, priced=True):
        code = main(["cloud", "eval", *run, *_FAST, "--max-cost", "0.25"])
        captured = capsys.readouterr()

        assert code == 0, captured.err
        assert (
            "Candidate cost threshold: each trial's agent is interrupted once its observed cost "
            "reaches 0.25 USD; one model call can cross it before its cost is known."
        ) in captured.err
        lines = captured.out.splitlines()
        assert lines[0].startswith("Run ") and lines[0].endswith(": completed (passed, score 1)")
        assert lines[1] == "Trials: 2 passed, 0 failed, 0 error, 0 unavailable"
        run_id = lines[0].split()[1].rstrip(":")
        shown = _ok(capsys, "runs", "show", "--application", _AGENT, run_id)
        assert shown["run"]["cost_budget"] == {"max_estimated_cost": "0.25", "currency": "USD"}

        assert main(["cloud", "eval", "runs", "list", "--application", _AGENT]) == 0
        table = capsys.readouterr().out.splitlines()
        assert table[0].split() == ["RUN", "STATUS", "SUITE", "CREATED", "RESULT"]
        assert table[1].split()[:3] == [run_id, "completed", corpus.suites[0].id]


def test_refuses_a_corpus_for_a_target_this_release_does_not_have(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(target_key="triage-agent")
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch) as agent:
        error = _refused(capsys, *run)

        assert error["category"] == "eval_target_not_found"
        assert error["message"] == (
            "This release has no eval target named triage-agent; its targets are refund-agent."
        )
        assert error["targets"] == ["refund-agent"]
        catalog = agent.get("/api/evals/corpora", auth=("operator", _PASSWORD)).json()
        assert catalog["items"] == []


def test_refuses_runs_beyond_the_targets_limits_before_importing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    two_trials = _corpus(trials=2)
    one_trial = _corpus(trials=1)
    provider = _provider(trials=1)

    with _agent(tmp_path, monkeypatch, provider=provider, repeatable=False) as agent:
        error = _refused(capsys, *_run_arguments(_corpus_file(tmp_path, two_trials), two_trials))
        assert error["category"] == "eval_run_out_of_bounds"
        assert error["message"].startswith(
            "Suite refund-regressions runs 2 trials per case, and the eval target refund-agent "
            "allows at most 1. The trial count is part of the corpus"
        )

        single = _run_arguments(_corpus_file(tmp_path, one_trial), one_trial)
        error = _refused(capsys, *single, "--max-concurrency", "2")
        assert (
            "--max-concurrency 2 is more than the eval target refund-agent allows (1)."
            in (error["message"])
        )
        assert "suite refund-regressions's trial policy allows (1)" in error["message"]
        error = _refused(capsys, *single, "--max-cost", "1")
        assert error["message"] == (
            "The eval target refund-agent publishes no pricing, so it can't interrupt a trial "
            "at a cost; run it without --max-cost."
        )
        error = _refused(capsys, *single, "--currency", "EUR")
        assert error["message"] == "--currency applies only with --max-cost."
        error = _refused(capsys, *single, "--suite", "missing")
        assert error["category"] == "suite_not_found"

        # Nothing reached the Agent's store or its model.
        assert agent.get("/api/evals/corpora", auth=("operator", _PASSWORD)).json()["items"] == []
        assert provider.requests == []


def test_an_agent_without_eval_routes_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    server = create_server(
        _target(_provider(trials=1)).app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password=_PASSWORD),
            dashboard=DashboardConfig(enabled=False),
        ),
    )
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _FakeCloud())
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)

        for argv in (
            _run_arguments(_corpus_file(tmp_path, corpus), corpus),
            ("runs", "list", "--application", _AGENT),
        ):
            error = _refused(capsys, *argv)
            assert error["category"] == "evals_unavailable"
            assert error["message"].startswith(f"The Agent {_AGENT} serves no evals")


def test_trial_count_is_not_a_run_option(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "cloud",
            "eval",
            "run",
            "--application",
            _AGENT,
            "--corpus",
            "e.json",
            "--suite",
            "s",
            "--trials",
            "3",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "--trials" in error["message"]


def test_a_wait_that_runs_out_names_the_run_and_cancelling_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    provider = _BlockingProvider()

    with _agent(tmp_path, monkeypatch, provider=provider):
        error = _refused(
            capsys,
            *run,
            "--operation-id",
            "slow-run",
            "--wait-seconds",
            "0.3",
            "--poll-seconds",
            "0.1",
        )
        assert error["category"] == "eval_run_still_running"
        run_id = error["run_id"]
        assert error["message"].startswith(f"The eval run {run_id} is still ")
        assert error["operation_id"] == "slow-run"
        assert (
            error["commands"]["show"]
            == f"cayu cloud eval runs show --application {_AGENT} {run_id}"
        )
        continuation = _continuation(error["commands"]["continue"])
        assert continuation[-4:-1] == (
            "--operation-id",
            "slow-run",
            "--execution-profile-revision",
        )
        assert provider.started.wait(timeout=10)

        cancelled = _ok(capsys, "runs", "cancel", "--application", _AGENT, run_id)
        assert cancelled["run"]["status"] in {"cancelling", "cancelled"}
        deadline = time.monotonic() + 30
        while _ok(capsys, "runs", "show", "--application", _AGENT, run_id)["run"]["status"] != (
            "cancelled"
        ):
            assert time.monotonic() < deadline
            time.sleep(0.05)

        # Continuing the same admission reports how the run ended.
        error = _refused(capsys, *continuation, *_FAST)
        assert (error["category"], error["run_id"]) == ("eval_run_cancelled", run_id)

        assert main(["cloud", "eval", *continuation, *_FAST]) == 2
        rendered = capsys.readouterr().err
        assert f"Error: The eval run {run_id} was cancelled." in rendered
        assert f"Run: {run_id}" in rendered


def test_a_lost_admission_answer_is_continued_with_the_printed_operation_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    send = CayuApi._bounded_request

    def lose_admission_answer(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        # The Agent admitted the run; a proxy lost its answer.
        return (
            httpx.Response(502)
            if (method, url.endswith("/evals/runs")) == ("POST", True)
            else response
        )

    with _agent(tmp_path, monkeypatch):
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", lose_admission_answer)
            error = _refused(capsys, *run, *_FAST)
        assert error["category"] == "outcome_unknown"
        assert error["commands"]["continue"] in error["message"]
        continuation = _continuation(error["commands"]["continue"])
        assert continuation[-4:-2] == ("--operation-id", error["operation_id"])

        continued = _ok(capsys, *continuation, *_FAST)
        listed = _ok(capsys, "runs", "list", "--application", _AGENT)
        assert [item["id"] for item in listed["items"]] == [continued["run"]["id"]]
        assert continued["summary"]["status"] == "passed"


def test_a_wait_rides_out_reads_the_agent_cannot_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    send = CayuApi._bounded_request
    dropped: list[str] = []

    def drop_first_run_read(self: CayuApi, method: str, url: str, *args, **kwargs):
        if method == "GET" and "/evals/runs/" in url and not dropped:
            dropped.append(url)
            raise httpx.ConnectError("connection refused")
        return send(self, method, url, *args, **kwargs)

    with _agent(tmp_path, monkeypatch):
        monkeypatch.setattr(CayuApi, "_bounded_request", drop_first_run_read)
        code, answer, progress = _cloud(capsys, *run, *_FAST)

    assert code == 0, answer
    assert answer["result"]["run"]["status"] == "completed"
    assert len(dropped) == 1
    run_id = answer["result"]["run"]["id"]
    assert f"Could not connect to the agent {_AGENT}. Run {run_id} goes on in the Agent" in progress


def test_the_printed_continuation_replays_an_admission_after_a_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch):
        first = _ok(capsys, *run, *_FAST)
    reviewed = first["execution_profile_revision"]
    continuation = _continuation(first["commands"]["continue"])
    assert continuation[-2:] == ("--execution-profile-revision", reviewed)

    # A deploy: the same Agent and store, on a release whose target publishes a new profile.
    with _agent(tmp_path, monkeypatch, release="release-2026-09-01") as agent:
        catalog = agent.get("/api/evals/targets", auth=("operator", _PASSWORD)).json()
        assert catalog["items"][0]["execution_profile"]["revision"] != reviewed

        replayed = _ok(capsys, *continuation, *_FAST)
        assert replayed["run"]["id"] == first["run"]["id"]
        assert replayed["summary"] == first["summary"]

        # A key that admitted nothing under the old release has no run to continue.
        error = _refused(
            capsys,
            *run,
            *_FAST,
            "--operation-id",
            "never-admitted",
            "--execution-profile-revision",
            reviewed,
        )
        assert error["category"] == "eval_release_changed"
        assert error["message"].startswith(
            "The Agent's release changed before this run was admitted, so there is no run to "
            "continue"
        )
        assert error["message"].endswith(
            "Start a new run without --operation-id and --execution-profile-revision."
        )
        listed = _ok(capsys, "runs", "list", "--application", _AGENT)
        assert [item["id"] for item in listed["items"]] == [first["run"]["id"]]

        error = _refused(capsys, *run, "--execution-profile-revision", reviewed)
        assert error["message"].startswith("--execution-profile-revision repeats an earlier")


def test_a_baseline_retry_repeats_its_decision_and_never_overrides_a_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)
    send = CayuApi._bounded_request

    def never_sent(self: CayuApi, method: str, url: str, *args, **kwargs):
        if method == "POST" and url.endswith("/baseline"):
            raise httpx.ConnectError("connection reset")
        return send(self, method, url, *args, **kwargs)

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        return httpx.Response(502) if method == "POST" and url.endswith("/baseline") else response

    with _agent(tmp_path, monkeypatch) as agent:
        first = _ok(capsys, *run, *_FAST)["run"]["id"]
        second = _ok(capsys, *run, *_FAST)["run"]["id"]

        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", never_sent)
            error = _refused(capsys, "baseline", "--application", _AGENT, first)
        assert (error["category"], error["expected_generation"]) == ("outcome_unknown", 0)
        lost = _continuation(error["commands"]["continue"])
        assert lost[-4:] == ("--operation-id", error["operation_id"], "--expected-generation", "0")

        # Someone approves the other run meanwhile; repeating the lost decision must not undo it.
        approved = _ok(capsys, "baseline", "--application", _AGENT, second)
        assert approved["baseline"]["generation"] == 1
        error = _refused(capsys, *lost)
        assert error["category"] == "eval_baseline_conflict"
        assert error["message"].startswith(
            "The baseline of suite refund-regressions moved to generation 1 after this approval "
            "was decided over generation 0, so it was not applied."
        )
        current = agent.get(f"/api/evals/runs/{second}/result", auth=("operator", _PASSWORD))
        assert current.json()["baseline"] == approved["baseline"]

        # An approval applied whose answer was lost replays as that same decision.
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, "baseline", "--application", _AGENT, first)
        assert (error["category"], error["expected_generation"]) == ("outcome_unknown", 1)
        replayed = _ok(capsys, *_continuation(error["commands"]["continue"]))
        assert replayed["changed"] is True
        assert replayed["mutation"]["operation_id"] == error["operation_id"]
        assert replayed["mutation"]["resulting_generation"] == 2
        assert replayed["baseline"]["generation"] == 2

        error = _refused(
            capsys,
            "baseline",
            "--application",
            _AGENT,
            first,
            "--operation-id",
            "sha256:" + "a" * 64,
        )
        assert error["message"].startswith("--operation-id repeats an earlier approval")


def test_a_continuation_replays_before_reimporting_a_corpus_a_new_release_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=2)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch):
        first = _ok(capsys, *run, *_FAST)
    continuation = _continuation(first["commands"]["continue"])

    # The new release allows one trial per case: it refuses to import this corpus again.
    with _agent(
        tmp_path,
        monkeypatch,
        release="release-2026-09-01",
        limits=CorpusExecutionLimits(max_trials=1),
    ) as agent:
        imported = agent.post(
            "/api/evals/corpora",
            json=corpus.model_dump(mode="json"),
            auth=("operator", _PASSWORD),
        )
        assert imported.status_code == 400, imported.text

        replayed = _ok(capsys, *continuation, *_FAST)
        assert replayed["run"]["id"] == first["run"]["id"]
        assert replayed["summary"] == first["summary"]


def test_a_continuation_names_an_operation_id_that_admitted_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    path = _corpus_file(tmp_path, corpus)

    with _agent(tmp_path, monkeypatch):
        first = _ok(capsys, *_run_arguments(path, corpus), *_FAST)
    continuation = _continuation(first["commands"]["continue"])

    # After a deploy, the corpus file the continuation names has changed.
    changed = _corpus(trials=1, input_text="Refund order 43.")
    assert changed.revision != corpus.revision
    path.write_text(eval_corpus_to_json(changed), encoding="utf-8")
    with _agent(tmp_path, monkeypatch, release="release-2026-09-01"):
        error = _refused(capsys, *continuation, *_FAST)
        assert error["category"] == "operation_id_conflict"
        assert error["message"].startswith(
            f"Operation ID {first['operation_id']} already admitted a different eval run "
            "request (its corpus file or options have changed since), so this one admitted "
            "nothing. The run it admitted still exists:"
        )
        assert "new run" not in error["message"]
        listed = _ok(capsys, *_continuation(error["commands"]["list"]))
        assert [item["id"] for item in listed["items"]] == [first["run"]["id"]]


def test_the_bound_tells_the_candidate_cost_threshold_apart_from_ai_judge_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    judge_prices = PriceBook(
        price_book_version="judge-prices-v1",
        generated_at="2026-08-30T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="judge-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        ),
    )
    judge, _ = _model_judge_target(max_estimated_cost="0.1", price_book=judge_prices)
    corpus = _model_judge_corpus(judge)
    run = _run_arguments(_corpus_file(tmp_path, corpus), corpus)

    with _agent(tmp_path, monkeypatch, priced=True, model_judges=(judge,)) as agent:
        code, answer, progress = _cloud(capsys, *run, *_FAST, "--max-cost", "0.25")
        catalog = agent.get("/api/evals/targets", auth=("operator", _PASSWORD)).json()

    assert code == 0, answer
    profile = catalog["items"][0]["judge_profiles"][0]
    bound = answer["result"]["bound"]
    assert bound["candidate_cost_threshold"] == {"per_trial": "0.25", "currency": "USD"}
    assert bound["judges"] == [
        {
            "key": "quality-judge",
            "evaluations": 1,
            "limits": {
                "max_estimated_cost": "0.1",
                "cost_currency": "USD",
                "max_total_tokens": profile["max_total_tokens"],
                "timeout_seconds": profile["timeout_seconds"],
            },
        }
    ]
    assert (
        "AI judges run under their own judge profiles, not --max-cost: quality-judge "
        "(1 evaluation), each with a cost threshold of 0.1 USD (one model call can exceed it "
        f"before its cost is known), {profile['max_total_tokens']} tokens and "
        f"{profile['timeout_seconds']}s."
    ) in progress
    assert "across all trials" not in progress


def test_admission_refusals_the_cli_tells_apart_are_the_agents_own() -> None:
    def literals(module: ModuleType) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    admission = literals(_eval_run_admission)
    assert cloud_evals._PROFILE_REFUSALS.issubset(admission)
    assert cloud_evals._IDEMPOTENCY_CONFLICT in admission
    assert cloud_evals._CORPUS_NOT_FOUND in literals(_corpus_routes)


def test_run_statuses_and_page_bound_match_the_eval_store() -> None:
    statuses = tuple(status.value for status in EvalRunStatus)
    terminal = {status.value for status in TERMINAL_EVAL_RUN_STATUSES}

    assert statuses == cloud_evals.RUN_STATUSES
    assert terminal == cloud_evals.TERMINAL_RUN_STATUSES
    assert EVAL_STORE_MAX_PAGE_SIZE == cloud_evals.MAX_LIST_LIMIT

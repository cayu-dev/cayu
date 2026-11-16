"""`cayu cloud eval run` runs isolated by default: Cayu Cloud's evaluation service runs the
corpus suite against a Release in allocations of its own, and the live Agent is never asked
anything. `runs list|show|cancel` read those evaluations by id."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
from tests.evals.test_corpus_execution import _corpus

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.evals.corpus import eval_corpus_to_json

_AGENT = "month-end-agent"
_RELEASE = "dep_selected"
_FAST = ("--poll-seconds", "0.01", "--wait-seconds", "5")

_RECEIPT = {
    "record_type": "cayu-cloud.evaluation-receipt",
    "status": "completed",
    "case_facts": [
        {"case_id": "refund-approval", "outcome": "pass"},
        {"case_id": "refund-denial", "outcome": "unstable"},
    ],
    "metrics": {"accuracy": 0.5, "case_count": 2},
    "fingerprint": "3f9a",
}


class _EvaluationsCloud:
    """Cayu Cloud's evaluation service, as the CLI calls it; it never serves the Agent."""

    def __init__(self, *, settle: str = "completed", polls: int = 2) -> None:
        self.requests: list[str] = []
        self.submitted: list[dict[str, Any]] = []
        self.settle = settle
        self.polls = polls
        self.evaluations: dict[str, dict[str, Any]] = {}
        self.keys: dict[str, str] = {}
        self.catalog_queries: list[dict[str, str]] = []
        self.catalog: dict[str, Any] = {
            "release_id": _RELEASE,
            "version": "2.1",
            "selected": True,
            "eval_targets": ["evals.agent:build_eval"],
            "model_access_problem": None,
            "execution": {"available": True, "reason": None},
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        idempotency_key: str | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.requests.append(f"{method} {path}")
        base = f"/v1/applications/{_AGENT}"
        if path == "/v1/applications":
            return {"items": [{"id": _AGENT, "name": "Month end"}]}
        if path == base and method == "GET":
            return {"id": _AGENT, "current_deployment_id": _RELEASE}
        if path == f"{base}/evaluations" and method == "POST":
            assert payload is not None
            self.submitted.append(payload)
            key = str(payload["idempotency_key"])
            if key not in self.keys:
                evaluation_id = f"evl_{len(self.keys) + 1}"
                self.keys[key] = evaluation_id
                self.evaluations[evaluation_id] = {
                    "id": evaluation_id,
                    "status": "queued",
                    "release_id": payload["release_id"],
                    "suite": {"kind": "corpus", "suite_id": payload["suite"]["suite_id"]},
                    "created_at": "2026-11-16T10:00:00+00:00",
                    "failure": None,
                    "environment": {"database": "postgres", "isolation": "trial", "databases": []},
                    "_polls": 0,
                }
            return self._public(self.keys[key])
        if path == f"{base}/evaluation-catalog" and method == "GET":
            self.catalog_queries.append(dict(query or {}))
            return self.catalog
        if path == f"{base}/evaluations" and method == "GET":
            return {
                "items": [self._public(item) for item in reversed(self.evaluations)],
                "next_cursor": None,
            }
        if path.startswith(f"{base}/evaluations/evl_"):
            evaluation_id, _, action = path.removeprefix(f"{base}/evaluations/").partition("/")
            evaluation = self.evaluations[evaluation_id]
            if action == "receipt":
                assert evaluation["status"] in {"completed", "failed", "cancelled"}
                return _RECEIPT
            if action == "cancel":
                evaluation["status"] = "cancelling"
                return self._public(evaluation_id)
            evaluation["_polls"] += 1
            if evaluation["_polls"] >= self.polls and evaluation["status"] in {"queued", "running"}:
                evaluation["status"] = self.settle
                evaluation["environment"]["databases"] = [{"name": "db", "state": "dropped"}]
                if self.settle == "failed":
                    evaluation["failure"] = {
                        "code": "seed_failed",
                        "message": "The Release's seed evals.seed:load failed (exit code 1).",
                    }
            elif evaluation["status"] == "queued":
                evaluation["status"] = "running"
            return self._public(evaluation_id)
        raise AssertionError(f"unexpected Cloud request {method} {path}")

    def _public(self, evaluation_id: str) -> dict[str, Any]:
        return {
            key: value for key, value in self.evaluations[evaluation_id].items() if key[0] != "_"
        }


@pytest.fixture
def cloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _EvaluationsCloud:
    fake = _EvaluationsCloud()
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: fake)
    return fake


def _corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "evals.json"
    path.write_text(eval_corpus_to_json(_corpus(trials=2)), encoding="utf-8")
    return path


def _eval(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict, str]:
    code = main(["cloud", "eval", *argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def _run(tmp_path: Path, *extra: str) -> tuple[str, ...]:
    return (
        *("run", "--application", _AGENT, "--corpus", str(_corpus_file(tmp_path))),
        *("--suite", "refund-regressions", *_FAST, *extra),
    )


def test_a_run_is_isolated_by_default_and_never_asks_the_live_agent(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "receipt.json"
    code, answer, progress = _eval(capsys, *_run(tmp_path, "--output", str(receipt)))

    assert code == 0, answer
    result = answer["result"]
    assert (answer["operation"], result["on"]) == ("eval.run", "isolated")
    assert result["run"]["status"] == "completed"
    assert result["summary"] == {
        "status": "completed",
        "accuracy": 0.5,
        "cases": {"pass": 1, "unstable": 1},
        "fingerprint": "3f9a",
    }
    assert json.loads(receipt.read_text()) == _RECEIPT
    assert result["output"] == str(receipt)
    (body,) = cloud.submitted
    # The selected Release, the corpus inline and the one suite, under the printed key.
    assert body["release_id"] == _RELEASE
    assert body["suite"]["suite_id"] == "refund-regressions"
    assert body["suite"]["corpus"] == json.loads(eval_corpus_to_json(_corpus(trials=2)))
    assert "eval_target" not in body["suite"]
    assert (body["trials"], body["max_concurrency"], body["purpose"]) == (1, 1, "development")
    assert "environment" not in body
    assert body["idempotency_key"] == result["operation_id"]
    assert result["bound"]["total_trials"] == 2
    # Only Cloud's evaluation routes: no Agent address, login or service.
    assert not [item for item in cloud.requests if "/service" in item or "operator" in item]
    assert "Isolated evaluation of Release dep_selected" in progress
    assert "nothing reaches the live Agent's stores" in progress
    assert "executes on the live Agent" not in progress


def test_a_runs_options_reach_its_request_and_its_continuation_repeats_it(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    options = (
        *("--release", "dep_candidate", "--target", "evals.agent:build_eval"),
        *("--trials", "3", "--max-concurrency", "4", "--isolation", "run"),
        *("--database", "postgres", "--max-parallel-trials", "2", "--no-wait"),
    )
    code, answer, progress = _eval(capsys, *_run(tmp_path, *options))

    assert code == 0, answer
    result = answer["result"]
    assert result["run"]["status"] == "queued" and result["summary"] is None
    # A run sharing one database runs its allocations one at a time, whatever it asks.
    assert "run at once" not in progress
    (body,) = cloud.submitted
    assert body["release_id"] == "dep_candidate"
    assert body["suite"]["eval_target"] == "evals.agent:build_eval"
    assert (body["trials"], body["max_concurrency"], body["max_parallel_trials"]) == (3, 4, 2)
    assert body["environment"] == {"database": "postgres", "isolation": "run"}
    # The continuation repeats the same request under the same key: nothing new starts.
    again = shlex.split(result["commands"]["continue"])
    assert again[:3] == ["cayu", "cloud", "eval"]
    for option in (
        *("--release", "--target", "--trials", "--isolation", "--database"),
        "--max-parallel-trials",
    ):
        assert option in again
    code, repeated, _ = _eval(capsys, *again[3:], *_FAST)
    assert code == 0, repeated
    assert repeated["result"]["run"]["id"] == result["run"]["id"]
    assert len(cloud.keys) == 1
    assert result["commands"]["show"].endswith(f"runs show --application {_AGENT} evl_1")


def test_each_place_refuses_the_options_only_the_other_takes(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    for extra, named in (
        (("--max-cost", "0.5"), "--max-cost"),
        (("--execution-profile-revision", "sha256:" + "a" * 64), "--execution-profile-revision"),
        (("--on", "live", "--release", "dep_x"), "--release"),
        (("--on", "live", "--trials", "2"), "--trials"),
        (("--on", "live", "--isolation", "run"), "--isolation"),
        (("--on", "live", "--max-parallel-trials", "2"), "--max-parallel-trials"),
        (("--max-parallel-trials", "11"), "--max-parallel-trials"),
    ):
        code, answer, _ = _eval(capsys, *_run(tmp_path, *extra))
        assert code == 2, answer
        assert answer["error"]["category"] == "invalid_input"
        assert named in answer["error"]["message"]
    assert cloud.requests == []


def test_a_failed_evaluation_fails_the_command_with_its_reason(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.settle = "failed"
    code, answer, _ = _eval(capsys, *_run(tmp_path))

    assert code != 0
    error = answer["error"]
    assert error["category"] == "eval_run_failed"
    assert "seed_failed" in error["message"]
    assert not [item for item in cloud.requests if item.endswith("/receipt")]


def test_runs_lists_and_shows_and_cancels_isolated_runs_by_id(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _eval(capsys, *_run(tmp_path))[0] == 0
    assert _eval(capsys, *_run(tmp_path, "--no-wait"))[0] == 0

    code, listed, _ = _eval(capsys, "runs", "list", "--application", _AGENT)
    assert code == 0, listed
    items = listed["result"]["items"]
    assert [(item["id"], item["on"], item["status"]) for item in items] == [
        ("evl_2", "isolated", "queued"),
        ("evl_1", "isolated", "completed"),
    ]
    code, shown, _ = _eval(capsys, "runs", "show", "--application", _AGENT, "evl_1")
    assert code == 0, shown
    assert shown["result"]["summary"]["cases"] == {"pass": 1, "unstable": 1}
    code, cancelled, _ = _eval(capsys, "runs", "cancel", "--application", _AGENT, "evl_2")
    assert code == 0, cancelled
    assert cancelled["result"]["run"]["status"] == "cancelling"
    assert f"POST /v1/applications/{_AGENT}/evaluations/evl_2/cancel" in cloud.requests
    # The live Agent's runs need --on live.
    code, refused, _ = _eval(
        capsys, "runs", "list", "--application", _AGENT, "--target-key", "refund-agent"
    )
    assert code == 2 and "--on live" in refused["error"]["message"]


def test_the_human_output_names_the_evaluation_its_cases_and_its_databases(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cloud", "eval", *_run(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Evaluation evl_1: completed" in out
    assert "Cases: 1 pass, 1 unstable (accuracy 0.5)" in out
    assert "Eval databases: 1 (dropped)" in out


def test_a_cloud_without_isolated_corpus_runs_says_to_run_live(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    from cayu.cli._cloud_api import CloudApiError

    original = cloud.request

    def refusing(method: str, path: str, **options: Any) -> dict[str, Any]:
        if method == "POST" and path.endswith("/evaluations"):
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 422.",
                status_code=422,
                code="suite_kind_unsupported",
            )
        return original(method, path, **options)

    monkeypatch.setattr(cloud, "request", refusing)
    code, answer, _ = _eval(capsys, *_run(tmp_path))
    assert code == 2, answer
    assert answer["error"]["category"] == "isolated_runs_unavailable"
    assert "--on live" in answer["error"]["message"]


def test_the_api_client_keeps_the_refusals_eval_run_explains() -> None:
    import httpx

    from cayu.cli._cloud_api import _rejection

    for status, code in (
        (422, "suite_kind_unsupported"),
        (503, "evaluations_unavailable"),
        (409, "database_unsupported"),
    ):
        response = httpx.Response(status, json={"detail": {"code": code, "message": "refused"}})
        assert _rejection(response).code == code


def test_a_read_cloud_cant_answer_for_a_while_is_read_again(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    from cayu.cli._cloud_api import CloudApiError

    original = cloud.request
    failures = iter([True, True])

    def flaky(method: str, path: str, **options: Any) -> dict[str, Any]:
        if method == "GET" and path.endswith("/evaluations/evl_1") and next(failures, False):
            raise CloudApiError(
                "api_unavailable", "Cayu Cloud API returned HTTP 502.", status_code=502
            )
        return original(method, path, **options)

    monkeypatch.setattr(cloud, "request", flaky)
    code, answer, progress = _eval(capsys, *_run(tmp_path))
    assert code == 0, answer
    assert answer["result"]["run"]["status"] == "completed"
    assert progress.count("goes on in Cayu Cloud; reading it again") == 1


def test_a_read_cloud_refuses_names_the_evaluation_that_goes_on(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    from cayu.cli._cloud_api import CloudApiError

    original = cloud.request

    def refusing(method: str, path: str, **options: Any) -> dict[str, Any]:
        if method == "GET" and path.endswith("/evaluations/evl_1"):
            raise CloudApiError(
                "api_request_rejected", "Cayu Cloud API returned HTTP 403.", status_code=403
            )
        return original(method, path, **options)

    monkeypatch.setattr(cloud, "request", refusing)
    code, answer, _ = _eval(capsys, *_run(tmp_path))
    assert code != 0
    assert "evl_1" in answer["error"]["message"]
    assert "runs show" in answer["error"]["message"]


def test_a_status_filter_reads_further_pages_until_it_has_enough(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    pages = {
        None: {"items": [{"id": "evl_3", "status": "completed"}], "next_cursor": "c1"},
        "c1": {"items": [{"id": "evl_2", "status": "completed"}], "next_cursor": "c2"},
        "c2": {"items": [{"id": "evl_1", "status": "failed"}], "next_cursor": None},
    }
    original = cloud.request
    queries: list[dict[str, str]] = []

    def paged(method: str, path: str, **options: Any) -> dict[str, Any]:
        if method == "GET" and path.endswith("/evaluations"):
            query = options.get("query") or {}
            queries.append(dict(query))
            return pages[query.get("cursor")]
        return original(method, path, **options)

    monkeypatch.setattr(cloud, "request", paged)
    code, listed, _ = _eval(
        capsys, "runs", "list", "--application", _AGENT, "--status", "failed", "--limit", "1"
    )
    assert code == 0, listed
    assert [item["id"] for item in listed["result"]["items"]] == ["evl_1"]
    assert listed["result"]["has_more"] is False
    assert [query.get("cursor") for query in queries] == [None, "c1", "c2"]


def test_a_run_without_parallel_trials_sends_none_and_says_so_when_asked(
    tmp_path: Path, cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _eval(capsys, *_run(tmp_path, "--no-wait"))[0] == 0
    assert "max_parallel_trials" not in cloud.submitted[-1]
    code, _, progress = _eval(
        capsys, *_run(tmp_path, "--trials", "4", "--max-parallel-trials", "3", "--no-wait")
    )
    assert code == 0
    assert "Up to 3 allocations run at once" in progress


_ENVIRONMENT = {
    "default_eval_target": "evals.agent:build_eval",
    "seed": {"target": "evals.seed:load_demo_book", "timeout_seconds": 120},
    "seed_problem": None,
    "fixtures": ["demo-data/"],
    "database": "postgres",
    "databases_available": True,
    "eval_only_variables": [
        {"name": "BROKER_KEY", "secret": True},
        {"name": "STUB_URL", "secret": False},
    ],
    "brokered_credentials": ["OPENAI_API_KEY"],
    "limits": {"max_trials": 10, "max_parallel_trials": 4, "max_concurrency": 64},
}


def test_environment_show_says_what_an_isolated_run_would_get(
    cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud.catalog["environment"] = _ENVIRONMENT
    code, answer, _ = _eval(
        capsys, "environment", "show", "--application", _AGENT, "--release", "dep_candidate"
    )

    assert code == 0, answer
    assert answer["operation"] == "eval.environment.show"
    assert answer["result"]["environment"] == _ENVIRONMENT
    assert cloud.catalog_queries == [{"release_id": "dep_candidate"}]
    # Text names the variables, never a value, and says what each trial gets.
    assert main(["cloud", "eval", "environment", "show", "--application", _AGENT]) == 0
    text = capsys.readouterr().out
    assert cloud.catalog_queries[-1] == {}
    assert "Seed: evals.seed:load_demo_book, up to 120s" in text
    assert "Eval-only variables: BROKER_KEY (secret), STUB_URL" in text
    assert "Model keys brokered to trials: OPENAI_API_KEY" in text
    assert "its own PostgreSQL database for each allocation" in text
    assert not [item for item in cloud.requests if "/service" in item]


def test_environment_show_on_a_cloud_that_doesnt_describe_it_yet(
    cloud: _EvaluationsCloud, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cloud", "eval", "environment", "show", "--application", _AGENT]) == 0
    text = capsys.readouterr().out
    assert "doesn't describe its eval environments yet" in text
    assert "evals.agent:build_eval" in text

"""`cayu eval calibrate` runs the calibration workbench's server contract from the terminal.

The tests serve the real eval routes, a durable eval store and a scripted judge, save a suite
with a structured AI judge check, and calibrate it end to end: against a running server
(`--server-url`) and against a Cloud Agent (`--application`) reached through a fake Cayu
Cloud, as `cayu cloud eval` is.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from tests.cli.test_cloud_evals import _AGENT, _PASSWORD
from tests.cli.test_eval_curation_cli import _LOCAL, _command, _ok, _refused, _served
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.evals.test_structured_model_judge import _corpus as _judged_corpus
from tests.evals.test_structured_model_judge import _rubric
from tests.evals.test_structured_model_judge import _target as _judged_target
from tests.server.test_server_eval_judge_calibration import _judge_with_trials

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi, basic_authorization
from cayu.cli._eval_calibration import _RUN_ID_CONFLICT
from cayu.evals import (
    CorpusUserMessageSpec,
    EvalCaseDraftV2,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV2,
    RunInputSpec,
    StructuredModelJudgeAssertionDraftV1,
    StructuredRubricDraftV1,
    eval_corpus_to_json,
    model_judge_profile,
)
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvalsConfig,
    ServerConfig,
    _judge_calibration_routes,
    create_server,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.vaults.redaction import SecretRedactor

_BEARER = "Bearer calibration-cli-fixture-4d2e9a71c3b8"
_BASIC = basic_authorization("operator", _PASSWORD)
_CASE = "refund-answer"
_CHECK = "answer-quality"
_SECRET = "calibration-secret-5c1f"


def _authenticate(request: Request) -> AuthContext:
    """The operator by a bearer token locally, or by the login Cloud hands the CLI."""

    if request.headers.get("Authorization") in {_BEARER, _BASIC}:
        return AuthContext(subject="eval-operator")
    raise HTTPException(status_code=401, detail="unauthorized")


@contextmanager
def _judged_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, judge_calls: int
) -> Iterator[tuple[list, str, str]]:
    """A server with two suites that have the same structured judge check, an authored suite
    and a corpus suite, and an application that won't publish ``_SECRET``. Yields the judge's
    requests and the two suites' `suites list` ids."""

    judge, judge_provider = _judge_with_trials(judge_calls)
    target, candidate_provider = _judged_target(judge, secret_redactor=SecretRedactor(_SECRET))
    store = SQLiteEvalStore(tmp_path / "evals.db")
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(target=target, store=store),
        ),
    )
    profile = model_judge_profile(judge)
    draft = EvalSuiteDraftV2(
        id="refund-quality",
        target_key=target.key,
        name="Refund quality",
        cases=(
            EvalCaseDraftV2(
                id=_CASE,
                name="Refund answer",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(
                    StructuredModelJudgeAssertionDraftV1(
                        id=_CHECK,
                        judge_profile_key=profile.key,
                        judge_profile_revision=profile.revision,
                        rubric=StructuredRubricDraftV1.from_rubric(_rubric()),
                        threshold="0.6",
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _BEARER)
    try:
        with _served(server, monkeypatch) as client:
            headers = {"Authorization": _BEARER}
            preview = client.post(
                "/api/evals/suites/preview",
                headers=headers,
                json={"draft": draft.model_dump(mode="json")},
            ).json()
            saved = client.post(
                "/api/evals/suites",
                headers=headers,
                json={
                    "expected_suite_revision": preview["suite"]["revision"],
                    "suite": preview["suite"],
                },
            )
            assert saved.status_code == 201, saved.text
            corpus = _judged_corpus(judge)
            imported = client.post(
                "/api/evals/corpora", headers=headers, content=eval_corpus_to_json(corpus)
            )
            assert imported.status_code == 201, imported.text
            suites = client.get("/api/evals/suite-views", headers=headers).json()["items"]
            ids = {item["source"]: item["id"] for item in suites}
            yield judge_provider.requests, ids["authored"], ids["corpus"]
        # Calibration calls only the judge.
        assert candidate_provider.requests == []
    finally:
        asyncio.run(store.close())


def _files(tmp_path: Path, *, final_output: str, **evidence: object) -> tuple[str, ...]:
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text(json.dumps({"final_output": final_output, **evidence}))
    labels_file = tmp_path / "labels.json"
    # Scores are decimal numbers, written in any notation and in any order.
    labels_file.write_text('{"usefulness": 0.50, "correctness": 1.0}')
    return ("--evidence", str(evidence_file), "--labels", str(labels_file))


def test_calibrate_previews_then_runs_the_reviewed_definition_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with _judged_server(tmp_path, monkeypatch, judge_calls=2) as (judge_requests, suite, _):
        files = _files(tmp_path, final_output="Refunds are available within 30 days.")
        check = ("calibrate", "--suite", suite, "--case", _CASE, "--assertion", _CHECK, *files)
        previewed = _ok(capsys, *check, "--trials", "2", "--preview", *_LOCAL)
        # What the files leave out is what the workbench proposes.
        draft = previewed["draft"]
        assert (draft["id"], draft["evidence_source_id"], draft["task"]) == (
            "answer-quality-calibration",
            "answer-quality-known-evidence",
            "Refund answer",
        )
        assert draft["human_criteria"] == [
            {"criterion_id": "correctness", "score": "1"},
            {"criterion_id": "usefulness", "score": "0.5"},
        ]
        assert previewed["preview"]["ready"] is True
        assert previewed["preview"]["work"]["judge_calls"] == 2
        assert previewed["report"] is None
        assert judge_requests == []

        run = _command(previewed["commands"]["run"])
        assert run[-4:] == (
            "--run-id",
            previewed["run_id"],
            "--expected-definition-revision",
            previewed["preview"]["definition"]["revision"],
        )
        calibrated = _ok(capsys, *run)
        report = calibrated["report"]
        assert report["run_id"] == previewed["run_id"]
        assert report["definition"] == previewed["preview"]["definition"]
        assert report["definition"]["human_label"]["aggregate_score"] == "0.625"
        assert [trial["pass_agreement"] for trial in report["trials"]] == [True, True]
        assert len(judge_requests) == 2

        # The same run command returns the stored report and calls no judge.
        assert _ok(capsys, *run)["report"] == report
        assert len(judge_requests) == 2
        assert main(["eval", *run, "--table"]) == 0
        text = capsys.readouterr().out
        assert "Calibrated check answer-quality of case refund-answer" in text
        assert "Trial 1: passed, score 0.625, absolute error 0, pass agreement yes" in text

        # A check that can't run now still answers its run ID with the stored report; a new
        # run ID is refused.
        with monkeypatch.context() as unready:
            unready.setattr(
                _judge_calibration_routes,
                "_candidate_judge_route_relation",
                lambda *_: "same_model",
            )
            named = (*check, "--trials", "2", "--run-id")
            replayed = _ok(capsys, *named, report["run_id"], *_LOCAL)
            assert replayed["preview"]["ready"] is False
            assert replayed["report"] == report
            error = _refused(capsys, *named, "another-run", *_LOCAL)
            assert error["category"] == "calibration_not_ready"
            assert [item["code"] for item in error["diagnostics"]] == ["same_model_forbidden"]
        assert len(judge_requests) == 2

        # Changed evidence no longer matches the previewed definition.
        _files(tmp_path, final_output="Refunds are never available.")
        error = _refused(capsys, *run)
        assert error["category"] == "calibration_changed"
        assert error["expected_definition_revision"] == report["definition"]["revision"]
        # And the run ID can't calibrate it either.
        error = _refused(capsys, *check, "--trials", "2", "--run-id", report["run_id"], *_LOCAL)
        assert (error["category"], error["run_id"]) == ("run_id_conflict", report["run_id"])
        assert len(judge_requests) == 2


def test_a_lost_calibration_answer_is_continued_without_calling_the_judge_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    send = CayuApi._bounded_request

    def answer_lost(self: CayuApi, method: str, url: str, *args, **kwargs):
        response = send(self, method, url, *args, **kwargs)
        lost = method == "POST" and url.endswith("/judge-calibrations")
        return httpx.Response(502) if lost else response

    with _judged_server(tmp_path, monkeypatch, judge_calls=1) as (judge_requests, suite, _):
        files = _files(tmp_path, final_output="Refunds are available within 30 days.")
        check = ("calibrate", "--suite", suite, "--case", _CASE, "--assertion", _CHECK, *files)
        with monkeypatch.context() as lossy:
            lossy.setattr(CayuApi, "_bounded_request", answer_lost)
            error = _refused(capsys, *check, "--trials", "1", *_LOCAL)
        assert error["category"] == "outcome_unknown"
        assert len(judge_requests) == 1
        run = _command(error["commands"]["run"])
        assert run[run.index("--run-id") + 1] == error["run_id"]

        calibrated = _ok(capsys, *run)
        assert calibrated["report"]["run_id"] == error["run_id"]
        assert len(calibrated["report"]["trials"]) == 1
        assert len(judge_requests) == 1


def test_calibrate_reaches_a_cloud_agent_with_the_cli_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud = _FakeCloud()
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    on_cloud = ("--application", _AGENT)

    with _judged_server(tmp_path, monkeypatch, judge_calls=1) as (judge_requests, suite, _):
        files = _files(tmp_path, final_output="Refunds are available within 30 days.")
        check = ("calibrate", "--suite", suite, "--case", _CASE, "--assertion", _CHECK, *files)
        previewed = _ok(capsys, *check, "--trials", "1", "--preview", *on_cloud)
        assert previewed["application_id"] == _AGENT
        run = _command(previewed["commands"]["run"])
        assert run[-6:-4] == on_cloud
        calibrated = _ok(capsys, *run)
        assert calibrated["report"]["run_id"] == previewed["run_id"]
        # The Agent's report is the server's: one stored report for the run ID.
        assert (
            _ok(capsys, *check, "--trials", "1", "--run-id", previewed["run_id"], *_LOCAL)["report"]
            == (calibrated["report"])
        )
        assert len(judge_requests) == 1
    assert "GET /v1/applications/bug-hunter-agent/operator-credentials" in cloud.requests


def test_calibration_inputs_are_checked_against_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with _judged_server(tmp_path, monkeypatch, judge_calls=0) as (judge_requests, suite, _):
        files = _files(tmp_path, final_output="Refunds are available within 30 days.")
        check = ("calibrate", "--suite", suite, "--case", _CASE, *files, *_LOCAL)
        for argv, category, message in (
            (
                ("--assertion", "missing-check"),
                "not_found",
                f"Case {_CASE} has no check missing-check.",
            ),
            (
                ("--assertion", _CHECK, "--case", "missing-case"),
                "not_found",
                f"Suite {suite} has no case missing-case.",
            ),
        ):
            error = _refused(capsys, *check, *argv)
            assert (error["category"], error["message"]) == (category, message)

        (tmp_path / "labels.json").write_text('{"correctness": 1, "tone": 0.5}')
        error = _refused(capsys, *check, "--assertion", _CHECK)
        assert (error["category"], error["message"]) == (
            "invalid_input",
            f"--labels {tmp_path / 'labels.json'} must score exactly the rubric's criteria: "
            "correctness, usefulness. Missing: usefulness. Not in the rubric: tone.",
        )
        (tmp_path / "labels.json").write_text('{"correctness": 1, "usefulness": 0.5}')
        _files(tmp_path, final_output="An answer.", transcript="user: refund?")
        error = _refused(capsys, *check, "--assertion", _CHECK)
        assert error["message"] == (
            f"Check {_CHECK} judges only the final output; remove the transcript from "
            f"--evidence {tmp_path / 'evidence.json'}."
        )
        assert judge_requests == []


def test_calibration_options_and_files_are_checked_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text('{"correctness": 1}')
    evidence = tmp_path / "evidence.json"
    check = ("calibrate", "--suite", "s", "--case", "c", "--assertion", "a", "--labels")
    revision = "sha256:" + "a" * 64
    for content, extra, message in (
        (
            '{"final_output": "x"}',
            ("--preview", "--expected-definition-revision", revision),
            "--preview previews a new calibration; leave out the expected revision.",
        ),
        (None, (), f"Could not read --evidence {evidence}: No such file or directory."),
        ("[]", (), f"--evidence {evidence} must hold one JSON object."),
        (
            '{"answer": "x"}',
            (),
            f"--evidence {evidence} has unknown fields: answer. It takes final_output, "
            "source_id, task, transcript.",
        ),
        ('{"task": "x"}', (), f"--evidence {evidence} needs final_output, the answer to judge."),
        ('{"final_output": 1}', (), f"--evidence {evidence}: final_output must be a string."),
    ):
        if content is None:
            evidence.unlink(missing_ok=True)
        else:
            evidence.write_text(content)
        error = _refused(capsys, *check, str(labels), "--evidence", str(evidence), *extra)
        assert (error["category"], error["message"]) == ("invalid_input", message)

    evidence.write_text('{"final_output": "x"}')
    for score in ("true", "1.5", '"-0.1"', '"half"', "NaN"):
        labels.write_text(f'{{"correctness": {score}}}')
        error = _refused(capsys, *check, str(labels), "--evidence", str(evidence))
        assert error["message"] == (
            f"--labels {labels}: the score of correctness must be a number from 0 to 1."
        )
    assert main(["eval", *check, str(labels), "--evidence", str(evidence), "-o", str(labels)]) == 1
    assert "output must not refer to an input file" in capsys.readouterr().err


def test_the_run_id_conflict_the_cli_tells_apart_is_the_servers_own() -> None:
    tree = ast.parse(inspect.getsource(_judge_calibration_routes))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert _RUN_ID_CONFLICT in literals


def test_calibrate_reads_a_corpus_suites_check_and_reports_why_it_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with _judged_server(tmp_path, monkeypatch, judge_calls=0) as (judge_requests, _, suite):
        check = ("calibrate", "--suite", suite, "--case", "answer-case", "--assertion", _CHECK)
        # Evidence the application won't publish.
        files = _files(tmp_path, final_output=f"Refunds need {_SECRET}.")
        previewed = _ok(capsys, *check, *files, "--preview", *_LOCAL)
        # The corpus case's check, with the case's name as its task.
        assert previewed["draft"]["task"] == "Answer case"
        assert previewed["draft"]["assertion"]["rubric"]["id"] == _CHECK
        assert previewed["preview"]["ready"] is False
        [diagnostic] = previewed["preview"]["diagnostics"]
        assert diagnostic["code"] == "unsafe_public_material"
        assert previewed["commands"] == {}
        assert main(["eval", *check, *files, "--preview", *_LOCAL, "--table"]) == 0
        text = capsys.readouterr().out
        assert text.startswith("Calibration of check answer-quality of case answer-case needs")
        assert diagnostic["message"] in text

        error = _refused(capsys, *check, *files, *_LOCAL)
        assert error["category"] == "calibration_not_ready"
        assert error["diagnostics"] == [diagnostic]
        assert judge_requests == []

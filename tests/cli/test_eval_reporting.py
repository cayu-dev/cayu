from __future__ import annotations

import json
import os

import httpx
import pytest

from cayu.cli import main

_TARGET = "tests.cli.test_evals_cli:build_slow_eval_plan"
_TOKEN = "report-token-cli-private-canary-000000000001"


def test_cli_saves_exact_report_and_republish_never_loads_target(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/reviewer")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/ci")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "event.json"))
    (tmp_path / "event.json").write_text(
        json.dumps({"number": 42, "pull_request": {"head": {"sha": "b" * 40}}})
    )
    sent = []

    def upload(request):
        assert request.url == "https://dashboard.example/api/evals/reports"
        sent.append(request.content)
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        raise httpx.ReadTimeout("private transport details must not leak", request=request)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    result = tmp_path / "result.json"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--output",
                str(result),
                "--report-to",
                "https://dashboard.example/api",
                "--report-target",
                "reviewer",
                "--report-file",
                str(report),
            ]
        )
        == 2
    )
    assert json.loads(result.read_text())["status"] == "passed"
    document = json.loads(report.read_text())
    evidence = document["report"]
    assert evidence["declared_suite_revision"] is None
    assert evidence["origin"]["commit"] == "a" * 40
    assert evidence["origin"]["pr_head_commit"] == "b" * 40
    assert evidence["origin"]["pull_request"] == 42
    assert _TOKEN not in report.read_text()
    captured = capsys.readouterr()
    assert "private transport details" not in captured.out + captured.err
    frozen = report.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Exact report retry must not resolve or execute a Python target")

    monkeypatch.setattr("cayu.cli.evals._load_eval_plan", forbidden)
    assert (
        main(["eval", "publish", str(report), "--report-to", "https://dashboard.example/api"]) == 2
    )
    assert report.read_bytes() == frozen
    assert len(sent) == 2 and sent[0] == sent[1]


@pytest.mark.parametrize(
    "destination", ["http://dashboard.example/api", "http://localhost:8000/api"]
)
def test_cli_rejects_untrusted_destination_before_execution(tmp_path, monkeypatch, destination):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid reporting destination must reject before evaluation")

    monkeypatch.setattr("cayu.cli.evals._load_eval_plan", forbidden)
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--report-to",
                destination,
                "--report-target",
                "reviewer",
                "--report-file",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "report.json").exists()


def test_cli_rejects_campaign_reporting_before_package_resolution(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting a campaign must reject before launching work")

    monkeypatch.setattr("cayu.cli._benchmark_campaigns.run_benchmark", forbidden)
    assert (
        main(
            [
                "eval",
                "run",
                "--package",
                "missing.json",
                "--report-to",
                "https://dashboard.example/api",
                "--report-target",
                "reviewer",
                "--report-file",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )


@pytest.mark.process
@pytest.mark.skipif(os.name != "posix", reason="POSIX process backend")
@pytest.mark.parametrize("metadata_safe", [True, False])
def test_cli_process_reporting_keeps_worker_redaction_and_original_result(
    tmp_path, monkeypatch, capsys, metadata_safe
):
    from tests.cli.test_process_execution import _project

    _project(
        tmp_path,
        extra="""
from cayu.vaults.redaction import SecretRedactor

def build_report_eval():
    plan = build_eval()
    plan.app.redact_json = SecretRedactor("done").redact_json
    return plan
""",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)
    if not metadata_safe:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_HEAD_REF", "done")
    sent = []

    def upload(request):
        from cayu.evals._report_publication import prepare_report_record
        from cayu.evals.reported import EvalReport

        report = EvalReport.model_validate_json(request.content)
        sent.append(report)
        record = prepare_report_record(
            report, "ci-process-publisher", redact_json=lambda value: value
        )
        return httpx.Response(200, json=record.receipt.model_dump(mode="json"))

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    result = tmp_path / "result.json"
    report = tmp_path / "report.json"
    exit_code = main(
        [
            "eval",
            "run",
            "process_project:build_report_eval",
            "--processes",
            "2",
            "--max-concurrency",
            "2",
            "--process-directory",
            str(tmp_path / "workers"),
            "--output",
            str(result),
            "--report-to",
            "https://dashboard.example/api",
            "--report-target",
            "reviewer",
            "--report-file",
            str(report),
        ]
    )
    assert exit_code == (0 if metadata_safe else 2)
    assert json.loads(result.read_text())["status"] == "passed"
    launch = json.loads((tmp_path / "workers/launch.json").read_text())
    assert "reporting_context" not in launch and len(launch["reporting_context_hash"]) == 64
    if not metadata_safe:
        assert sent == [] and not report.exists()
        return
    assert len(sent) == 1 and sent[0].evidence.run.cases[0].status == "passed"
    assert sent[0].evidence.run.cases[0].trials[0].final_output == "[REDACTED_SECRET]"
    assert report.is_file() and _TOKEN not in report.read_text()
    assert "Eval report published" in capsys.readouterr().err


def test_cli_without_reporting_keeps_output_and_has_no_network_or_saved_envelope(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary eval must not start an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    result = tmp_path / "result.json"
    assert main(["eval", "run", _TARGET, "--output", str(result)]) == 0
    assert json.loads(result.read_text())["status"] == "passed"
    assert not (tmp_path / ".cayu/evals/reports").exists()


@pytest.mark.parametrize(
    "mode,expected_exit",
    [("corpus", 0), ("corpus_failed", 1), ("workflow", 0), ("workflow_error", 2)],
)
def test_cli_reports_ordinary_corpus_and_workflow_terminal_results(
    tmp_path, monkeypatch, mode, expected_exit
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)
    sent = []

    def upload(request):
        from cayu.evals._report_publication import prepare_report_record
        from cayu.evals.reported import EvalReport

        report = EvalReport.model_validate_json(request.content)
        sent.append(report)
        return httpx.Response(
            200,
            json=prepare_report_record(
                report, "ci-publisher", redact_json=lambda value: value
            ).receipt.model_dump(mode="json"),
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    extra = []
    if mode.startswith("corpus"):
        from tests.cli.test_evals_corpus_cli import _corpus

        corpus = _corpus()
        source = tmp_path / "corpus.json"
        source.write_text(corpus.model_dump_json())
        target = "tests.cli.test_evals_corpus_cli:" + (
            "build_failing_corpus_eval_plan"
            if mode == "corpus_failed"
            else "build_corpus_eval_plan"
        )
        target_key = corpus.target_key
        extra = ["--corpus", str(source), "--report-associate-corpus"]
    else:
        target = (
            "tests.evals.test_workflow_eval_target:build_cli_workflow_eval_plan"
            if mode == "workflow"
            else "tests.evals.test_workflow_failure_projection:build_failure_cli_plan"
        )
        target_key = "workflow-target"
    result_file = tmp_path / "result.json"
    assert (
        main(
            [
                "eval",
                "run",
                target,
                "--output",
                str(result_file),
                "--report-to",
                "https://dashboard.example/api",
                "--report-target",
                target_key,
                "--report-file",
                str(tmp_path / "report.json"),
                *extra,
            ]
        )
        == expected_exit
    )
    assert len(sent) == 1
    result = json.loads(result_file.read_text())
    if mode.startswith("corpus"):
        assert sent[0].evidence.kind == "corpus"
        assert sent[0].evidence.result.model_dump(mode="json") == result
        assert sent[0].association.corpus_revision == corpus.revision
    else:
        assert sent[0].evidence.kind == "direct"
        assert sent[0].evidence.run.status == result["status"]
        assert sent[0].evidence.run.cases[0].trials[0].structured_output == result["cases"][0][
            "trials"
        ][0].get("structured_output")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--report-program", " "),
        ("--report-suite-revision", "main"),
        ("--report-pull-request", "0"),
    ],
)
def test_cli_rejects_invalid_report_identity_before_execution(tmp_path, monkeypatch, flag, value):
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)
    monkeypatch.setattr(
        "cayu.cli.evals._load_eval_plan",
        lambda *_: pytest.fail("Invalid reporting context executed a target"),
    )
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--report-to",
                "https://dashboard.example/api",
                "--report-target",
                "reviewer",
                "--report-file",
                str(tmp_path / "report.json"),
                flag,
                value,
            ]
        )
        == 2
    )
    assert not (tmp_path / "report.json").exists()


def test_cli_report_file_publication_failure_keeps_result_and_never_posts(tmp_path, monkeypatch):
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)

    def fail_link(*_):
        raise OSError("report destination failed")

    monkeypatch.setattr(os, "link", fail_link)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_: pytest.fail("Failed local publication dispatched HTTP")
    )
    result = tmp_path / "result.json"
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--output",
                str(result),
                "--report-to",
                "https://dashboard.example/api",
                "--report-target",
                "reviewer",
                "--report-file",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    assert json.loads(result.read_text())["status"] == "passed"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("field", ["publisher_id", "run_key", "id", None])
def test_cli_publish_screens_success_receipt_and_checks_derived_identity(
    tmp_path, monkeypatch, capsys, caplog, field
):
    import asyncio
    import warnings

    from tests.server.test_server_eval_reporting import _payload

    from cayu.cli._eval_reporting import SavedEvalReport
    from cayu.evals._report_publication import prepare_report_record
    from cayu.evals.reported import EvalReport

    report = EvalReport.model_validate(asyncio.run(_payload("reviewer")))
    saved = SavedEvalReport(destination="https://dashboard.example/api", report=report)
    path = tmp_path / "saved.json"
    path.write_text(saved.model_dump_json())
    frozen = path.read_bytes()
    monkeypatch.setenv("CAYU_EVAL_REPORT_TOKEN", _TOKEN)

    def forbidden(*args, **kwargs):
        raise AssertionError("A saved report retry cannot load or execute its workload")

    monkeypatch.setattr("cayu.cli.evals._load_eval_plan", forbidden)

    def upload(request):
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        receipt = prepare_report_record(
            report,
            _TOKEN if field == "publisher_id" else "normal-publisher",
            redact_json=lambda value: value,
        ).receipt.model_dump(mode="json")
        if field:
            receipt[field] = "f" * 64 if field == "id" else _TOKEN
        return httpx.Response(200, json=receipt)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code = main(["eval", "publish", str(path), "--report-to", "https://dashboard.example/api"])
    captured = capsys.readouterr()
    diagnostics = captured.out + captured.err + caplog.text + str([str(w.message) for w in caught])
    assert _TOKEN not in diagnostics
    assert code == (2 if field else 0)
    if field:
        assert not captured.out
    else:
        assert json.loads(captured.out)["publisher_id"] == "normal-publisher"
    assert path.read_bytes() == frozen


@pytest.mark.parametrize(
    "destination,expected,allow,accepted",
    [
        ("https://attacker.example/api", None, False, False),
        ("https://attacker.example/api", "https://dashboard.example/api", False, False),
        ("http://localhost:8000/api", "http://localhost:8000/api", False, False),
        ("https://dashboard.example/api", "https://dashboard.example/api/", False, True),
        ("http://localhost:8000/api", "http://localhost:8000/api", True, True),
    ],
)
def test_cli_publish_requires_independent_destination_before_credential_access(
    tmp_path, monkeypatch, destination, expected, allow, accepted
):
    import asyncio

    from tests.server.test_server_eval_reporting import _payload

    from cayu.cli._eval_reporting import SavedEvalReport
    from cayu.evals._report_publication import prepare_report_record
    from cayu.evals.reported import EvalReport

    report = EvalReport.model_validate(asyncio.run(_payload("reviewer")))
    saved = SavedEvalReport(destination=destination, allow_loopback_http=True, report=report)
    path = tmp_path / "downloaded-report.json"
    path.write_text(saved.model_dump_json())
    original = path.read_bytes()
    credential_reads = []
    sent = []

    def credential(_):
        credential_reads.append(True)
        return _TOKEN

    def upload(request):
        sent.append(str(request.url))
        return httpx.Response(
            200,
            json=prepare_report_record(
                report, "trusted-publisher", redact_json=lambda value: value
            ).receipt.model_dump(mode="json"),
        )

    monkeypatch.setattr("cayu.cli._eval_reporting.report_token", credential)
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    argv = ["eval", "publish", str(path)]
    if expected is not None:
        argv += ["--report-to", expected]
    if allow:
        argv.append("--report-allow-loopback-http")
    code = main(argv)
    assert credential_reads == ([True] if accepted else [])
    assert sent == ([destination + "/evals/reports"] if accepted else [])
    assert code == (0 if accepted else 2)
    assert path.read_bytes() == original


@pytest.mark.parametrize("explicit_file", [False, True])
def test_cli_reporting_paths_follow_discovered_project_root(tmp_path, monkeypatch, explicit_file):
    from tests.cli.test_process_execution import _project

    from cayu.evals._report_publication import prepare_report_record
    from cayu.evals.reported import EvalReport

    _project(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "report-token.txt").write_text(_TOKEN)
    monkeypatch.chdir(nested)

    def upload(request):
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        report = EvalReport.model_validate_json(request.content)
        return httpx.Response(
            200,
            json=prepare_report_record(
                report, "path-publisher", redact_json=lambda value: value
            ).receipt.model_dump(mode="json"),
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upload), **kwargs),
    )
    argv = [
        "eval",
        "run",
        "--output",
        "result.json",
        "--report-to",
        "https://dashboard.example/api",
        "--report-target",
        "reviewer",
        "--report-token-file",
        "report-token.txt",
    ]
    if explicit_file:
        argv += ["--report-file", "report.json"]
    assert main(argv) == 0
    assert (tmp_path / "result.json").is_file()
    assert (
        (tmp_path / "report.json").is_file()
        if explicit_file
        else list((tmp_path / ".cayu/evals/reports").glob("*.json"))
    )
    assert list(nested.iterdir()) == []


def test_publish_help_names_trusted_destination_and_next_step(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["eval", "publish", "--help"])
    assert stopped.value.code == 0
    text = capsys.readouterr().out
    assert "independently trusted" in text
    assert "Evals history" in text
    assert "--report-to" in text and "--report-allow-loopback-http" in text


def test_cli_saved_retry_rejects_false_pass_before_credential_or_transport(tmp_path, monkeypatch):
    import asyncio

    from tests.server.test_server_eval_reporting import _payload

    payload = asyncio.run(_payload("reviewer"))
    trial = payload["evidence"]["run"]["cases"][0]["trials"][0]
    trial["assertions"] = [{"name": "failed", "outcome": "failed", "score": 0.0}]
    path = tmp_path / "inconsistent-report.json"
    path.write_text(json.dumps({"destination": "https://dashboard.example/api", "report": payload}))
    original = path.read_bytes()
    credential_reads = []
    transport = []

    def credential(_):
        credential_reads.append(True)
        return _TOKEN

    def client(**kwargs):
        transport.append(True)
        raise RuntimeError("Transport must not receive inconsistent evidence")

    monkeypatch.setattr("cayu.cli._eval_reporting.report_token", credential)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    assert main(["eval", "publish", str(path), "--report-to", "https://dashboard.example/api"]) == 2
    assert credential_reads == [] and transport == []
    assert path.read_bytes() == original

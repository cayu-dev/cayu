from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_mcp_conformance")
client = _load("mcp_conformance_client")


def _check(status="SUCCESS", check_id="required"):
    return {"id": check_id, "name": "Check", "status": status}


@pytest.mark.parametrize(
    "checks", [None, {}, [], [None], [{}], [_check("INFO")], [_check(check_id="other")]]
)
def test_gate_requires_positive_evidence(checks):
    with pytest.raises(ValueError):
        runner.validate_checks(checks, frozenset({"required"}))


@pytest.mark.parametrize("status", ["FAILURE", "WARNING", "SKIPPED", "unknown"])
def test_gate_rejects_nonpassing_checks_even_with_required_success(status):
    with pytest.raises(ValueError):
        runner.validate_checks([_check(), _check(status)], frozenset({"required"}))


def test_gate_accepts_information_only_alongside_required_success():
    runner.validate_checks([_check(), _check("INFO", "server-info")], frozenset({"required"}))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "skipped", "failure", "error"])
def test_sdk_report_requires_all_tests_without_skips(tmp_path, mutation):
    names = [node.split("::")[1] for node in runner.SDK_TESTS]
    if mutation == "missing":
        names.pop()
    if mutation == "duplicate":
        names[-1] = names[0]
    child = f"<{mutation}/>" if mutation in {"skipped", "failure", "error"} else ""
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites><testsuite>"
        + "".join(f'<testcase name="{name}">{child}</testcase>' for name in names)
        + "</testsuite></testsuites>"
    )
    with pytest.raises(ValueError):
        runner.validate_sdk_report(path)


def test_sdk_report_accepts_complete_evidence(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites><testsuite>"
        + "".join(f'<testcase name="{node.split("::")[1]}"/>' for node in runner.SDK_TESTS)
        + "</testsuite></testsuites>"
    )
    runner.validate_sdk_report(path)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost",
        "http://example.com",
        "http://user:secret@localhost",
        "file:///tmp/server",
    ],
)
def test_adapter_rejects_nonlocal_peers(url):
    with pytest.raises(ValueError, match="loopback"):
        asyncio.run(client.run(url, "tools_call", "2026-07-28", {}))


@pytest.mark.parametrize(
    ("scenario", "version"),
    [("unknown", "2026-07-28"), ("tools_call", "auto"), ("initialize", "2026-07-28")],
)
def test_adapter_rejects_unimplemented_modes(scenario, version):
    with pytest.raises(ValueError, match="Unsupported"):
        asyncio.run(client.run("http://localhost", scenario, version, {}))


def test_adapter_closes_real_session_boundary_on_failure(monkeypatch):
    session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=ValueError("peer failure")), close=AsyncMock()
    )
    connect = AsyncMock(return_value=session)
    monkeypatch.setattr(client, "HttpMcpClient", lambda **kwargs: SimpleNamespace(connect=connect))
    with pytest.raises(ValueError, match="peer failure"):
        asyncio.run(client.run("http://localhost", "tools_call", "2026-07-28", {}))
    session.close.assert_awaited_once()


def test_adapter_echoes_the_client_schema_without_rewriting_it(monkeypatch):
    schema = {
        "$defs": {"value": {"type": "string"}},
        "properties": {"value": {"$ref": "#/$defs/value"}},
    }
    session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=[SimpleNamespace(name="json_schema_2020_12_tool", input_schema=schema)]
        ),
        call_tool=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        client,
        "HttpMcpClient",
        lambda **kwargs: SimpleNamespace(connect=AsyncMock(return_value=session)),
    )
    asyncio.run(
        client.run("http://localhost", "json-schema-2020-12-preservation", "2026-07-28", {})
    )
    session.call_tool.assert_awaited_once_with("json_schema_echo", {"schema": schema})
    assert session.call_tool.call_args.args[1]["schema"] is schema
    session.close.assert_awaited_once()


@pytest.mark.parametrize("content", [[], [{"type": "text", "text": "wrong result"}]])
def test_adapter_requires_the_actual_tool_result(monkeypatch, content):
    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=[]),
        call_tool=AsyncMock(return_value=SimpleNamespace(is_error=False, content=content)),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        client,
        "HttpMcpClient",
        lambda **kwargs: SimpleNamespace(connect=AsyncMock(return_value=session)),
    )
    with pytest.raises(AssertionError, match="preserve"):
        asyncio.run(client.run("http://localhost", "tools_call", "2026-07-28", {}))
    session.close.assert_awaited_once()


@pytest.mark.parametrize(
    "scenario", runner.SCENARIOS, ids=lambda scenario: f"{scenario.version}-{scenario.name}"
)
def test_covered_scenarios_have_nonempty_evidence_and_adapter_support(scenario):
    assert scenario.required_checks
    assert scenario.name in client.SCENARIOS
    assert scenario.name not in runner.BLOCKED_SCENARIOS


def test_docker_build_pins_the_same_official_revision():
    dockerfile = (ROOT / "scripts/mcp-conformance.Dockerfile").read_text()
    assert f"git fetch --depth 1 origin {runner.UPSTREAM_REVISION}" in dockerfile
    assert "npm ci --ignore-scripts" in dockerfile
    assert "uv sync --frozen" in dockerfile


@pytest.mark.parametrize("return_code", [0, 1])
def test_nonzero_exit_or_missing_artifacts_fail_with_report(tmp_path, monkeypatch, return_code):
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["run", "--upstream", str(tmp_path), "--output", str(output)])
    monkeypatch.setattr(runner, "verify_upstream", lambda path: None)
    monkeypatch.setattr(runner, "run_bounded", lambda *args, **kwargs: return_code)
    assert runner.main() == 1
    report = json.loads((output / "summary.json").read_text())
    assert report["full_conformance"] is False
    assert report["covered_subset_passed"] is False
    assert all(record["status"] == "failed" for record in report["results"])


def test_output_directory_cannot_reuse_stale_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run", "--upstream", str(tmp_path), "--output", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        runner.main()


@pytest.mark.parametrize(
    "mode", ["pass", "exit", "skip", "empty", "missing", "duplicate", "blocked"]
)
def test_complete_gate_keeps_official_and_sdk_evidence_separate(tmp_path, monkeypatch, mode):
    output = tmp_path / "evidence"
    args = ["run", "--upstream", str(tmp_path), "--output", str(output)]
    if mode == "blocked":
        args.append("--include-blocked")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "verify_upstream", lambda path: None)

    def execute(command, directory, **kwargs):
        if "--scenario" in command:
            name = command[command.index("--scenario") + 1]
            version = command[command.index("--spec-version") + 1]
            scenario = next(
                (s for s in runner.SCENARIOS if (s.name, s.version) == (name, version)), None
            )
            checks = (
                [_check(check_id=key) for key in scenario.required_checks]
                if scenario
                else [_check()]
            )
            if mode == "skip":
                checks.append(_check("SKIPPED"))
            elif mode == "empty":
                checks = []
            elif mode == "missing":
                checks.pop()
            artifact = directory / "fresh"
            artifact.mkdir()
            (artifact / "checks.json").write_text(json.dumps(checks))
            if mode == "duplicate":
                duplicate = directory / "another"
                duplicate.mkdir()
                (duplicate / "checks.json").write_text(json.dumps(checks))
            return 1 if mode == "exit" else 0
        (directory / "junit.xml").write_text(
            "<testsuites><testsuite>"
            + "".join(f'<testcase name="{node.split("::")[1]}"/>' for node in runner.SDK_TESTS)
            + "</testsuite></testsuites>"
        )
        return 0

    monkeypatch.setattr(runner, "run_bounded", execute)
    assert runner.main() == (0 if mode == "pass" else 1)
    report = json.loads((output / "summary.json").read_text())
    assert report["full_conformance"] is False
    assert report["covered_subset_passed"] is (mode == "pass")
    assert report["results"][-1]["kind"] == "sdk"
    assert report["results"][-1]["status"] == "passed"
    assert all(record["kind"] == "official" for record in report["results"][:-1])


def test_wrong_upstream_revision_fails_before_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "wrong\n")
    with pytest.raises(ValueError, match="pinned revision"):
        runner.verify_upstream(tmp_path)


def test_timeout_reaps_child_process(tmp_path):
    if sys.platform == "win32":
        pytest.skip("Process group cleanup is POSIX-only; use Docker on Windows.")
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=0.1
        )
    assert (tmp_path / "runner.stderr.txt").exists()

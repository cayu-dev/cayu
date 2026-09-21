"""Failed phases remain diagnosable without the final summary or artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests import conftest as reporting


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_ci_failure_evidence_is_bounded_and_emitted_once(tmp_path, monkeypatch, capsys, phase):
    target = tmp_path / "reports.jsonl"
    monkeypatch.setattr(reporting, "_CI_REPORT_PATH", target)
    report = SimpleNamespace(
        nodeid="test_example.py::test_failed",
        when=phase,
        duration=0.1,
        outcome="failed",
        failed=True,
        longreprtext="x" * 20000,
    )
    reporting.pytest_runtest_logreport(report)
    record = json.loads(target.read_text())
    assert record["failure"] == "x" * 16384
    assert record["failure_truncated"] is True
    assert capsys.readouterr().err == (f"\nCI failure: {report.nodeid} [{phase}]\n{'x' * 16384}\n")
    # Workers/non-CI runs neither emit duplicate diagnostics nor write reports.
    monkeypatch.setattr(reporting, "_CI_REPORT_PATH", None)
    reporting.pytest_runtest_logreport(report)
    assert not capsys.readouterr().err
    assert len(target.read_text().splitlines()) == 1


def test_ci_controller_prints_failure_before_summary_with_xdist(tmp_path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["tests.conftest"]\n')
    (tmp_path / "test_sample.py").write_text(
        "def test_failure():\n    assert False, 'ci-immediate-evidence'\n"
        "def test_success():\n    assert True\n"
    )
    env = os.environ.copy()
    env.pop("CAYU_CI_REPORT_OWNER_PID", None)
    env.pop("CAYU_REQUIRE_CURRENT_TEST_DURATIONS", None)
    env["GITHUB_ACTIONS"] = "true"
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "2", "--splits", "1", "--group", "1"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout
    assert result.stdout.count("CI failure: test_sample.py::test_failure [call]") == 1
    assert result.stdout.index("CI failure:") < result.stdout.index("FAILURES")
    records = [
        json.loads(line) for line in (tmp_path / ".ci-test-reports.jsonl").read_text().splitlines()
    ]
    failures = [record for record in records if record.get("outcome") == "failed"]
    assert len(failures) == 1
    assert "ci-immediate-evidence" in failures[0]["failure"]
    assert any(record.get("outcome") == "passed" for record in records)

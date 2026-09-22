"""Failed phases remain diagnosable without the final summary or artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests import conftest as reporting


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_ci_failure_evidence_is_bounded_and_emitted_once(tmp_path, monkeypatch, capsys, phase):
    monkeypatch.setattr(reporting, "_CI_FAILURE_LOGGING", True)
    report = SimpleNamespace(
        nodeid="test_example.py::test_failed",
        when=phase,
        duration=0.1,
        outcome="failed",
        failed=True,
        longreprtext="x" * 20000,
    )
    reporting.pytest_runtest_logreport(report)
    assert capsys.readouterr().err == (f"\nCI failure: {report.nodeid} [{phase}]\n{'x' * 16384}\n")
    # Workers/non-CI runs neither emit duplicate diagnostics nor write reports.
    monkeypatch.setattr(reporting, "_CI_FAILURE_LOGGING", False)
    reporting.pytest_runtest_logreport(report)
    assert not capsys.readouterr().err
    assert not (tmp_path / ".ci-test-reports.jsonl").exists()


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
    assert "ci-immediate-evidence" in result.stdout
    assert not (tmp_path / ".ci-test-reports.jsonl").exists()

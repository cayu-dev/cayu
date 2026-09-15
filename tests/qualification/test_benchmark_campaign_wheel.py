"""Credential-free acceptance through installed commands, with real process loss."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

_TARGET = "cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan"


def test_installed_benchmark_campaign_lifecycle(tmp_path):
    selected = os.environ.get("CAYU_BENCHMARK_WHEEL")
    uv = shutil.which("uv")
    if selected is None or uv is None:
        pytest.skip("Set CAYU_BENCHMARK_WHEEL to a built wheel and install uv.")
    wheel = Path(selected).resolve()
    assert wheel.is_file() and wheel.suffix == ".whl"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    venv = tmp_path / "installed"
    work = tmp_path / "operator"
    work.mkdir()
    subprocess.run(
        [uv, "venv", str(venv)], check=True, capture_output=True, timeout=60, env=environment
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def install(extras=""):
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), str(wheel) + extras],
            check=True,
            capture_output=True,
            timeout=180,
            env=environment,
        )

    def command(*arguments, expected=0, timeout=90):
        result = subprocess.run(
            [str(python), "-I", "-m", "cayu", *arguments],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        assert result.returncode == expected, (
            arguments,
            result.stdout[-4000:],
            result.stderr[-4000:],
        )
        return result

    install()
    imported = subprocess.run(
        [str(python), "-I", "-c", "import cayu; print(cayu.__file__)"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        cwd=work,
    )
    assert str(venv) in imported.stdout
    # Package authoring and validation do not require the optional HTTP server.
    command("eval", "package", "init", "package")
    command("eval", "package", "validate", "package")
    install("[server,files]")
    calls_path = work / ".cayu/synthetic-benchmark/calls.jsonl"

    def calls():
        return (
            []
            if not calls_path.exists()
            else [json.loads(line) for line in calls_path.read_text().splitlines()]
        )

    command("eval", "package", "init", "failures-package", "--failure-modes")
    command(
        "eval",
        "run",
        _TARGET,
        "--package",
        "failures-package",
        "--campaign-directory",
        "failures",
        "--max-retry-attempts",
        "1",
        "--output",
        "failures.json",
        expected=2,
    )
    failure = json.loads((work / "failures.json").read_text())
    categories = {row["case_id"]: row["failure_category"] for row in failure["trials"]}
    assert categories["provider-failure"] == "provider_failure"
    assert categories["scoring-failure"] == "scoring_failure"
    provider_trial = next(row for row in failure["trials"] if row["case_id"] == "provider-failure")
    assert provider_trial["usage_availability"] == "unavailable"
    assert provider_trial["usage"] is None
    command("eval", "retry", "failures", _TARGET, "--case", "provider-failure", "--trial", "1")
    after_retry = len(calls())
    command("eval", "retry", "failures", _TARGET, "--case", "provider-failure", "--trial", "1")
    command(
        "eval", "retry", "failures", _TARGET, "--case", "wrong-answer", "--trial", "1", expected=2
    )
    command("eval", "status", "failures", "--sessions", "--json", "--output", "status.json")
    command("eval", "failures", "failures", "--json", "--output", "failed-trials.json")
    command("eval", "report", "failures", "--output", "report.html")
    command("eval", "export", "failures", "--output", "evidence.zip")
    offline = command("eval", "status", "evidence.zip", "--json")
    assert "offline_export" in json.loads(offline.stdout)["limitations"]
    assert len(calls()) == after_retry
    assert len(json.loads((work / "status.json").read_text())["successors"]) == 1

    command(
        "eval",
        "run",
        _TARGET,
        "--package",
        "package",
        "--campaign-directory",
        "control",
        "--trials",
        "2",
        "--max-concurrency",
        "2",
        "--output",
        "control.json",
        expected=1,
    )
    before_read_only = len(calls())
    command("eval", "compare", "control", "control", "--output", "comparison.json")
    assert json.loads((work / "comparison.json").read_text())["compatibility"] == "comparable"
    command(
        "eval",
        "run",
        _TARGET,
        "--package",
        "package",
        "--campaign-directory",
        "subset",
        "--case",
        "echo",
        "--admit-only",
        "--output",
        "subset.json",
    )
    command("eval", "compare", "control", "subset", "--output", "incompatible.json", expected=2)
    command("eval", "package", "init", "scorer-v2", "--scorer-version", "2")
    command("eval", "rescore", "control", "--package", "scorer-v2", "--output", "rescored.json")
    rescored = json.loads((work / "rescored.json").read_text())
    assert rescored["candidate_calls"] == rescored["judge_calls"] == 0
    assert len(rescored["trials"]) == 6
    command("eval", "cancel", "subset")
    command("eval", "resume", "subset", _TARGET, expected=2)
    assert len(calls()) == before_read_only

    if os.name == "nt":
        pytest.skip("POSIX process-loss qualification requires SIGKILL.")
    command("eval", "package", "init", "interruption-package", "--interruption-case")
    interrupted_before = len(calls())
    with (
        (work / "interrupted.stdout").open("w") as stdout,
        (work / "interrupted.stderr").open("w") as stderr,
    ):
        process = subprocess.Popen(
            [
                str(python),
                "-I",
                "-m",
                "cayu",
                "eval",
                "run",
                _TARGET,
                "--package",
                "interruption-package",
                "--campaign-directory",
                "interrupted",
                "--case",
                "echo",
                "--case",
                "interruption",
            ],
            cwd=work,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 25
            while len(calls()) < interrupted_before + 2 and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.05)
            assert len(calls()) == interrupted_before + 2
            connection = sqlite3.connect(
                f"file:{work / 'interrupted/evals.sqlite3'}?mode=ro", uri=True
            )
            try:
                assert (
                    connection.execute(
                        "SELECT COUNT(*) FROM cayu_eval_run_trial_checkpoints"
                    ).fetchone()[0]
                    == 1
                )
            finally:
                connection.close()
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    command("eval", "resume", "interrupted", _TARGET, expected=2, timeout=75)
    recovered = json.loads(command("eval", "status", "interrupted", "--json").stdout)
    assert recovered["counts"]["passed"] == recovered["counts"]["unavailable"] == 1
    blocked = next(row for row in recovered["trials"] if row["case_id"] == "interruption")
    assert blocked["diagnostic_code"] == "recovery_reexecution_blocked"
    assert blocked["prior_session_references"]
    assert len(calls()) == interrupted_before + 2

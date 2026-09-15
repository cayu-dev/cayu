from __future__ import annotations

import json
import zipfile

from cayu.cli import main

_TARGET = "cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan"


def test_supported_package_launch_inspection_report_export(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "package", "init", "package"]) == 0
    assert main(["eval", "package", "validate", "package"]) == 0
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--package",
                "package",
                "--campaign-directory",
                "campaign",
                "--case",
                "echo",
                "--case",
                "attachment",
                "--max-concurrency",
                "2",
                "--trials",
                "2",
                "--output",
                "results.json",
                "--html-output",
                "results.html",
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["status"] == "passed"
    assert result["counts"]["total"] == 4
    assert "Admission:" in capsys.readouterr().err
    assert (
        main(["eval", "status", "campaign", "--json", "--sessions", "--output", "status.json"]) == 0
    )
    inspected = json.loads((tmp_path / "status.json").read_text())
    assert all(row["session_association"] == "result_revision" for row in inspected["trials"])
    assert main(["eval", "failures", "campaign", "--json", "--output", "failures.json"]) == 0
    assert json.loads((tmp_path / "failures.json").read_text())["trials"] == []
    assert main(["eval", "report", "campaign", "--output", "report.html"]) == 0
    assert main(["eval", "export", "campaign", "--output", "campaign.zip"]) == 0
    with zipfile.ZipFile(tmp_path / "campaign.zip") as archive:
        assert set(archive.namelist()) == {"inspection.json", "export.json"}
        assert b"CORRECT" not in archive.read("inspection.json")
    # Inspection aliases must not overwrite any retained source or linked store.
    assert main(["eval", "status", "campaign", "--output", "campaign/campaign.json"]) == 2
    assert (
        main(
            [
                "eval",
                "status",
                "campaign",
                "--output",
                inspected["trials"][0]["session_sqlite_path"],
            ]
        )
        == 2
    )


def test_package_validation_and_admission_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "package", "init", "package"]) == 0
    assert main(["eval", "run", _TARGET, "--package", "package", "--processes", "2"]) == 2
    assert main(["eval", "run", _TARGET, "--package", "package", "--model", "unbound"]) == 2
    assert main(["eval", "run", _TARGET, "--package", "package", "--case", "missing"]) == 2
    assert main(["eval", "run", _TARGET, "--trials", "2"]) == 2
    assert (
        main(
            [
                "eval",
                "run",
                _TARGET,
                "--package",
                "package",
                "--campaign-directory",
                "campaign",
                "--admit-only",
                "--output",
                "admitted.json",
            ]
        )
        == 0
    )
    admitted = json.loads((tmp_path / "admitted.json").read_text())
    assert admitted["counts"]["pending"] == 3
    assert all(row["run_status"] == "queued" for row in admitted["trials"])
    assert (
        main(["eval", "run", _TARGET, "--package", "package", "--campaign-directory", "campaign"])
        == 2
    )
    capsys.readouterr()

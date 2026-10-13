from __future__ import annotations

import copy
import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/report_ci_timings.py"
_REPORTER = runpy.run_path(str(_SCRIPT))
_report = _REPORTER["build_report"]
_fetch = _REPORTER["fetch_snapshot"]


def _at(seconds):
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)).isoformat()


def _run(attempt=1, *, start=0, conclusion="success"):
    return {
        "id": 7,
        "head_sha": "a" * 40,
        "path": ".github/workflows/ci.yml",
        "event": "pull_request",
        "status": "completed",
        "conclusion": conclusion,
        "run_attempt": attempt,
        "created_at": _at(0),
        "run_started_at": _at(start),
        "updated_at": _at(180),
    }


def _job(job_id, name, start, end, *, created=None, attempt=1, conclusion="success"):
    return {
        "id": job_id,
        "run_id": 7,
        "run_attempt": attempt,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": _at(start - 2 if created is None else created),
        "started_at": _at(start),
        "completed_at": _at(end),
        "runner_name": "private-runner-name",
        "runner_id": 456,
        "labels": ["private-runner-label"],
        "steps": [
            {
                "number": 1,
                "name": "Test",
                "conclusion": conclusion,
                "started_at": _at(start + 1),
                "completed_at": _at(end - 1),
            }
        ],
    }


def _snapshot(*attempts):
    return {
        "schema_version": 1,
        "repository": "example/project",
        "run": copy.deepcopy(attempts[-1]["run"]),
        "attempts": list(attempts),
    }


def _attempt(run, *jobs):
    return {"run": run, "total_count": len(jobs), "jobs": list(jobs)}


def _retry_snapshot():
    collection = _job(1, "Collection", 2, 20)
    failed = _job(2, "Tests", 30, 60, conclusion="failure")
    carried = {**collection, "id": 3, "run_attempt": 2, "created_at": _at(103)}
    retried = _job(4, "Tests", 110, 140, attempt=2)
    return _snapshot(
        _attempt(_run(conclusion="failure"), collection, failed),
        _attempt(_run(2, start=100), carried, retried),
    )


def test_retry_elapsed_includes_first_attempt_and_gap_without_double_counting():
    report = _report(_retry_snapshot())
    assert report["successful_selected_elapsed_seconds"] == 140
    assert report["observed_selected_job_execution_seconds"] == 78
    assert report["observed_selected_job_waiting_seconds"] == 6
    assert report["latest_selected_conclusions"] == {"success": 2}
    retry = report["attempts"][1]
    assert retry["gap_after_previous_selected_completion_seconds"] == 40
    assert retry["jobs"][0]["carried_from_job_id"] == 1
    assert retry["jobs"][0]["created_to_started_seconds"] is None
    assert retry["jobs"][0]["run_to_created_seconds"] is None
    assert retry["jobs"][1]["steps"][0]["seconds"] == 28
    # Reports keep timing identity, without copying runner inventory from API evidence.
    serialized = json.dumps(report)
    assert "private-runner" not in serialized and "runner_id" not in serialized


def test_third_attempt_retains_original_execution_identity():
    snapshot = _retry_snapshot()
    second = snapshot["attempts"][1]
    third_jobs = copy.deepcopy(second["jobs"])
    for job in third_jobs:
        job.update(id=job["id"] + 2, run_attempt=3, created_at=_at(200))
    third_run = _run(3, start=200)
    snapshot["attempts"].append(_attempt(third_run, *third_jobs))
    snapshot["run"] = third_run
    report = _report(snapshot)
    assert report["observed_selected_job_execution_seconds"] == 78
    assert report["successful_selected_elapsed_seconds"] == 140
    assert [job["carried_from_job_id"] for job in report["attempts"][2]["jobs"]] == [1, 4]


def test_dependency_wait_is_separate_from_job_waiting_and_parallel_execution():
    report = _report(
        _snapshot(
            _attempt(
                _run(),
                _job(1, "Collection", 2, 20),
                _job(2, "Tests A", 30, 80, created=22),
                _job(3, "Tests B", 32, 70, created=22),
            )
        )
    )
    assert report["successful_selected_elapsed_seconds"] == 80
    assert report["observed_selected_job_execution_seconds"] == 106
    second = report["attempts"][0]["jobs"][1]
    assert second["run_to_created_seconds"] == 22
    assert second["created_to_started_seconds"] == 8
    assert report["observed_selected_job_waiting_seconds"] == 20


def test_explicit_publishing_exclusion_preserves_validation_and_lists_excluded_job():
    snapshot = _snapshot(
        _attempt(
            _run(conclusion="failure"),
            _job(1, "Tests", 2, 20),
            _job(2, "Publish", 300, 320, conclusion="failure"),
        )
    )
    full = _report(snapshot)
    assert not full["selected_jobs_green"]
    assert full["successful_selected_elapsed_seconds"] is None
    assert full["elapsed_to_selected_completion_seconds"] == 320
    selected = _report(snapshot, exclude_jobs=("Publish",))
    assert selected["successful_selected_elapsed_seconds"] == 20
    assert selected["excluded_job_names"] == ["Publish"]
    assert selected["run_conclusion"] == "failure"
    assert not selected["attempts"][0]["jobs"][1]["selected"]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "neutral", "skipped"])
def test_non_successful_runs_are_never_reported_as_time_to_green(conclusion):
    report = _report(
        _snapshot(
            _attempt(_run(conclusion=conclusion), _job(1, "Tests", 2, 20, conclusion=conclusion))
        )
    )
    assert not report["selected_jobs_green"]
    assert report["successful_selected_elapsed_seconds"] is None


def test_skipped_jobs_do_not_extend_validation_elapsed_or_runner_work():
    report = _report(
        _snapshot(
            _attempt(
                _run(), _job(1, "Tests", 2, 20), _job(2, "Optional", 200, 210, conclusion="skipped")
            )
        )
    )
    assert report["successful_selected_elapsed_seconds"] == 20
    assert report["observed_selected_job_execution_seconds"] == 18
    assert report["latest_selected_conclusions"] == {"skipped": 1, "success": 1}


def test_missing_timestamps_are_unknown_and_never_invent_zero_duration():
    job = _job(1, "Tests", 2, 20)
    job.update(created_at=None, started_at=None, completed_at=None)
    job["steps"][0].update(started_at=None, completed_at=None)
    report = _report(_snapshot(_attempt(_run(), job)))
    assert report["successful_selected_elapsed_seconds"] is None
    assert report["jobs_with_unavailable_execution_seconds"] == [1]
    assert report["jobs_with_unavailable_waiting_seconds"] == [1]
    assert report["attempts"][0]["jobs"][0]["steps"][0]["seconds"] is None


@pytest.mark.parametrize(
    "metric",
    [
        "execution_seconds",
        "created_to_started_seconds",
        "run_to_created_seconds",
        "steps.1.seconds",
    ],
)
def test_reversed_intervals_preserve_other_measurements_and_original_timestamps(metric):
    job = _job(2, "Tests B", 120, 140)
    if metric == "execution_seconds":
        job["completed_at"] = _at(119)
    elif metric == "created_to_started_seconds":
        job["created_at"] = _at(121)
    elif metric == "run_to_created_seconds":
        job["created_at"] = _at(-1)
    else:
        job["steps"][0]["completed_at"] = _at(120)
    snapshot = _snapshot(_attempt(_run(), _job(1, "Tests A", 2, 20), job))
    original = copy.deepcopy(snapshot)
    report = _report(snapshot)
    measured = report["attempts"][0]["jobs"][1]
    assert measured["timing_anomalies"] == [metric]
    assert snapshot == original
    assert measured["started_at"] == job["started_at"]
    assert measured["completed_at"] == job["completed_at"]
    assert report["selected_jobs_green"]
    assert report["successful_selected_elapsed_seconds"] == (
        None if metric == "execution_seconds" else 140
    )
    assert report["observed_selected_job_execution_seconds"] == (
        18 if metric == "execution_seconds" else 38
    )
    assert report["jobs_with_unavailable_execution_seconds"] == (
        [2] if metric == "execution_seconds" else []
    )
    assert report["jobs_with_unavailable_waiting_seconds"] == (
        [2] if metric == "created_to_started_seconds" else []
    )
    if metric == "steps.1.seconds":
        assert measured["steps"][0]["seconds"] is None
    else:
        assert measured[metric] is None


@pytest.mark.parametrize("exclude_publishing", [False, True])
def test_cancelled_publication_with_reversed_timestamps_does_not_drop_report(exclude_publishing):
    publication = _job(2, "Publish", 121, 120, conclusion="cancelled")
    publication["steps"] = []
    snapshot = _snapshot(
        _attempt(_run(conclusion="cancelled"), _job(1, "Tests", 2, 20), publication)
    )
    report = _report(snapshot, exclude_jobs=("Publish",) if exclude_publishing else ())
    assert report["selected_jobs_green"] is exclude_publishing
    assert report["successful_selected_elapsed_seconds"] == (20 if exclude_publishing else None)
    assert report["elapsed_to_selected_completion_seconds"] == (20 if exclude_publishing else None)
    assert report["observed_selected_job_execution_seconds"] == 18
    assert report["jobs_with_unavailable_execution_seconds"] == ([] if exclude_publishing else [2])
    assert report["attempts"][0]["jobs"][1]["timing_anomalies"] == ["execution_seconds"]


def test_skipped_job_and_step_anomalies_do_not_extend_elapsed_or_runner_work():
    skipped = _job(2, "Optional", 121, 120, conclusion="skipped")
    report = _report(_snapshot(_attempt(_run(), _job(1, "Tests", 2, 20), skipped)))
    assert report["successful_selected_elapsed_seconds"] == 20
    assert report["observed_selected_job_execution_seconds"] == 18
    assert report["jobs_with_unavailable_execution_seconds"] == []
    assert report["attempts"][0]["jobs"][1]["timing_anomalies"] == [
        "execution_seconds",
        "steps.1.seconds",
    ]


def test_carried_anomalous_execution_is_unavailable_and_counted_only_once():
    snapshot = _retry_snapshot()
    for attempt in snapshot["attempts"]:
        attempt["jobs"][0]["completed_at"] = _at(1)
    report = _report(snapshot)
    assert report["selected_jobs_green"]
    assert report["successful_selected_elapsed_seconds"] is None
    assert report["observed_selected_job_execution_seconds"] == 60
    assert report["jobs_with_unavailable_execution_seconds"] == [1]
    assert report["attempts"][1]["jobs"][0]["carried_from_job_id"] == 1


def test_cli_preserves_anomalous_snapshot_for_offline_replay(tmp_path, monkeypatch, capsys):
    publication = _job(2, "Publish", 121, 120, conclusion="cancelled")
    publication["steps"] = []
    snapshot = _snapshot(
        _attempt(_run(conclusion="cancelled"), _job(1, "Tests", 2, 20), publication)
    )
    main = _REPORTER["main"]
    monkeypatch.setitem(main.__globals__, "fetch_snapshot", lambda *_: snapshot)
    path = tmp_path / "snapshot.json"
    assert main(["--repo", "example/project", "--run-id", "7", "--snapshot", str(path)]) == 0
    fetched = capsys.readouterr()
    assert not fetched.err
    assert json.loads(path.read_text()) == snapshot
    assert json.loads(fetched.out)["jobs_with_unavailable_execution_seconds"] == [2]
    assert main(["--input", str(path)]) == 0
    assert capsys.readouterr().out == fetched.out


def test_workflow_completion_before_creation_remains_unavailable():
    snapshot = _snapshot(_attempt(_run(), _job(1, "Tests", 2, 20)))
    snapshot["run"]["created_at"] = _at(21)
    report = _report(snapshot)
    assert report["selected_jobs_green"]
    assert report["successful_selected_elapsed_seconds"] is None
    assert report["timing_anomalies"] == ["elapsed_to_selected_completion_seconds"]
    assert report["observed_selected_job_execution_seconds"] == 18


@pytest.mark.parametrize(
    "defect",
    [
        "attempt",
        "count",
        "duplicate",
        "revision",
        "run",
        "unfinished",
        "unmatched_carry",
        "timezone",
    ],
)
def test_incomplete_or_inconsistent_evidence_is_rejected(defect):
    snapshot = _retry_snapshot()
    attempt = snapshot["attempts"][1]
    if defect == "attempt":
        snapshot["attempts"].pop(0)
    elif defect == "count":
        attempt["jobs"].pop()
    elif defect == "duplicate":
        attempt["jobs"].append(copy.deepcopy(attempt["jobs"][0]))
        attempt["total_count"] += 1
    elif defect == "revision":
        snapshot["attempts"][0]["run"]["head_sha"] = "b" * 40
    elif defect == "run":
        attempt["jobs"][1]["run_id"] = 8
    elif defect == "unfinished":
        attempt["jobs"][1]["status"] = "in_progress"
    elif defect == "unmatched_carry":
        attempt["jobs"][0]["started_at"] = _at(3)
    elif defect == "timezone":
        snapshot["run"]["created_at"] = "2026-01-01T00:00:00"
    with pytest.raises(ValueError):
        _report(snapshot)


@pytest.mark.parametrize("excluded", [("Typo",), ("Collection", "Tests")])
def test_exclusions_must_match_and_leave_a_nonempty_selection(excluded):
    with pytest.raises(ValueError):
        _report(_retry_snapshot(), exclude_jobs=excluded)


def test_fetch_uses_read_only_paginated_requests_and_verifies_snapshot_stability(monkeypatch):
    snapshot = _retry_snapshot()
    calls = []

    def request(command, **kwargs):
        assert command[:6] == ["gh", "api", "--hostname", "github.com", "--method", "GET"]
        assert kwargs["check"] and kwargs["timeout"] == 120
        endpoint = command[-1]
        calls.append(endpoint)
        if "/attempts/" not in endpoint:
            payload = snapshot["run"]
        else:
            number = int(endpoint.split("/attempts/")[1].split("/")[0])
            item = snapshot["attempts"][number - 1]
            if "/jobs?" in endpoint:
                assert "--paginate" in command and "--slurp" in command
                payload = [{"total_count": 2, "jobs": [job]} for job in item["jobs"]]
            else:
                payload = item["run"]
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(subprocess, "run", request)
    assert _fetch("example/project", 7) == snapshot
    assert calls[0] == calls[-1] == "repos/example/project/actions/runs/7"
    assert len(calls) == 6


@pytest.mark.parametrize("defect", ["pagination", "rerun", "in_progress"])
def test_fetch_rejects_missing_pages_and_races(monkeypatch, defect):
    run = _run()
    job = _job(1, "Tests", 2, 20)
    reads = 0

    def get(endpoint, *, paginate=False):
        nonlocal reads
        if paginate:
            return [{"total_count": 2 if defect == "pagination" else 1, "jobs": [job]}]
        if "/attempts/" in endpoint:
            return run
        reads += 1
        if defect == "in_progress":
            return {**run, "status": "in_progress"}
        if defect == "rerun" and reads == 2:
            return {**run, "run_attempt": 2}
        return run

    monkeypatch.setitem(_fetch.__globals__, "_get", get)
    with pytest.raises(ValueError):
        _fetch("example/project", 7)


def test_cli_replays_snapshot_offline_and_returns_actionable_errors(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_retry_snapshot()))
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--input", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == _report(_retry_snapshot())
    path.write_text("not json")
    invalid = subprocess.run(
        [sys.executable, str(_SCRIPT), "--input", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid.returncode == 2
    assert "Cannot report CI timings" in invalid.stderr
    assert not invalid.stdout and "Traceback" not in invalid.stderr

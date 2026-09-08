from __future__ import annotations

import asyncio
import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import (
    EvalCaseResult,
    EvalRun,
    EvalStatus,
    EvalSuiteTrialPolicyV1,
    EvalTrialResult,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SQLiteSessionStore,
    export_process_eval_run,
    inspect_eval_sessions,
    inspect_process_eval_run,
)
from cayu.cli import main


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def _receipts(root: Path, *, results: int = 0, terminal: str | None = None) -> Path:
    directory = root / "workers"
    directory.mkdir()
    _write(
        directory / "launch.json",
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "target": "absent_module:do_not_import",
            "processes": 2,
            "max_concurrency": 2,
            "case_timeout_seconds": 60,
            "startup_timeout_seconds": 120,
            "shutdown_grace_seconds": 30,
        },
    )
    identity = {
        "fingerprint": "a" * 64,
        "suite_id": "suite",
        "case_ids": ["case-a", "case-b"],
        "metadata": {"source_revision": "sha256:" + "b" * 64},
    }
    for index in range(2):
        _write(
            directory / f"ready-{index}.json",
            {"launch_id": "launch-1", "index": index, "pid": 900000 + index, "identity": identity},
        )
    _write(
        directory / "start.json",
        {"launch_id": "launch-1", "fingerprint": "a" * 64, "assignments": [["case-a"], ["case-b"]]},
    )
    now = datetime.now(UTC)
    for index in range(results):
        trial = EvalTrialResult(
            trial_number=1,
            status=EvalStatus.ERROR,
            error="bounded failure",
            started_at=now,
            completed_at=now,
        )
        case = EvalCaseResult.from_trials(
            case_id=identity["case_ids"][index],
            trials=[trial],
            trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1, max_concurrency=2),
        )
        result = EvalRun(
            run_id=f"worker-run-{index}",
            suite_id="suite",
            status=EvalStatus.ERROR,
            cases=[case],
            started_at=now,
            completed_at=now,
            metadata=identity["metadata"],
        )
        _write(directory / f"result-{index}.json", result.model_dump(mode="json"))
    if terminal is not None:
        _write(
            directory / f"{terminal}.json",
            {
                "launch_id": "launch-1",
                **(
                    {"exception_type": "RuntimeError", "automatic_replay": False}
                    if terminal == "incomplete"
                    else {}
                ),
            },
        )
    return directory


def test_old_partial_run_does_not_infer_case_completion_or_pid_liveness(tmp_path, capsys):
    directory = _receipts(tmp_path, results=1)
    assert main(["eval", "status", str(directory), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "admitted"
    assert result["owner_liveness"] == "not_checked"
    assert result["result_status"] is None
    assert result["counts"]["results_recorded"] == 1
    assert result["counts"]["assigned"] == 2
    assert result["counts"]["trials_observed_started"] == 0
    assert result["cases"][1]["observed_state"] == "not_observed"
    assert result["limitations"]


def test_complete_run_requires_all_results_and_preserves_failure(tmp_path):
    directory = _receipts(tmp_path, results=2, terminal="completed")
    result = asyncio.run(inspect_process_eval_run(directory))
    assert result.phase == "completed"
    assert result.result_status is EvalStatus.ERROR
    assert result.counts["error"] == 2
    assert result.plan_metadata["source_revision"] == "sha256:" + "b" * 64


@pytest.mark.parametrize(
    "corruption",
    [
        "launch",
        "assignment",
        "fingerprint",
        "duplicate",
        "missing_result",
        "both_terminal",
        "result_metadata",
    ],
)
def test_mismatched_receipts_cannot_claim_complete(tmp_path, corruption):
    directory = _receipts(tmp_path, results=2, terminal="completed")
    if corruption == "launch":
        _write(directory / "completed.json", {"launch_id": "other-launch"})
    elif corruption in {"assignment", "fingerprint"}:
        p = directory / "start.json"
        d = json.loads(p.read_text())
        d["assignments" if corruption == "assignment" else "fingerprint"] = (
            [["case-b"], ["case-a"]] if corruption == "assignment" else "c" * 64
        )
        _write(p, d)
    elif corruption == "duplicate":
        p = directory / "ready-0.json"
        d = json.loads(p.read_text())
        d["identity"]["case_ids"] = ["case-a", "case-a"]
        _write(p, d)
    elif corruption == "missing_result":
        (directory / "result-1.json").unlink()
    elif corruption == "both_terminal":
        _write(directory / "incomplete.json", {"launch_id": "launch-1"})
    else:
        p = directory / "result-1.json"
        d = json.loads(p.read_text())
        d["metadata"] = {"source_revision": "changed"}
        _write(p, d)
    with pytest.raises(ValueError):
        asyncio.run(inspect_process_eval_run(directory))


def test_failed_admission_is_inspectable_without_accepting_conflicting_identity(tmp_path):
    directory = _receipts(tmp_path, terminal="incomplete")
    (directory / "start.json").unlink()
    p = directory / "ready-1.json"
    d = json.loads(p.read_text())
    d["identity"]["fingerprint"] = "c" * 64
    _write(p, d)
    result = asyncio.run(inspect_process_eval_run(directory))
    assert result.phase == "incomplete"
    assert result.cases == ()
    assert "worker_admission_identities_disagree" in result.limitations


@pytest.mark.parametrize("kind", ["symlink", "duplicate_json", "oversized", "fifo"])
def test_receipt_reads_are_bounded_and_regular_files_only(tmp_path, monkeypatch, kind):
    directory = _receipts(tmp_path)
    path = directory / "launch.json"
    if kind == "symlink":
        other = tmp_path / "elsewhere.json"
        path.rename(other)
        path.symlink_to(other)
    elif kind == "duplicate_json":
        path.write_text('{"schema_version":1,"schema_version":2}')
    elif kind == "oversized":
        from cayu.evals import _inspection_documents

        monkeypatch.setattr(_inspection_documents, "PROCESS_DOCUMENT_MAX_BYTES", 8)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO requires POSIX")
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(ValueError):
        asyncio.run(inspect_process_eval_run(directory))


def test_export_contains_exact_hashed_receipts_not_application_files(tmp_path):
    directory = _receipts(tmp_path, results=1)
    (directory / "worker-0.log").write_text("not a requested export input")
    (directory / "private-key").write_text("not an export input")
    (directory / "progress-0.json.pending").write_text("unfinished write")
    output = tmp_path / "receipts.zip"
    result = export_process_eval_run(directory, output)
    assert result.phase == "admitted"
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("export.json"))
        assert "private-key" not in archive.namelist()
        assert "worker-0.log" not in archive.namelist()
        assert not any(name.endswith(".pending") for name in archive.namelist())
        for item in manifest["files"]:
            data = archive.read(item["path"])
            assert data == (directory / item["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
    if os.name == "posix":
        assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="new file"):
        export_process_eval_run(directory, output)
    with pytest.raises(ValueError, match="outside"):
        export_process_eval_run(directory, directory / "export.zip")


async def _seed_store(store):
    for session_id, parent in (("root", None), ("child", "root")):
        await store.create(
            RunRequest(
                agent_name="researcher",
                session_id=session_id,
                parent_session_id=parent,
                messages=[Message.text("user", "task")],
            ),
            identity=SessionIdentity(provider_name="test", model="model"),
        )
        await store.update_status(session_id, SessionStatus.RUNNING)
    for number in range(3):
        await store.append_event(
            "root",
            Event(
                type=EventType.MODEL_ERROR,
                session_id="root",
                agent_name="researcher",
                payload={
                    "error": f"overload-{number}",
                    "provider_error_code": "overloaded",
                    "retry": True,
                    "arguments": {"unrequested_secret": "do-not-export-arguments"},
                },
            ),
        )
    await store.append_event(
        "child",
        Event(
            type=EventType.MODEL_ERROR,
            session_id="child",
            payload={
                "error": "unknown provider outcome",
                "provider_recovery_disposition": "manual_settlement_required",
            },
        ),
    )


def test_public_session_inspection_bounds_diagnostics_and_distinguishes_settlement():
    async def exercise():
        store = InMemorySessionStore()
        await _seed_store(store)
        result = await inspect_eval_sessions(
            store, "root", max_diagnostics=2, now=datetime.now(UTC) + timedelta(minutes=5)
        )
        assert len(result.sessions) == 2
        assert all(session.activity == "stale" for session in result.sessions)
        assert result.sessions[1].manual_settlement_reported
        assert any(item.retry_scheduled for item in result.diagnostics)
        assert len(result.diagnostics) == 2
        assert "diagnostics_truncated" in result.limitations
        assert "do-not-export-arguments" not in result.model_dump_json()
        assert result.owner_liveness == "not_checked"
        truncated = await inspect_eval_sessions(store, "root", max_sessions=1)
        assert len(truncated.sessions) == 1
        assert "sessions_truncated" in truncated.limitations

    asyncio.run(exercise())


def test_legacy_session_manifest_is_bound_and_sqlite_inspection_is_read_only(tmp_path):
    directory = _receipts(tmp_path)
    database = tmp_path / "sessions.sqlite3"

    async def seed():
        store = SQLiteSessionStore(database)
        try:
            await _seed_store(store)
        finally:
            await store.close()

    asyncio.run(seed())
    before = database.read_bytes()
    evidence = tmp_path / "sessions.json"
    _write(
        evidence,
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "cases": [{"case_id": "case-a", "session_id": "root", "sqlite_path": str(database)}],
        },
    )
    result = asyncio.run(
        inspect_process_eval_run(directory, include_sessions=True, session_evidence=evidence)
    )
    assert result.cases[0].session_inspection is not None
    assert len(result.cases[0].session_inspection.sessions) == 2
    assert database.read_bytes() == before
    assert "portable_session_locator_unavailable" in result.cases[1].limitations
    d = json.loads(evidence.read_text())
    d["launch_id"] = "different"
    _write(evidence, d)
    with pytest.raises(ValueError, match="different evaluation launch"):
        asyncio.run(inspect_process_eval_run(directory, session_evidence=evidence))


def test_failure_cli_filters_cases_and_refuses_receipt_overwrite(tmp_path, capsys):
    directory = _receipts(tmp_path, results=2, terminal="completed")
    assert (
        main(["eval", "failures", str(directory), "--no-sessions", "--case", "case-b", "--json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert [case["case_id"] for case in result["cases"]] == ["case-b"]
    original = (directory / "launch.json").read_bytes()
    assert (
        main(
            ["eval", "status", str(directory), "--json", "--output", str(directory / "launch.json")]
        )
        == 2
    )
    assert (directory / "launch.json").read_bytes() == original


def test_finished_progress_cannot_promote_a_run_without_worker_results(tmp_path):
    directory = _receipts(tmp_path)
    now = datetime.now(UTC).isoformat()
    _write(
        directory / "progress-0.json",
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "index": 0,
            "fingerprint": "a" * 64,
            "observed_at": now,
            "trials": [
                {
                    "case_id": "case-a",
                    "trial_number": 1,
                    "state": "finished",
                    "status": "passed",
                    "score": 1.0,
                    "started_at": now,
                    "completed_at": now,
                }
            ],
        },
    )
    result = asyncio.run(inspect_process_eval_run(directory))
    assert result.cases[0].observed_trial_status is EvalStatus.PASSED
    assert result.cases[0].result_status is None
    assert result.result_status is None
    assert result.counts["passed"] == 0
    assert result.counts["trials_observed_finished"] == 1


@pytest.mark.parametrize(
    "changed", ["wrong_case", "wrong_launch", "unfinished_result", "naive_timestamp"]
)
def test_progress_rejects_mismatched_or_contradictory_observations(tmp_path, changed):
    directory = _receipts(tmp_path)
    now = datetime.now(UTC).isoformat()
    trial = {"case_id": "case-a", "trial_number": 1, "state": "started", "started_at": now}
    document = {
        "schema_version": 1,
        "launch_id": "launch-1",
        "index": 0,
        "fingerprint": "a" * 64,
        "observed_at": now,
        "trials": [trial],
    }
    if changed == "wrong_case":
        trial["case_id"] = "case-b"
    elif changed == "wrong_launch":
        document["launch_id"] = "other"
    elif changed == "unfinished_result":
        trial["status"] = "passed"
    else:
        trial["started_at"] = "2026-01-01T00:00:00"
    _write(directory / "progress-0.json", document)
    with pytest.raises(ValueError):
        asyncio.run(inspect_process_eval_run(directory))


def test_missing_session_store_is_not_created_or_reported_as_empty_success(tmp_path):
    directory = _receipts(tmp_path)
    database = tmp_path / "missing.sqlite3"
    evidence = tmp_path / "sessions.json"
    _write(
        evidence,
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "cases": [{"case_id": "case-a", "session_id": "root", "sqlite_path": str(database)}],
        },
    )
    result = asyncio.run(
        inspect_process_eval_run(directory, include_sessions=True, session_evidence=evidence)
    )
    assert result.cases[0].session_inspection is None
    assert any("session_inspection_unavailable" in item for item in result.cases[0].limitations)
    assert not database.exists()
    assert not list(tmp_path.glob("missing.sqlite3*"))


def test_nested_eval_observation_cannot_replace_outer_case_identity(tmp_path):
    from cayu.evals._process_progress import (
        ProcessEvalProgress,
        observe_eval_session,
        observe_eval_trial,
    )

    progress = ProcessEvalProgress(
        tmp_path, launch_id="launch", index=0, fingerprint="a" * 64, case_ids=("case-a",)
    )
    store = InMemorySessionStore()
    with progress.activate(), observe_eval_trial("case-a", 1):
        observe_eval_session(store, "outer-root")
        with observe_eval_trial("case-a", 1):
            observe_eval_session(store, "nested-root")
    recorded = json.loads((tmp_path / "progress-0.json").read_text())
    assert len(recorded["trials"]) == 1
    assert recorded["trials"][0]["session"]["session_id"] == "outer-root"


@pytest.mark.parametrize("sessions", [True, False])
def test_inspection_output_cannot_overwrite_linked_database(tmp_path, capsys, sessions):
    directory = _receipts(tmp_path)
    database = tmp_path / "sessions.sqlite3"

    async def seed():
        store = SQLiteSessionStore(database)
        try:
            await _seed_store(store)
        finally:
            await store.close()

    asyncio.run(seed())
    evidence = tmp_path / "sessions.json"
    _write(
        evidence,
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "cases": [{"case_id": "case-a", "session_id": "root", "sqlite_path": str(database)}],
        },
    )
    before = database.read_bytes()
    assert (
        main(
            [
                "eval",
                "status",
                str(directory),
                "--session-evidence",
                str(evidence),
                "--sessions" if sessions else "--no-sessions",
                "--json",
                "--output",
                str(database),
            ]
        )
        == 2
    )
    assert "must not overwrite" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert database.read_bytes() == before


@pytest.mark.parametrize("target", ["symlink", "hardlink", "wal", "shm", "journal", "receipt"])
def test_filtered_inspection_protects_source_aliases_and_sidecars(tmp_path, capsys, target):
    directory = _receipts(tmp_path)
    database = tmp_path / "sessions.sqlite3"
    database.write_bytes(b"preserve database evidence")
    now = datetime.now(UTC).isoformat()
    _write(
        directory / "progress-0.json",
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "index": 0,
            "fingerprint": "a" * 64,
            "observed_at": now,
            "trials": [
                {
                    "case_id": "case-a",
                    "trial_number": 1,
                    "state": "started",
                    "started_at": now,
                    "session": {"session_id": "root", "sqlite_path": str(database)},
                }
            ],
        },
    )
    output = tmp_path / "output.json"
    source = database
    if target == "symlink":
        output.symlink_to(database)
    elif target in {"hardlink", "receipt"}:
        if target == "receipt":
            source = directory / "launch.json"
        os.link(source, output)
    else:
        source = Path(str(database) + "-" + target)
        source.write_bytes(b"preserve SQLite sidecar evidence")
        output = source
    before = source.read_bytes()
    assert (
        main(
            [
                "eval",
                "failures",
                str(directory),
                "--no-sessions",
                "--case",
                "case-b",
                "--json",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "must not overwrite" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert source.read_bytes() == before


@pytest.mark.parametrize("state", ["finished", "interrupted"])
@pytest.mark.parametrize("results", [0, 1])
def test_failure_progress_diagnostics_remain_provisional(tmp_path, capsys, state, results):
    directory = _receipts(tmp_path, results=results)
    now = datetime.now(UTC).isoformat()
    trial = {"case_id": "case-a", "trial_number": 1, "state": state, "started_at": now}
    if state == "finished":
        trial.update(status="error", error="trial exceeded its timeout", completed_at=now)
    else:
        trial.update(exception_type="CancelledError")
    _write(
        directory / "progress-0.json",
        {
            "schema_version": 1,
            "launch_id": "launch-1",
            "index": 0,
            "fingerprint": "a" * 64,
            "observed_at": now,
            "trials": [trial],
        },
    )
    args = ["eval", "failures", str(directory), "--no-sessions", "--case", "case-a"]
    assert main([*args, "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    case = snapshot["cases"][0]
    assert case["observed_error"] == trial.get("error")
    assert case["observed_exception_type"] == trial.get("exception_type")
    assert case["error"] == ("bounded failure" if results else None)
    assert case["result_status"] == ("error" if results else None)
    assert snapshot["result_status"] is None
    assert snapshot["counts"]["results_recorded"] == results
    assert main(args) == 0
    rendered = capsys.readouterr().out
    assert "(provisional):" in rendered
    assert trial.get("error", trial.get("exception_type")) in rendered

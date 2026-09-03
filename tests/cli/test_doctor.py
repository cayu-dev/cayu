from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

import cayu.cli.doctor as doctor_cli
from cayu import Event, RunRequest, SQLiteSessionStore, SQLiteTaskStore
from cayu.cli import main
from cayu.cli.doctor import _run_bounded_worker
from cayu.runtime.sessions import MAX_SESSION_ID_BYTES, SessionIdentity
from cayu.support_bundles import (
    CollectorDisposition,
    SupportBundleOutcome,
    SupportBundleReport,
    SupportCollectorResult,
    encode_support_bundle,
    minimal_support_bundle_report,
    validate_support_bundle_archive,
)


def _write_project(root: Path, body: str) -> None:
    (root / "project.py").write_text(body, encoding="utf-8")
    sys.modules.pop("project", None)


def _report_document(bundle: Path) -> dict:
    payload = bundle.read_bytes()
    validate_support_bundle_archive(payload)
    with zipfile.ZipFile(bundle) as archive:
        return json.loads(archive.read("report.json"))


def _write_partial_worker_payload(
    connection,
    _target: str | None,
    _sessions: tuple[str, ...],
) -> None:
    os.write(connection.fileno(), struct.pack("!i", 256) + b"{")
    time.sleep(30)


def _interrupt_parent_worker(
    _connection,
    _target: str | None,
    _sessions: tuple[str, ...],
) -> None:
    os.kill(os.getppid(), signal.SIGINT)
    time.sleep(30)


def _block_during_publication(path: str, payload: bytes, temporary_name: str) -> None:
    from cayu.support_bundles import write_support_bundle_atomic

    Path(path).with_name("publisher-started").write_text("started", encoding="utf-8")

    def block_fsync(_descriptor: int) -> None:
        time.sleep(30)

    os.__dict__["fsync"] = block_fsync
    write_support_bundle_atomic(path, payload, _temporary_name=temporary_name)


def _publish_then_lose_acknowledgement(
    path: str,
    payload: bytes,
    temporary_name: str,
) -> None:
    from cayu.support_bundles import write_support_bundle_atomic

    write_support_bundle_atomic(path, payload, _temporary_name=temporary_name)
    raise SystemExit(1)


def _terminate_parent_during_publication(
    path: str,
    payload: bytes,
    temporary_name: str,
) -> None:
    del payload
    staging = Path(path).absolute().parent / temporary_name
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(30)


def _leave_staging_then_fail(path: str, payload: bytes, temporary_name: str) -> None:
    del payload
    staging = Path(path).absolute().parent / temporary_name
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    raise SystemExit(1)


def _block_during_staging_reconciliation(path: str, temporary_name: str) -> None:
    Path(path).with_name("reconciliation-started").write_text("started", encoding="utf-8")
    del temporary_name
    time.sleep(30)


def _block_past_child_deadline_during_publication(
    path: str,
    payload: bytes,
    temporary_name: str,
) -> None:
    del payload
    Path(path).with_name("stacked-publisher-started").write_text("started", encoding="utf-8")
    os.__dict__["_exit"] = lambda _code: None
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    staging = Path(path).absolute().parent / temporary_name
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    time.sleep(30)


def _block_past_child_deadline_during_reconciliation(
    path: str,
    temporary_name: str,
) -> None:
    del temporary_name
    Path(path).with_name("stacked-reconciliation-started").write_text(
        "started",
        encoding="utf-8",
    )
    os.__dict__["_exit"] = lambda _code: None
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)


def _return_with_non_daemon_thread(marker: str) -> None:
    thread = threading.Thread(target=time.sleep, args=(30,), daemon=False)
    thread.start()
    Path(marker).write_text(str(os.getpid()), encoding="utf-8")


def _send_worker_payload_then_stall_shutdown(
    connection,
    _target: str | None,
    _sessions: tuple[str, ...],
) -> None:
    thread = threading.Thread(target=time.sleep, args=(30,), daemon=False)
    thread.start()
    report = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="unsettled_worker_payload",
    )
    connection.send_bytes(report.model_dump_json().encode())
    connection.close()


def _send_worker_payload_then_settle_shutdown(
    connection,
    _target: str | None,
    _sessions: tuple[str, ...],
) -> None:
    thread = threading.Thread(target=time.sleep, args=(1.5,), daemon=False)
    thread.start()
    report = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="settled_worker_payload",
    )
    connection.send_bytes(report.model_dump_json().encode())
    connection.close()


def _linux_process_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def _worktree_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    worktree_source = str(Path(__file__).parents[2] / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        worktree_source if inherited is None else worktree_source + os.pathsep + inherited
    )
    return environment


def test_doctor_boots_once_embeds_check_and_suppresses_project_side_channels(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """import logging
import warnings
from pathlib import Path

from cayu import CayuApp


def build_app():
    count_path = Path("factory-count.txt")
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    if count == 0:
        print("project-stdout-secret-canary")
        print("project-stderr-secret-canary", file=__import__("sys").stderr)
        warnings.warn("project-warning-secret-canary")
        logging.warning("project-log-secret-canary")
    return CayuApp(enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "support.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--json",
            ]
        )
        == 1
    )

    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "bundle_written": True,
        "outcome": "partial",
        "schema_version": "1",
    }
    assert output.err == ""
    assert "secret-canary" not in output.out
    assert (tmp_path / "factory-count.txt").read_text() == "1"
    assert stat_mode(bundle) == 0o600

    document = _report_document(bundle)
    assert document["outcome"] == "partial"
    assert "secret-canary" not in json.dumps(document)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["check"]["disposition"] == "collected"
    assert collectors["check"]["evidence"]["report"]["schema_version"] == "2"
    assert collectors["tasks"]["disposition"] == "unavailable"
    assert collectors["tasks"]["reason_code"] == "task_store_not_configured"
    assert collectors["tasks"]["evidence"] is None
    assert collectors["tasks"]["evidence_bytes"] == 0
    assert type(collectors["tasks"]["duration_ms"]) is int
    assert collectors["manifest"]["evidence"]["mcp_manifest_policy_configured"] is False
    assert "provider_live_health" not in collectors

    assert main(["check", "project:build_app", "--json"]) == 1
    check_document = json.loads(capsys.readouterr().out)
    assert collectors["check"]["evidence"]["report"] == check_document


def test_boot_failure_still_writes_minimal_safe_bundle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    print("boot-output-secret-canary")
    raise RuntimeError("boot-exception-secret-canary")
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "failed.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--json",
            ]
        )
        == 2
    )

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "boot_failed"
    assert "secret-canary" not in output.out + output.err
    document = _report_document(bundle)
    assert document["outcome"] == "boot_failed"
    assert document["collectors"] == [
        {
            "disposition": "failed",
            "duration_ms": 0,
            "evidence": None,
            "evidence_bytes": 0,
            "name": "bootstrap",
            "reason_code": "application_boot_failed",
        }
    ]
    assert "secret-canary" not in json.dumps(document)


def test_doctor_redacts_one_forbidden_collector_and_preserves_safe_sections(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import AggregateAccuracy, CayuApp, InMemorySessionStore


class UnsafeSnapshotStore(InMemorySessionStore):
    async def aggregate_operational_snapshot(self, filters=None):
        snapshot = await super().aggregate_operational_snapshot(filters)
        return snapshot.model_copy(
            update={
                "accuracy": AggregateAccuracy(
                    kind="truncated",
                    reason="/private/session-snapshot-path-canary",
                    limit=1,
                )
            }
        )


def build_app():
    return CayuApp(session_store=UnsafeSnapshotStore(), enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "partial.zip"

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "partial"
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["sessions"]["disposition"] == "redacted"
    assert collectors["sessions"]["reason_code"] == ("collector_evidence_forbidden_content")
    assert collectors["runtime_identity"]["disposition"] == "collected"
    assert collectors["check"]["disposition"] == "collected"
    assert collectors["manifest"]["disposition"] == "collected"
    assert "session-snapshot-path-canary" not in json.dumps(document)


def test_doctor_does_not_create_a_missing_configured_sqlite_store(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "read-only-doctor"

[tool.cayu]
factory = "project:build_app"

[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"
""",
        encoding="utf-8",
    )
    _write_project(
        tmp_path,
        """from cayu import CayuApp, SQLiteSessionStore, SQLiteTaskStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore("data/cayu.db"),
        task_store=SQLiteTaskStore("data/cayu.db"),
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "missing-store.zip"

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 1

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "partial"
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["sessions"]["reason_code"] == "store_source_not_available"
    assert collectors["tasks"]["reason_code"] == "store_source_not_available"
    stores = collectors["stores"]["evidence"]["stores"]
    readiness = {item["role"]: item["schema_readiness"] for item in stores}
    durability = {item["role"]: item["durability"] for item in stores}
    assert readiness["session"] == "unavailable"
    assert readiness["task"] == "unavailable"
    assert readiness["eval"] == "unavailable"
    assert durability["session"] == "durable"
    assert durability["task"] == "durable"
    assert not (tmp_path / "data").exists()


def test_invalid_duplicate_session_request_writes_validation_failed_bundle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    raise AssertionError("factory must not run for invalid input")
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "invalid.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--session",
                "same",
                "--session",
                "same",
                "--json",
            ]
        )
        == 3
    )

    assert json.loads(capsys.readouterr().out)["outcome"] == "validation_failed"
    document = _report_document(bundle)
    assert document["collectors"][0]["reason_code"] == "duplicate_session_selector"


def _outcome_report(outcome: SupportBundleOutcome) -> SupportBundleReport:
    baseline = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="test_boot_failure",
    )
    if outcome is SupportBundleOutcome.BOOT_FAILED:
        return minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failed",
        )
    if outcome is SupportBundleOutcome.VALIDATION_FAILED:
        return minimal_support_bundle_report(
            outcome=SupportBundleOutcome.VALIDATION_FAILED,
            reason_code="test_validation_failed",
        )
    collectors = (
        ()
        if outcome is SupportBundleOutcome.CLEAN
        else (
            SupportCollectorResult(
                name="injected",
                disposition=CollectorDisposition.FAILED,
                duration_ms=0,
                evidence_bytes=0,
                reason_code="collector_failed",
            ),
        )
    )
    return SupportBundleReport.from_results(
        generated_at=datetime(2026, 1, 2, tzinfo=UTC),
        outcome=outcome,
        limits=baseline.limits,
        collection_duration_ms=0,
        collectors=collectors,
    )


@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    (
        (SupportBundleOutcome.CLEAN, 0),
        (SupportBundleOutcome.PARTIAL, 1),
        (SupportBundleOutcome.BOOT_FAILED, 2),
        (SupportBundleOutcome.VALIDATION_FAILED, 3),
    ),
)
def test_human_output_reports_each_stable_bundle_outcome(
    outcome: SupportBundleOutcome,
    exit_code: int,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_worker",
        lambda *_args, **_kwargs: _outcome_report(outcome),
    )

    assert main(["doctor", "--bundle", str(tmp_path / f"{outcome.value}.zip")]) == exit_code

    output = capsys.readouterr()
    assert output.out == f"Diagnostic support bundle: {outcome.value}.\n"
    assert output.err == ""


def test_human_output_reports_stable_output_write_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_worker",
        lambda *_args, **_kwargs: _outcome_report(SupportBundleOutcome.CLEAN),
    )

    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_publisher",
        lambda *_args, **_kwargs: False,
    )

    assert main(["doctor", "--bundle", str(tmp_path / "unwritten.zip")]) == 4

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Diagnostic support bundle: output_write_failed.\n"
    assert "secret-canary" not in output.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX blocking publication contract")
def test_whole_command_bounds_blocking_publication_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    limits = doctor_cli.DEFAULT_SUPPORT_BUNDLE_LIMITS.model_copy(
        update={
            "worker_timeout_seconds": 0.2,
            "publication_timeout_seconds": 5.0,
            "command_timeout_seconds": 16.0,
        }
    )
    monkeypatch.setattr(doctor_cli, "DEFAULT_SUPPORT_BUNDLE_LIMITS", limits)
    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_worker",
        lambda *_args, **_kwargs: _outcome_report(SupportBundleOutcome.CLEAN),
    )
    monkeypatch.setattr(
        doctor_cli,
        "_doctor_publisher_entry",
        _block_during_publication,
    )
    bundle = tmp_path / "blocked.zip"
    started = time.monotonic()

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 4

    assert time.monotonic() - started < 10
    assert json.loads(capsys.readouterr().out)["outcome"] == "output_write_failed"
    assert (tmp_path / "publisher-started").read_text(encoding="utf-8") == "started"
    assert not bundle.exists()
    assert list(tmp_path.glob(".blocked.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_doctor_refuses_symlink_output_without_touching_target(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outside"
    target.write_text("unchanged", encoding="utf-8")
    bundle = tmp_path / "support.zip"
    bundle.symlink_to(target)

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 4

    assert json.loads(capsys.readouterr().out) == {
        "bundle_written": False,
        "outcome": "output_write_failed",
        "schema_version": "1",
    }
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert bundle.is_symlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication acknowledgement contract")
def test_publication_acknowledgement_loss_reports_failure_with_complete_bundle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "support.zip"
    monkeypatch.setattr(
        doctor_cli,
        "_doctor_publisher_entry",
        _publish_then_lose_acknowledgement,
    )

    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 4

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "output_write_failed"
    _report_document(bundle)
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal delivery contract")
def test_parent_sigterm_reaps_publisher_and_cleans_exact_staging(tmp_path: Path) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="publication_sigterm_test",
        )
    )
    bundle = tmp_path / "support.zip"

    with pytest.raises(SystemExit) as terminated:
        doctor_cli._run_bounded_publisher(
            bundle,
            payload,
            publication_timeout_seconds=10,
            _publisher_entry=_terminate_parent_during_publication,
        )

    assert terminated.value.code == 128 + signal.SIGTERM
    assert not bundle.exists()
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []


def test_staging_reconciliation_is_bounded_when_cleanup_blocks(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    limits = doctor_cli.DEFAULT_SUPPORT_BUNDLE_LIMITS.model_copy(
        update={
            "worker_timeout_seconds": 0.2,
            "publication_timeout_seconds": 5.0,
            "reconciliation_timeout_seconds": 5.0,
            "command_timeout_seconds": 16.0,
        }
    )
    monkeypatch.setattr(doctor_cli, "DEFAULT_SUPPORT_BUNDLE_LIMITS", limits)
    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_worker",
        lambda *_args, **_kwargs: _outcome_report(SupportBundleOutcome.CLEAN),
    )
    monkeypatch.setattr(doctor_cli, "_doctor_publisher_entry", _leave_staging_then_fail)
    monkeypatch.setattr(
        doctor_cli,
        "_doctor_staging_reconciliation_entry",
        _block_during_staging_reconciliation,
    )
    bundle = tmp_path / "support.zip"
    started = time.monotonic()

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 4

    assert time.monotonic() - started < 15
    assert json.loads(capsys.readouterr().out)["outcome"] == "output_write_failed"
    assert (tmp_path / "reconciliation-started").read_text(encoding="utf-8") == "started"
    staging = list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp"))
    assert len(staging) == 1
    staging[0].unlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX stacked process teardown contract")
def test_complete_command_deadline_bounds_stacked_publication_teardown(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    terminate_seconds = 1.0
    kill_seconds = 0.05
    limits = doctor_cli.DEFAULT_SUPPORT_BUNDLE_LIMITS.model_copy(
        update={
            "worker_timeout_seconds": 0.5,
            "publication_timeout_seconds": 5.0,
            "reconciliation_timeout_seconds": 5.0,
            "command_timeout_seconds": 12.0,
        }
    )
    monkeypatch.setattr(doctor_cli, "DEFAULT_SUPPORT_BUNDLE_LIMITS", limits)
    monkeypatch.setattr(doctor_cli, "_PROCESS_TERMINATE_SECONDS", terminate_seconds)
    monkeypatch.setattr(doctor_cli, "_PROCESS_KILL_SECONDS", kill_seconds)
    monkeypatch.setattr(
        doctor_cli,
        "_PROCESS_TEARDOWN_RESERVE_SECONDS",
        limits.reconciliation_timeout_seconds + terminate_seconds + kill_seconds + 0.1,
    )

    def consume_worker_budget(
        *_args,
        worker_timeout_seconds: float,
        **_kwargs,
    ) -> SupportBundleReport:
        time.sleep(worker_timeout_seconds)
        return _outcome_report(SupportBundleOutcome.CLEAN)

    monkeypatch.setattr(doctor_cli, "_run_bounded_worker", consume_worker_budget)
    monkeypatch.setattr(
        doctor_cli,
        "_doctor_publisher_entry",
        _block_past_child_deadline_during_publication,
    )
    monkeypatch.setattr(
        doctor_cli,
        "_doctor_staging_reconciliation_entry",
        _block_past_child_deadline_during_reconciliation,
    )
    bundle = tmp_path / "stacked-timeout.zip"
    started = time.monotonic()

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 4

    elapsed = time.monotonic() - started
    assert elapsed < limits.command_timeout_seconds + 0.25
    assert json.loads(capsys.readouterr().out)["outcome"] == "output_write_failed"
    publisher_marker = tmp_path / "stacked-publisher-started"
    assert publisher_marker.exists(), elapsed
    assert publisher_marker.read_text(encoding="utf-8") == "started"
    assert (tmp_path / "stacked-reconciliation-started").read_text(encoding="utf-8") == "started"
    assert not bundle.exists()
    staging = list(tmp_path.glob(".stacked-timeout.zip.cayu-doctor-*.tmp"))
    assert len(staging) == 1
    staging[0].unlink()


def test_parent_deadline_terminates_a_hanging_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_project(
        tmp_path,
        """import time


def build_app():
    time.sleep(30)
""",
    )
    monkeypatch.chdir(tmp_path)
    started = time.monotonic()

    report = _run_bounded_worker(
        "project:build_app",
        (),
        worker_timeout_seconds=0.2,
    )

    assert time.monotonic() - started < 5
    assert report.outcome is SupportBundleOutcome.BOOT_FAILED
    assert report.collectors[0].disposition is CollectorDisposition.FAILED
    assert report.collectors[0].reason_code == "worker_deadline_or_exit"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal delivery contract")
def test_parent_interrupt_terminates_worker() -> None:
    existing_children = {child.pid for child in doctor_cli.multiprocessing.active_children()}

    with pytest.raises(KeyboardInterrupt):
        _run_bounded_worker(
            None,
            (),
            worker_timeout_seconds=10,
            _worker_entry=_interrupt_parent_worker,
        )

    assert {
        child.pid for child in doctor_cli.multiprocessing.active_children()
    } <= existing_children


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-lifetime regression")
def test_parent_sigterm_reaps_the_owned_worker(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        """import os
import time
from pathlib import Path


def build_app():
    Path("worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)
""",
    )
    marker = tmp_path / "worker.pid"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cayu",
            "doctor",
            "project:build_app",
            "--bundle",
            str(tmp_path / "support.zip"),
        ],
        cwd=tmp_path,
        env=_worktree_subprocess_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker_pid: int | None = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        worker_pid = int(marker.read_text(encoding="utf-8"))

        owner.terminate()
        owner.wait(timeout=3)

        assert owner.poll() is not None
        deadline = time.monotonic() + 2
        while _linux_process_running(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _linux_process_running(worker_pid)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        if worker_pid is not None and _linux_process_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death regression")
def test_worker_self_deadline_survives_abrupt_parent_death(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        """import os
import time
from pathlib import Path


def build_app():
    Path("worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(30)
""",
    )
    marker = tmp_path / "worker.pid"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from cayu.cli.doctor import _run_bounded_worker; "
                "_run_bounded_worker('project:build_app', (), "
                "worker_timeout_seconds=5)"
            ),
        ],
        cwd=tmp_path,
        env=_worktree_subprocess_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker_pid: int | None = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        worker_pid = int(marker.read_text(encoding="utf-8"))

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=2)
        deadline = time.monotonic() + 6
        while _linux_process_running(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert owner.poll() is not None
        assert not _linux_process_running(worker_pid)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        if worker_pid is not None and _linux_process_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


def test_child_lifetime_guard_covers_interpreter_shutdown(tmp_path: Path) -> None:
    marker = tmp_path / "shutdown-worker.pid"
    process = doctor_cli.multiprocessing.get_context("spawn").Process(
        target=doctor_cli._bounded_process_entry,
        args=(0.2, _return_with_non_daemon_thread, str(marker)),
    )
    process.start()
    try:
        process.join(timeout=5)

        assert marker.exists()
        assert not process.is_alive()
        assert process.exitcode == doctor_cli._SELF_DEADLINE_EXIT_CODE
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        process.close()


def test_worker_rejects_payload_when_process_requires_forced_teardown() -> None:
    report = _run_bounded_worker(
        None,
        (),
        worker_timeout_seconds=10,
        _worker_entry=_send_worker_payload_then_stall_shutdown,
    )

    assert report.outcome is SupportBundleOutcome.BOOT_FAILED
    assert report.collectors[0].reason_code == "worker_deadline_or_exit"


def test_worker_accepts_payload_when_process_settles_before_deadline() -> None:
    report = _run_bounded_worker(
        None,
        (),
        worker_timeout_seconds=20,
        _worker_entry=_send_worker_payload_then_settle_shutdown,
    )

    assert report.outcome is SupportBundleOutcome.BOOT_FAILED
    assert report.collectors[0].reason_code == "settled_worker_payload"


def test_store_failure_returns_partial_bundle_and_preserves_safe_sections(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp
from cayu.runtime.sessions import InMemorySessionStore


class FailingSnapshotStore(InMemorySessionStore):
    async def aggregate_operational_snapshot(self, filters=None):
        del filters
        raise RuntimeError("store-snapshot-secret-canary")


def build_app():
    return CayuApp(session_store=FailingSnapshotStore(), enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "partial.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--json",
            ]
        )
        == 1
    )

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "partial"
    assert "store-snapshot-secret-canary" not in output.out + output.err
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["runtime_identity"]["disposition"] == "collected"
    assert collectors["check"]["disposition"] == "collected"
    assert collectors["sessions"]["disposition"] == "failed"
    assert collectors["sessions"]["reason_code"] == "collector_failed"
    stores = collectors["stores"]["evidence"]["stores"]
    session_descriptor = next(item for item in stores if item["role"] == "session")
    assert session_descriptor["schema_readiness"] == "unavailable"
    assert document["collected_count"] > 0
    assert document["omitted_count"] > 0
    assert "store-snapshot-secret-canary" not in json.dumps(document)


def test_control_plane_cleanup_failure_is_typed_and_preserves_collected_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)

    def fail_cleanup(_context) -> None:
        raise RuntimeError("cleanup-secret-canary")

    monkeypatch.setattr(doctor_cli, "close_project_control_plane_context", fail_cleanup)
    report = doctor_cli._collect_project_report(
        target="project:build_app",
        sessions=(),
    )
    validated = validate_support_bundle_archive(encode_support_bundle(report))

    assert validated.outcome is SupportBundleOutcome.PARTIAL
    assert validated.collectors[-1].name == "control_plane_cleanup"
    assert validated.collectors[-1].reason_code == "control_plane_cleanup_failed"
    assert "cleanup-secret-canary" not in validated.model_dump_json()


def test_final_archive_validation_failure_has_stable_safe_outcome(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="injected_boot_failure",
    )
    monkeypatch.setattr(
        doctor_cli,
        "_run_bounded_worker",
        lambda *_args, **_kwargs: report,
    )
    original_encode = doctor_cli.encode_support_bundle
    calls = 0

    def fail_first_encode(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("archive-validation-secret-canary")
        return original_encode(value)

    monkeypatch.setattr(doctor_cli, "encode_support_bundle", fail_first_encode)
    bundle = tmp_path / "validation-failed.zip"

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 3

    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "bundle_written": True,
        "outcome": "validation_failed",
        "schema_version": "1",
    }
    assert "archive-validation-secret-canary" not in output.out + output.err
    document = _report_document(bundle)
    assert document["outcome"] == "validation_failed"
    assert document["collectors"][0]["reason_code"] == "bundle_validation_failed"
    assert "archive-validation-secret-canary" not in json.dumps(document)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_oversized_session_selector_fails_before_project_boot(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_project(
        tmp_path,
        """from cayu import CayuApp


def build_app():
    raise AssertionError("factory must not run for oversized input")
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "oversized-selector.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--session",
                "x" * (MAX_SESSION_ID_BYTES + 1),
                "--json",
            ]
        )
        == 3
    )

    assert json.loads(capsys.readouterr().out)["outcome"] == "validation_failed"
    document = _report_document(bundle)
    assert document["collectors"][0]["reason_code"] == "session_selector_too_large"


@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe framing regression")
def test_parent_deadline_covers_a_partial_worker_payload() -> None:
    started = time.monotonic()

    report = _run_bounded_worker(
        None,
        (),
        worker_timeout_seconds=0.2,
        _worker_entry=_write_partial_worker_payload,
    )

    assert time.monotonic() - started < 5
    assert report.outcome is SupportBundleOutcome.BOOT_FAILED
    assert report.collectors[0].reason_code == "worker_deadline_or_exit"


def test_doctor_keeps_partial_bundle_when_sqlite_task_snapshot_times_out(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_database = tmp_path / "tasks.sqlite"

    async def seed_task_store() -> None:
        store = SQLiteTaskStore(task_database)
        await store.close()

    asyncio.run(seed_task_store())
    _write_project(
        tmp_path,
        """import time

from cayu import CayuApp, SQLiteTaskStore


def slow_progress():
    time.sleep(0.05)
    return 0


def build_app():
    task_store = SQLiteTaskStore("tasks.sqlite")
    task_store._connection.set_progress_handler(slow_progress, 1)
    return CayuApp(task_store=task_store, enable_logging=False)
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "sqlite-timeout.zip"

    started = time.monotonic()
    assert main(["doctor", "project:build_app", "--bundle", str(bundle), "--json"]) == 1
    elapsed = time.monotonic() - started

    output = capsys.readouterr()
    # The timed-out SQLite read must settle well before the 20-second worker
    # deadline, while allowing the separately spawned publication owner to boot.
    assert elapsed < 15
    assert json.loads(output.out)["outcome"] == "partial"
    assert output.err == ""
    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["tasks"]["disposition"] == "timed_out"
    assert collectors["tasks"]["reason_code"] == "collector_deadline_elapsed"
    assert collectors["artifacts"]["disposition"] == "collected"


def test_doctor_explicit_session_uses_sqlite_bounded_safe_tail(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    session_id = "sqlite-private-session-canary"

    async def seed_store() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    environment_name="local",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_event(
                session_id,
                Event(
                    type="custom.sqlite-cli-secret-type",
                    session_id=session_id,
                    payload={"transcript": "sqlite-cli-model-text-canary"},
                ),
            )
        finally:
            await store.close()

    asyncio.run(seed_store())
    _write_project(
        tmp_path,
        """from cayu import CayuApp, SQLiteSessionStore


def build_app():
    return CayuApp(
        session_store=SQLiteSessionStore("sessions.sqlite"),
        enable_logging=False,
    )
""",
    )
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "session-tail.zip"

    assert (
        main(
            [
                "doctor",
                "project:build_app",
                "--bundle",
                str(bundle),
                "--session",
                session_id,
                "--json",
            ]
        )
        == 1
    )

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "partial"
    document = _report_document(bundle)
    tail = next(item for item in document["collectors"] if item["name"] == "session_events.1")
    assert tail["disposition"] == "collected"
    assert tail["evidence"]["projection"] == "redacted_envelope_only"
    assert tail["evidence"]["returned_count"] == 1
    assert tail["evidence"]["tail_complete"] is True
    assert tail["evidence"]["first_sequence"] == 1
    assert tail["evidence"]["last_sequence"] == 1
    serialized = json.dumps(document)
    assert session_id not in serialized
    assert "sqlite-cli-model-text-canary" not in serialized
    assert "sqlite-cli-secret-type" not in serialized


def test_doctor_uses_maintained_service_once_and_composes_system_diagnostics(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    database = tmp_path / "service-runtime.sqlite"

    async def initialize_runtime_stores() -> None:
        session_store = SQLiteSessionStore(database)
        task_store = SQLiteTaskStore(database)
        await task_store.close()
        await session_store.close()

    asyncio.run(initialize_runtime_stores())
    database_before = database.read_bytes()

    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "maintained-doctor"

[tool.cayu]
factory = "service_project:build_app"
service_factory = "service_project:build_service"
""",
        encoding="utf-8",
    )
    (tmp_path / "service_project.py").write_text(
        """from pathlib import Path

from fastapi import HTTPException, Request

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    S3ArtifactStore,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ServiceIdentityStoreKind,
    create_agent_service,
)


class Store:
    category = ServiceIdentityStoreKind.DURABLE

    async def reserve(self, **kwargs):
        raise AssertionError

    async def find(self, **kwargs):
        raise AssertionError

    async def find_by_session_id(self, **kwargs):
        raise AssertionError

    async def claim_execution(self, **kwargs):
        raise AssertionError

    async def heartbeat_execution(self, **kwargs):
        raise AssertionError

    async def release_execution(self, **kwargs):
        raise AssertionError

    async def record_result_receipt(self, **kwargs):
        raise AssertionError

    async def record_recovery_status(self, **kwargs):
        raise AssertionError

    async def finish(self, **kwargs):
        raise AssertionError


async def product_auth(_request: Request):
    raise HTTPException(status_code=401)


def _build_app():
    count_path = Path("service-factory-count.txt")
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    app = CayuApp(
        session_store=SQLiteSessionStore("service-runtime.sqlite"),
        task_store=SQLiteTaskStore("service-runtime.sqlite"),
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="artifact-environment"),
            artifact_store=S3ArtifactStore(
                "private-support-bucket",
                prefix="private-support-prefix",
            ),
        ),
        default=True,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="scripted-model"))
    return app


def build_app():
    Path("fallback-app-called.txt").write_text("called")
    raise AssertionError("doctor must select the maintained service factory")


def build_service(*, mode, project_context=None):
    Path("service-mode.txt").write_text(mode.value)
    return create_agent_service(
        _build_app(),
        agent_name="agent",
        mode=mode,
        product_access=AuthenticatedProductAccess(dependency=product_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="secret-password")
        ),
        product_store=Store(),
        project_context=project_context,
    )
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("service_project", None)
    bundle = tmp_path / "service-support.zip"

    assert main(["doctor", "--bundle", str(bundle), "--json"]) == 0

    output = capsys.readouterr()
    assert json.loads(output.out)["outcome"] == "clean"
    assert output.err == ""
    assert (tmp_path / "service-factory-count.txt").read_text() == "1"
    assert (tmp_path / "service-mode.txt").read_text() == "production"
    assert not (tmp_path / "fallback-app-called.txt").exists()
    assert database.read_bytes() == database_before

    document = _report_document(bundle)
    collectors = {item["name"]: item for item in document["collectors"]}
    assert collectors["project_identity"]["evidence"]["service_declared"] is True
    check_report = collectors["check"]["evidence"]["report"]
    assert check_report["service_evidence"]["service_contract"] == ("verified_maintained")
    diagnostic_codes = {item["code"] for item in check_report["diagnostics"]}
    assert "PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE" not in diagnostic_codes
    assert "PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE" not in diagnostic_codes
    stores = collectors["stores"]["evidence"]["stores"]
    durability = {item["role"]: item["durability"] for item in stores}
    assert durability["session"] == "durable"
    assert durability["task"] == "durable"
    artifacts = collectors["artifacts"]["evidence"]
    assert artifacts == {
        "kind": "artifact_availability",
        "registered": True,
        "registration_count": 1,
        "availability": "configured_only_not_live_verified",
    }
    control_plane = collectors["control_plane"]
    assert control_plane["disposition"] == "collected"
    diagnostics = control_plane["evidence"]["report"]
    assert diagnostics["deployment"] == {
        "name": None,
        "name_status": "not_provided",
        "api_access": "authenticated",
        "dashboard_access": "authenticated",
        "dashboard_enabled": True,
        "docs_enabled": None,
    }
    assert diagnostics["capabilities"]["actor"] is None
    assert diagnostics["capabilities"]["configured_store_roles"] == [
        "session",
        "task",
        "artifact",
    ]
    assert diagnostics["capabilities"]["surfaces"]["tasks"]["configured"] is True
    assert diagnostics["capabilities"]["surfaces"]["artifacts"]["configured"] is True
    assert diagnostics["artifact_stores"]["total_count"] == 1
    assert diagnostics["artifact_stores"]["registrations"] == []
    assert diagnostics["artifact_stores"]["truncated"] is True

    artifact_store_id = "s3://private-support-bucket/private-support-prefix"
    artifact_store_fingerprint = f"sha256:{sha256(artifact_store_id.encode()).hexdigest()}"
    bundle_payload = bundle.read_bytes()
    serialized_document = json.dumps(document, sort_keys=True)
    with zipfile.ZipFile(bundle) as archive:
        summary = archive.read("summary.txt")
    for private_value in (artifact_store_id, artifact_store_fingerprint):
        assert private_value not in serialized_document
        assert private_value.encode() not in summary
        assert private_value.encode() not in bundle_payload

    actor_bearing_document = json.loads(json.dumps(document))
    actor_control_plane = next(
        item for item in actor_bearing_document["collectors"] if item["name"] == "control_plane"
    )
    actor_control_plane["evidence"]["report"]["capabilities"]["actor"] = {
        "subject": "private-operator-canary",
        "tenant": "private-tenant-canary",
    }
    with pytest.raises(ValueError, match="cannot contain a request actor"):
        SupportBundleReport.model_validate(actor_bearing_document)

    fingerprint_bearing_document = json.loads(json.dumps(document))
    fingerprint_control_plane = next(
        item
        for item in fingerprint_bearing_document["collectors"]
        if item["name"] == "control_plane"
    )
    artifact_diagnostics = fingerprint_control_plane["evidence"]["report"]["artifact_stores"]
    artifact_diagnostics["registrations"] = [
        {
            "fingerprint": artifact_store_fingerprint,
            "store_contract_operations": ["list", "read", "write", "delete"],
        }
    ]
    artifact_diagnostics["truncated"] = False
    with pytest.raises(ValueError, match="cannot contain artifact store identities"):
        SupportBundleReport.model_validate(fingerprint_bearing_document)

    for invalid_document, private_values in (
        (
            actor_bearing_document,
            ("private-operator-canary", "private-tenant-canary"),
        ),
        (fingerprint_bearing_document, (artifact_store_fingerprint,)),
    ):
        tampered = io.BytesIO()
        with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, member_payload in (
                (
                    "report.json",
                    (json.dumps(invalid_document, indent=2, sort_keys=True) + "\n").encode(),
                ),
                ("summary.txt", summary),
            ):
                member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(member, member_payload)
        with pytest.raises(ValueError, match="report is invalid") as rejected:
            validate_support_bundle_archive(tampered.getvalue())
        assert all(private_value not in str(rejected.value) for private_value in private_values)

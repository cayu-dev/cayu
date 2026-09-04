"""Exercise cayu doctor from an installed wheel and scaffolded project."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from cayu.support_bundles import (
    RecoveryCleanupEvidence,
    StoreSummaryEvidence,
    validate_support_bundle_archive,
)


def _run(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != expected_returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {argv!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed


def main() -> int:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "CAYU_PROVIDER",
            "CAYU_DATABASE_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        }
    }
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="cayu-doctor-wheel-") as temporary:
        root = Path(temporary)
        _run(
            [
                sys.executable,
                "-m",
                "cayu",
                "new",
                "proof",
                "--without",
                "memory",
                "--without",
                "knowledge",
                "--dir",
                str(root),
            ],
            cwd=root,
            environment=environment,
        )
        project = root / "proof"
        bundle = root / "support.zip"
        completed = _run(
            [
                sys.executable,
                "-m",
                "cayu",
                "doctor",
                "--bundle",
                str(bundle),
                "--json",
            ],
            cwd=project,
            environment=environment,
            expected_returncode=1,
        )
        assert json.loads(completed.stdout) == {
            "bundle_written": True,
            "outcome": "partial",
            "schema_version": "1",
        }
        report = validate_support_bundle_archive(bundle.read_bytes())
        assert report.outcome.value == "partial"
        assert report.command_version == "1"
        assert report.bundle_id.startswith("bundle_")
        assert report.collector_count == len(report.collectors)
        assert {item.name for item in report.collectors} >= {
            "check",
            "manifest",
            "recovery_cleanup",
            "runtime_identity",
            "sessions",
            "stores",
        }
        stores = next(item for item in report.collectors if item.name == "stores")
        assert isinstance(stores.evidence, StoreSummaryEvidence)
        readiness = {item.role: item.schema_readiness for item in stores.evidence.stores}
        assert readiness["session"] == "unavailable"
        assert readiness["task"] == "unavailable"
        assert readiness["eval"] == "unavailable"
        recovery_cleanup = next(
            item for item in report.collectors if item.name == "recovery_cleanup"
        )
        assert isinstance(recovery_cleanup.evidence, RecoveryCleanupEvidence)
        assert recovery_cleanup.evidence.snapshot.active_tasks == 0
        assert recovery_cleanup.evidence.snapshot.retained_tasks == 0
        database = project / "data" / "cayu.db"
        assert all(
            not candidate.exists()
            for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
        )
        assert all(not item.name.startswith("session_events.") for item in report.collectors)
        with zipfile.ZipFile(bundle) as archive:
            assert archive.namelist() == ["report.json", "summary.txt"]
        assert list(root.glob(".support.zip.cayu-doctor-*.tmp")) == []
        if os.name == "posix":
            assert stat.S_IMODE(bundle.stat().st_mode) == 0o600
    print("built-wheel cayu doctor smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

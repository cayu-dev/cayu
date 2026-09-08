"""Opt-in real Docker pause and worker-loss continuity proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_DOCKER_RECONNECT") != "1",
        reason="Set CAYU_RUN_DOCKER_RECONNECT=1 with Docker Desktop or local Linux Docker.",
    ),
]


@pytest.mark.parametrize("boundary", ["retain", "crash"])
def test_separate_worker_reconnects_exact_allocation(tmp_path, boundary):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/egress/docker_reconnect_worker.py"
    environment = {**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"}
    produced = subprocess.run(
        [sys.executable, str(worker), boundary, str(tmp_path)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert produced.returncode == 0, produced.stderr
    previous = json.loads(produced.stdout)
    try:
        resumed = subprocess.run(
            [sys.executable, str(worker), "reconnect", str(tmp_path)],
            cwd=root,
            env=environment,
            input=json.dumps(previous) + "\n",
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert resumed.returncode == 0, resumed.stderr
        assert all(json.loads(resumed.stdout).values())
    finally:
        # Emergency cleanup remains exact-ID scoped if a regression interrupts the proof.
        identity = previous["metadata"]["identity"]
        subprocess.run(
            ["docker", "rm", "-f", identity["container_id"]], capture_output=True, timeout=20
        )
        journal = json.loads((tmp_path / f"{identity['allocation_id']}.json").read_text())
        if journal.get("sidecar_id"):
            subprocess.run(
                ["docker", "rm", "-f", journal["sidecar_id"]], capture_output=True, timeout=20
            )
        subprocess.run(
            ["docker", "network", "rm", identity["network_id"]], capture_output=True, timeout=20
        )


@pytest.mark.parametrize("boundary", ["pause", "crash_before_receipt", "crash_after_receipt"])
def test_application_browser_human_pause_and_worker_loss(tmp_path, boundary):
    root = Path(__file__).resolve().parents[2]
    worker = (
        root
        / "tests/egress"
        / (
            "docker_browser_reconnect_worker.py"
            if boundary == "pause"
            else "docker_browser_receipt_worker.py"
        )
    )
    environment = {**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"}
    try:
        for mode in (boundary, "resume"):
            completed = subprocess.run(
                [sys.executable, str(worker), mode, str(tmp_path)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
        assert json.loads((tmp_path / "mutations.json").read_text()) == 1
        click = json.loads((tmp_path / "click-result.json").read_text())
        if boundary == "crash_before_receipt":
            assert click["error"] and click["value"]["error"] == "outcome_ambiguous"
        else:
            assert not click["error"]
            assert json.loads((tmp_path / "cookies.json").read_text()) >= 2
        metadata = json.loads((tmp_path / "allocation.json").read_text())
        for kind, identifier in [
            ("container", metadata["identity"]["container_id"]),
            ("network", metadata["identity"]["network_id"]),
        ]:
            inspected = subprocess.run(
                ["docker", "inspect", "--type", kind, identifier], capture_output=True, timeout=20
            )
            assert inspected.returncode != 0
    finally:
        for path in (tmp_path / "ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            for identifier in [
                journal.get("identity", {}).get("container_id"),
                journal.get("sidecar_id"),
            ]:
                if identifier:
                    subprocess.run(
                        ["docker", "rm", "-f", identifier], capture_output=True, timeout=20
                    )
            if journal.get("network_id"):
                subprocess.run(
                    ["docker", "network", "rm", journal["network_id"]],
                    capture_output=True,
                    timeout=20,
                )

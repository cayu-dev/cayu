"""Opt-in combined protected UI/restart proof and selected real Docker failures."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_DOCKER_VIEW_RECONNECT") != "1",
        reason="Requires local application/browser images, built dashboard, and host Chromium; see examples/browser_view_reconnect/README.md.",
    ),
]


def test_protected_ui_same_browser_after_worker_restart(tmp_path):
    from examples.browser_view_reconnect.run import run

    root = tmp_path / "combined"
    asyncio.run(run(root))
    evidence = json.loads((root / "evidence.json").read_text())
    assert evidence["same_browser_container"] and evidence["same_page"]
    assert evidence["changing_frames_before"] and evidence["changing_frames_after"]
    assert evidence["independent_mutation_count"] == 1
    assert evidence["application_container_survived"] and evidence["allocation_cleanup"]


def test_real_attachment_failures_preserve_other_allocation(tmp_path):
    from examples.browser_view_reconnect.run import docker

    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "faults"
    root.mkdir(mode=0o700)
    (root / "staging").mkdir(mode=0o700)
    control = docker(
        "run",
        "-d",
        "--mount",
        "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        "--mount",
        f"type=bind,src={root},dst={root}",
        "--mount",
        f"type=bind,src={repo},dst={repo},readonly",
        "--env",
        f"CAYU_DEMO_STATE={root}",
        "--env",
        f"TMPDIR={root / 'staging'}",
        "--env",
        f"PYTHONPATH={repo / 'src'}",
        "cayu-view-reconnect:local",
    )
    (root / "configuration.json").write_text(json.dumps({"control_server_container_id": control}))
    try:
        completed = subprocess.run(
            [
                "docker",
                "exec",
                control,
                "python",
                str(repo / "tests/egress/docker_control_reconnect_worker.py"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert all(json.loads(completed.stdout).values())
    finally:
        for path in (root / "fault-ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            if journal.get("sidecar_id"):
                docker("rm", "-f", journal["sidecar_id"], check=False)
            if journal.get("network_id"):
                docker(
                    "network", "disconnect", "--force", journal["network_id"], control, check=False
                )
                docker("network", "rm", journal["network_id"], check=False)
        docker("rm", "-f", control, check=False)


@pytest.mark.parametrize("boundary", ["before_attach", "after_attach"])
def test_real_worker_loss_during_control_attachment_fences_recovery(tmp_path, boundary):
    from examples.browser_view_reconnect.run import docker

    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "crash"
    root.mkdir(mode=0o700)
    (root / "staging").mkdir(mode=0o700)
    control = docker(
        "run",
        "-d",
        "--mount",
        "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        "--mount",
        f"type=bind,src={root},dst={root}",
        "--mount",
        f"type=bind,src={repo},dst={repo},readonly",
        "--env",
        f"CAYU_DEMO_STATE={root}",
        "--env",
        f"TMPDIR={root / 'staging'}",
        "--env",
        f"PYTHONPATH={repo / 'src'}",
        "cayu-view-reconnect:local",
    )
    (root / "configuration.json").write_text(json.dumps({"control_server_container_id": control}))
    try:
        for mode in ("create", boundary, "recover"):
            completed = subprocess.run(
                [
                    "docker",
                    "exec",
                    control,
                    "python",
                    str(repo / "tests/egress/docker_control_reconnect_crash_worker.py"),
                    mode,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "fenced"
    finally:
        for path in (root / "crash-ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            for identifier in (
                journal.get("identity", {}).get("container_id"),
                journal.get("sidecar_id"),
            ):
                if identifier:
                    docker("rm", "-f", identifier, check=False)
            if journal.get("network_id"):
                docker(
                    "network", "disconnect", "--force", journal["network_id"], control, check=False
                )
                docker("network", "rm", journal["network_id"], check=False)
        docker("rm", "-f", control, check=False)

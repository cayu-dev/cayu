"""Real Docker completed turns resume with a fresh allocation in the surviving app container.

Set CAYU_RUN_DOCKER_RECONNECT=1; requires cayu-view-reconnect:local and
pinned browser images. For the actual Runtime/browser upgrade lane also set
CAYU_DOCKER_COMPLETED_RESUME_OLD_SOURCE to a checkout of Runtime 1eed25e
(browser worker 11). It is mounted read-only and never modified.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_DOCKER_RECONNECT") != "1",
        reason="Requires opt-in real Docker images.",
    ),
]


def docker(*args, check=True, timeout=30):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("boundary", ["same-worker", "new-worker", "adopt", "adopt-after-refusal"])
def test_completed_browser_ordinary_resume(tmp_path, boundary):
    repo = Path(__file__).resolve().parents[2]
    old_source = os.environ.get("CAYU_DOCKER_COMPLETED_RESUME_OLD_SOURCE")
    upgrading = boundary.startswith("adopt")
    if upgrading and not old_source:
        pytest.skip("Set CAYU_DOCKER_COMPLETED_RESUME_OLD_SOURCE for a real old/new build upgrade.")
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "staging").mkdir(mode=0o700)
    mounts = []
    if upgrading:
        old_source = str(Path(old_source).resolve())
        mounts = ["--mount", f"type=bind,src={old_source},dst={old_source},readonly"]
    control = docker(
        "run",
        "-d",
        "--init",
        "--mount",
        "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        "--mount",
        f"type=bind,src={root},dst={root}",
        "--mount",
        f"type=bind,src={repo},dst={repo},readonly",
        *mounts,
        "--env",
        f"TMPDIR={root / 'staging'}",
        "cayu-view-reconnect:local",
    )
    (root / "configuration.json").write_text(json.dumps({"control_server_container_id": control}))
    worker = repo / "tests/egress/docker_completed_resume_worker.py"
    try:
        steps = (
            [("same-worker", repo)]
            if boundary == "same-worker"
            else [
                ("complete", Path(old_source) if upgrading else repo),
                *([("refused", Path(old_source))] if boundary == "adopt-after-refusal" else []),
                ("adopt" if upgrading else "resume", repo),
            ]
        )
        for mode, source in steps:
            docker(
                "exec",
                "--env",
                f"PYTHONPATH={source / 'src'}",
                control,
                "python",
                str(worker),
                mode,
                str(root),
                timeout=180,
            )
            state = json.loads(docker("inspect", control))[0]
            assert state["State"]["Running"]
        evidence = json.loads((root / "completed-resume-evidence.json").read_text())
        assert all(
            evidence[key]
            for key in (
                "fresh_allocation",
                "fresh_browser",
                "transcript_preserved",
                "same_session",
                "both_disposals_proven",
            )
        )
    finally:
        # Exact IDs only, including failures before publication of the fixture result.
        for path in (root / "ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            for identifier in [
                journal.get("identity", {}).get("container_id"),
                journal.get("sidecar_id"),
            ]:
                if identifier:
                    docker("rm", "-f", identifier, check=False)
            if journal.get("network_id"):
                docker(
                    "network", "disconnect", "--force", journal["network_id"], control, check=False
                )
                docker("network", "rm", journal["network_id"], check=False)
        docker("rm", "-f", control, check=False)

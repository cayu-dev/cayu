from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from tests.core.test_browser_session import _interactive_limits

from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD


@pytest.mark.process
@pytest.mark.parametrize("case", ["blank", "direct", "reordered", "post"])
def test_real_chromium_popup_guard_route_and_settlement(case: str):
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    root = Path(__file__).resolve().parents[2]
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker is unavailable")
    image = PINNED_BROWSER_SESSION_WORKLOAD.image
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip("Pinned Chromium image is unavailable")
    limits = _interactive_limits(
        max_pages=3,
        max_provisional_pages=2,
        max_page_creations_per_operation=2,
        max_total_page_creations=3,
        max_snapshot_bytes=16384,
        max_dom_nodes=1000,
        max_wait_ms=5000,
        max_response_bytes=1048576,
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/usr/local/bin/python",
            "-i",
            "-v",
            f"{root / 'src/cayu/tools/_browser_guest.py'}:/opt/cayu-browser/worker.py:ro",
            "-v",
            f"{root / 'tests/egress/browser_popup_guest_probe.py'}:/tmp/probe.py:ro",
            image,
            "/tmp/probe.py",
        ],
        input=json.dumps({"case": case, "limits": asdict(limits)}),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"] is True
    assert not result.stderr


@pytest.mark.process
@pytest.mark.parametrize("case_id", ["page-active-page-crash", "page-background-page-crash"])
def test_real_chromium_exact_page_crash_preserves_allocation(case_id: str):
    from tests.evals.test_browser_acceptance_execution import _project_scenario_execution

    from cayu.evals import BrowserAcceptanceFixtureV1
    from cayu.evals.internal.browser_acceptance import build

    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", PINNED_BROWSER_SESSION_WORKLOAD.image],
        capture_output=True,
        timeout=15,
    ).returncode:
        pytest.skip("Pinned Chromium image is unavailable")

    async def scenario(fixture):
        plan = await build(fixture)
        case = next(case for case in plan.manifest.cases if case.case_id == case_id)
        result = await plan.scenario_executor(case, 1, 1, 90)
        assert result.fault.boundary_observed
        assert result.fault.browser_dispatches == 3
        receipt = await _project_scenario_execution(plan, case, result, fixture)
        assert receipt.semantic_state.value == "passed", receipt

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))

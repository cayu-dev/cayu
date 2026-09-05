"""Real Chromium delivery races; no provider, external network, or downloaded image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from tests.egress._browser_visual_container import run_visual_container

from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_VISUAL_BROWSER_ACCEPTANCE") != "1",
        reason="Set CAYU_RUN_VISUAL_BROWSER_ACCEPTANCE=1 with the pinned browser image installed.",
    ),
]

_PROGRAM = r'''
import asyncio
import sys

sys.path.insert(0, "/repo/src/cayu/tools")
from _browser_visual_guest import VisualGuestFailure, VisualPageOwner
from playwright.async_api import async_playwright

async def main():
    case = sys.argv[1]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            context = await browser.new_context(viewport={"width":800,"height":600})
            page = await context.new_page()
            cdp = await context.new_cdp_session(page)
            owner = VisualPageOwner(cdp, "worker")
            await owner.install()
            html = """<!doctype html><title>Native guard</title>
              <a id="control" href="#original" style="display:block;width:200px;height:100px">Go</a>
              <script>window.effects=0;control.addEventListener('click',()=>window.effects++);</script>"""
            if case.startswith("shadow-") or case == "opaque-host":
                tag = "x-control" if case == "shadow-custom" else "div"
                html = f"""<!doctype html><title>Opaque surface</title>
                  <{tag} id="control" style="display:block;width:200px;height:100px;cursor:pointer"></{tag}>
                  <script>window.effects=0;
                  control.addEventListener('click',()=>window.effects++);</script>"""
                if case != "opaque-host":
                    kind = "password" if case == "shadow-password" else "file"
                    html += f"""<script>control.attachShadow({{mode:'closed'}}).innerHTML=
                      '<input type="{kind}" style="width:200px;height:100px">';</script>"""
            if case == "svg-use":
                html = """<!doctype html><title>SVG shadow surface</title>
                  <svg width="200" height="100" style="cursor:pointer" onclick="window.effects++">
                  <defs><rect id="shape" width="200" height="100" fill="green"/></defs>
                  <use href="#shape"/></svg><script>window.effects=0;</script>"""
            await context.route("https://visual.test/**", lambda route: route.fulfill(
                status=200, content_type="text/html", body=html))
            await page.goto("https://visual.test/")
            policy = {"allowed_origins":["https://visual.test"], "max_frame_depth":0,
                      "max_width":800,"max_height":600,"max_pixels":480000,
                      "max_targets":8,"max_label_bytes":64,"max_hit_tests":16,
                      "evidence_lifetime_ms":10000,"allow_coordinate_fallback":True}
            candidate = await owner.collect(policy, max_dom_nodes=100)
            pixels = await page.screenshot()
            evidence = await owner.seal(policy,candidate,pixels,session_id="session",
                                        page_id="page",revision="revision",control_epoch=1)
            request = {"operation":"click_visual_target","session_id":"session","page_id":"page",
                       "expected_revision":"revision","expected_control_epoch":1,
                       "visual_revision":evidence["visual_revision"],
                       "visual_ref":candidate["targets"][0]["ref"]}
            expected = None
            if case.startswith("shadow-") or case in {"opaque-host", "svg-use"}:
                expected = "unsupported_visual_surface"
                assert not candidate["targets"][0]["actionable"]
                assert "opaque_surface" in candidate["unsupported_reasons"]
            if case == "unsampled-point":
                request.pop("visual_ref")
                request.update(operation="click_visual_point",screenshot_sha256=evidence["screenshot_sha256"],
                               x=0.05,y=0.05)
                expected = "unsupported_visual_surface"
            if case in {"after-arm", "action-mutation"}:
                if case == "action-mutation":
                    # Mutation caused by the admitted first event is not stale
                    # pre-input evidence. The normal click must still complete.
                    await page.evaluate("control.addEventListener('pointerdown',()=>control.setAttribute('data-active','yes'))")
                original_send = cdp.send
                async def send(method, params=None):
                    if method == "Input.dispatchMouseEvent" and params["type"] == "mousePressed":
                        if case == "after-arm":
                            await page.evaluate("control.href='#substituted'")
                    return await original_send(method, params)
                cdp.send = send
                if case == "after-arm":
                    expected = "visual_evidence_expired"
            if expected:
                try:
                    await owner.click(policy,request)
                except VisualGuestFailure as error:
                    assert error.code == expected, error.code
                else:
                    raise AssertionError("Unproven input was delivered")
                assert await page.evaluate("window.effects") == 0
                assert page.url == "https://visual.test/"
            else:
                await owner.click(policy,request)
                assert await page.evaluate("window.effects") == 1
                assert page.url.endswith("#original")
            await owner.disarm(policy)
        finally:
            await browser.close()

asyncio.run(main())
'''


@pytest.mark.parametrize(
    "case",
    [
        "unchanged",
        "after-arm",
        "action-mutation",
        "shadow-password",
        "shadow-file",
        "shadow-custom",
        "opaque-host",
        "svg-use",
        "unsampled-point",
    ],
)
def test_visual_native_guard(case: str) -> None:
    root = Path(__file__).resolve().parents[2]
    run_visual_container(
        [
            "--network",
            "none",
            "--entrypoint",
            "python",
            "--volume",
            f"{root}:/repo:ro",
            PINNED_BROWSER_SESSION_WORKLOAD.image,
            "-c",
            _PROGRAM,
            case,
        ],
    )


def test_visual_container_timeout_reaps_the_running_container() -> None:
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_visual_container(
            [
                "--network",
                "none",
                "--entrypoint",
                "python",
                PINNED_BROWSER_SESSION_WORKLOAD.image,
                "-c",
                "import time; print('container-started', flush=True); time.sleep(300)",
            ],
            timeout=10,
        )
    assert caught.value.__cause__ is None
    assert b"container-started" in caught.value.output
    name = caught.value.cmd[3]
    inspected = subprocess.run(
        ["docker", "inspect", "--type", "container", name],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert inspected.returncode != 0 and inspected.stdout.strip() == "[]"

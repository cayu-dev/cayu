"""Run the synthetic Docker recording example with no live viewer connected."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import shlex
import ssl
import subprocess
from pathlib import Path

import httpx
from examples.browser_view_reconnect.run import certificates, docker
from playwright.async_api import async_playwright, expect

REPO = Path(__file__).resolve().parents[2]
SESSION = "browser-recording-demo"


async def run(
    root: Path, *, explicit_close: bool = False, enabled: bool = True, restart: bool = False
):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    (root / "staging").mkdir(mode=0o700)
    certificates(root)
    container = docker(
        "run",
        "-d",
        "--init",
        "--mount",
        "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        "--mount",
        f"type=bind,src={root},dst={root}",
        "--mount",
        f"type=bind,src={REPO},dst={REPO},readonly",
        "--env",
        f"CAYU_RECORDING_DEMO_STATE={root}",
        "--env",
        f"TMPDIR={root / 'staging'}",
        "--env",
        f"PYTHONPATH={REPO / 'src'}",
        "-p",
        "127.0.0.1::8443",
        "cayu-browser-recording:local",
    )
    tasks = []
    try:
        port = json.loads(docker("inspect", container))[0]["NetworkSettings"]["Ports"]["8443/tcp"][
            0
        ]["HostPort"]
        origin = f"https://127.0.0.1:{port}"
        password = secrets.token_urlsafe(32)
        path = root / "configuration.json"
        path.write_text(
            json.dumps(
                {
                    "control_server_container_id": container,
                    "password": password,
                    "explicit_close": explicit_close,
                    "enabled": enabled,
                    "restart": restart,
                    "review_key": secrets.token_hex(32),
                }
            )
        )
        os.chmod(path, 0o600)

        def start_worker():
            docker(
                "exec",
                "-d",
                container,
                "sh",
                "-c",
                f"exec python {shlex.quote(str(REPO / 'examples/browser_recording/app.py'))} > {shlex.quote(str(root / 'worker.log'))} 2>&1",
            )

        start_worker()
        tls = ssl.create_default_context(cafile=str(root / "certificate.pem"))
        async with httpx.AsyncClient(
            base_url=origin, verify=tls, auth=("operator", password), timeout=150
        ) as client:
            async with asyncio.timeout(30):
                while True:
                    try:
                        if (await client.get("/api/health")).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.1)
            ids = []
            for invocation in (1, 2):
                task = asyncio.create_task(
                    client.post(
                        "/api/run" if invocation == 1 else "/api/resume",
                        json={
                            "session_id": SESSION,
                            "prompt": f"Show the public recording page, invocation {invocation}.",
                            "limits": {"max_elapsed_seconds": 120},
                        },
                    )
                )
                tasks.append(task)
                async with asyncio.timeout(60):
                    while not (root / f"page-{invocation}.json").exists():
                        if task.done():
                            raise AssertionError((await task).text)
                        await asyncio.sleep(0.1)
                    while enabled:
                        records = (
                            await client.get(f"/api/browser-recordings/sessions/{SESSION}")
                        ).json()
                        if len(records) >= invocation and len(records[-1]["segments"]) >= 3:
                            break
                        if task.done():
                            raise AssertionError((await task).text)
                        await asyncio.sleep(0.2)
                if restart and invocation == 1:
                    before = records[-1]
                    (root / "pause").touch()
                    assert (await task).status_code == 200
                    worker = json.loads((root / "worker.json").read_text())
                    docker(
                        "exec",
                        container,
                        "python",
                        "-c",
                        "import os,signal,sys; from pathlib import Path; pid=int(sys.argv[1]); "
                        "assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19] == sys.argv[2]; os.kill(pid,signal.SIGKILL)",
                        str(worker["pid"]),
                        worker["start_time"],
                    )
                    start_worker()
                    async with asyncio.timeout(30):
                        while True:
                            try:
                                response = await client.get(
                                    f"/api/sessions/{SESSION}/human-review",
                                    params={"purpose": "demo"},
                                )
                                if (
                                    response.status_code == 200
                                    and response.json().get("status") == "permitted"
                                ):
                                    view = response.json()
                                    break
                            except httpx.TransportError:
                                pass
                            await asyncio.sleep(0.1)
                    task = asyncio.create_task(
                        client.post(
                            "/api/user-input/resolve",
                            json={
                                "session_id": SESSION,
                                "input_id": view["interaction_id"],
                                "answer": "yes",
                                "review_reference": view["reference"],
                            },
                        )
                    )
                    tasks.append(task)
                    async with asyncio.timeout(40):
                        while True:
                            records = (
                                await client.get(f"/api/browser-recordings/sessions/{SESSION}")
                            ).json()
                            if (root / "resumed.json").exists() and len(
                                records[-1]["segments"]
                            ) >= len(before["segments"]) + 2:
                                assert records[-1]["recording_id"] == before["recording_id"]
                                assert records[-1]["identity"] == before["identity"]
                                break
                            if task.done():
                                raise AssertionError((await task).text)
                            await asyncio.sleep(0.2)
                (root / f"finish-{invocation}").touch()
                result = await task
                assert result.status_code == 200, result.text
                async with asyncio.timeout(20):
                    while enabled:
                        records = (
                            await client.get(f"/api/browser-recordings/sessions/{SESSION}")
                        ).json()
                        current = records[-1]
                        if current["status"] != "recording":
                            break
                        await asyncio.sleep(0.1)
                if enabled:
                    assert current["status"] in {"complete", "partial"}, current
                    assert current["video_sha256"], current
                    video = await client.get(
                        f"/api/browser-recordings/{current['recording_id']}/media?download=true"
                    )
                    assert video.status_code == 200
                    subprocess.run(
                        ["ffmpeg", "-v", "error", "-xerror", "-i", "pipe:0", "-f", "null", "-"],
                        input=video.content,
                        check=True,
                        capture_output=True,
                        timeout=20,
                    )
                    (root / f"recording-{invocation}.webm").write_bytes(video.content)
                    ids.append(current["recording_id"])
            if enabled:
                assert len(set(ids)) == 2
                assert (
                    await client.get(f"/api/browser-recordings/{ids[0]}/media")
                ).status_code == 200
            else:
                assert (
                    await client.get(f"/api/browser-recordings/sessions/{SESSION}")
                ).json() == []
            if enabled:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(headless=True)
                    try:
                        context = await browser.new_context(
                            ignore_https_errors=True,
                            http_credentials={"username": "operator", "password": password},
                        )
                        page = await context.new_page()
                        await page.goto(origin + "/operator/sessions/" + SESSION)
                        selector = page.get_by_label("Select browser recording", exact=True)
                        await expect(selector).to_be_visible(timeout=20000)
                        await selector.select_option(ids[0])
                        player = page.locator("video")
                        await player.evaluate("video => video.play()")
                        await page.wait_for_function(
                            "() => { const v = document.querySelector('video'); return v && v.readyState >= 2 && Number.isFinite(v.duration) && v.duration > 0 }"
                        )
                        await page.screenshot(path=str(root / "playback.png"), full_page=True)
                    finally:
                        await browser.close()
            capture_secret = json.loads((root / "capture-configuration.json").read_text())[
                "credential"
            ].encode()
            for path in root.glob("sessions.sqlite*"):
                assert capture_secret not in path.read_bytes()
            assert not tuple((root / "ordinary-artifacts").rglob("*.webm"))
            evidence = {
                "recording_ids": ids,
                "no_live_viewer": True,
                "two_completed_invocations": True,
                "explicit_close": explicit_close,
                "enabled": enabled,
                "worker_restart": restart,
                "dashboard_playback": enabled,
            }
            (root / "evidence.json").write_text(json.dumps(evidence, indent=2))
            return evidence
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        docker("stop", "--time", "1", container, check=False)
        # Only allocations and networks in this invocation's private ownership journal.
        for path in (root / "ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            identifier = journal.get("identity", {}).get("container_id")
            if identifier:
                docker("rm", "-f", identifier, check=False)
            if journal.get("sidecar_id"):
                docker("rm", "-f", journal["sidecar_id"], check=False)
            if journal.get("network_id"):
                docker(
                    "network",
                    "disconnect",
                    "--force",
                    journal["network_id"],
                    container,
                    check=False,
                )
                docker("network", "rm", journal["network_id"], check=False)
        docker("rm", "-f", container, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--explicit-close", action="store_true")
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run(
                    args.state.resolve(),
                    explicit_close=args.explicit_close,
                    enabled=not args.disabled,
                    restart=args.restart,
                )
            ),
            indent=2,
        )
    )

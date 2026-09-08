"""Run and verify the real local Docker/server/operator-UI composition.

Requires the locally built application image, the pinned browser image, a built
Dashboard and host Python with cayu[server,browser], cryptography and Playwright.
No provider account, external portal, or raw frame export is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import secrets
import shlex
import sqlite3
import ssl
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from playwright.async_api import async_playwright, expect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin, Subprotocol

from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD

REPO = Path(__file__).resolve().parents[2]
SESSION = "viewer-reconnect-demo"


def docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], text=True, capture_output=True, check=check, timeout=30
    ).stdout.strip()


def certificates(root):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cayu disposable local demo")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("cayu-control"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    (root / "certificate.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (root / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(root / "key.pem", 0o600)
    return base64.b64encode(
        hashlib.sha256(
            key.public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        ).digest()
    ).decode()


async def until(predicate, *, timeout=90):
    async with asyncio.timeout(timeout):
        while True:
            result = await predicate()
            if result:
                return result
            await asyncio.sleep(0.1)


async def run(root, *, keep_server=False):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    (root / "staging").mkdir(mode=0o700)
    pin = certificates(root)
    control = docker(
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
        f"CAYU_DEMO_STATE={root}",
        "--env",
        f"TMPDIR={root / 'staging'}",
        "--env",
        f"PYTHONPATH={REPO / 'src'}",
        "-p",
        "127.0.0.1::8443",
        "cayu-view-reconnect:local",
    )
    published = json.loads(docker("inspect", control))[0]["NetworkSettings"]["Ports"]["8443/tcp"][
        0
    ]["HostPort"]
    origin = f"https://127.0.0.1:{published}"
    config = {
        "control_server_container_id": control,
        "origin": origin,
        "password": secrets.token_urlsafe(32),
        "viewer_key": secrets.token_hex(32),
        "review_key": secrets.token_hex(32),
    }
    (root / "configuration.json").write_text(json.dumps(config))
    os.chmod(root / "configuration.json", 0o600)
    tls = ssl.create_default_context(cafile=str(root / "certificate.pem"))
    evidence = {
        "docker_version": docker("version", "--format", "{{.Server.Version}}"),
        "docker_context": docker("context", "show"),
        "runtime_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "browser_image": PINNED_BROWSER_SESSION_WORKLOAD.image,
        "browser_image_id": docker(
            "image", "inspect", PINNED_BROWSER_SESSION_WORKLOAD.image, "--format", "{{.Id}}"
        ),
        "application_image_id": docker(
            "image", "inspect", "cayu-view-reconnect:local", "--format", "{{.Id}}"
        ),
        "control_server_container_id": control,
    }
    print("Starting protected application worker.", flush=True)

    requests = set()
    qualified = False

    def track(operation):
        task = asyncio.create_task(operation)
        requests.add(task)
        return task

    def stop_worker():
        worker_path = root / "worker.json"
        if not worker_path.exists():
            return
        worker = json.loads(worker_path.read_text())
        docker(
            "exec",
            control,
            "python",
            "-c",
            "import os,signal,sys; from pathlib import Path; pid=int(sys.argv[1]); "
            "assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19] == sys.argv[2]; "
            "os.kill(pid,signal.SIGKILL)",
            str(worker["pid"]),
            worker["start_time"],
        )

    def start():
        docker(
            "exec",
            "-d",
            control,
            "sh",
            "-c",
            f"exec python {shlex.quote(str(REPO / 'examples/browser_view_reconnect/app.py'))} > {shlex.quote(str(root / 'worker.log'))} 2>&1",
        )

    async def healthy(client):
        try:
            return (await client.get("/api/health")).status_code == 200
        except httpx.TransportError:
            return False

    async def stage(name, task):
        async def ready():
            if task.done():
                response = task.result()
                assert response.status_code == 200, (
                    f"Control request failed ({response.status_code}); inspect private application state."
                )
                raise AssertionError(
                    f"Invocation ended before {name}; inspect private worker/state files."
                )
            return (root / f"{name}.json").exists()

        await until(ready)

    async def finish(task):
        response = await asyncio.wait_for(task, 90)
        assert response.status_code == 200, (
            f"Control request failed ({response.status_code}); inspect private application state."
        )
        assert '"session.failed"' not in response.text, "Session failed; inspect its private state."

    async def review(client, kind):
        async def load():
            response = await client.get(
                f"/api/sessions/{SESSION}/human-review", params={"purpose": "demo"}
            )
            assert response.status_code == 200
            view = response.json()
            return view if view["kind"] == kind and view["status"] == "permitted" else None

        return await until(load)

    async def open_view(panel):
        await panel.reload()
        await panel.get_by_role("button", name="Discover browsers", exact=True).click()
        await panel.get_by_role("button", name="Browser 1 · agent_controlled", exact=True).click()
        await panel.get_by_role("button", name="View page 1", exact=True).click()
        await expect(
            panel.get_by_text("Live private view. Viewing grants no input authority.", exact=True)
        ).to_be_visible(timeout=20000)
        return panel.get_by_label("Private live browser frame", exact=True)

    async def pixel(canvas):
        assert await canvas.count() == 1, "Private viewer canvas is unavailable."
        # Read a synthetic solid-color patch inside the admitted browser's live
        # canvas. Never persist screenshots or pixels to artifacts/evidence.
        return await canvas.evaluate(
            "c => Array.from(c.getContext('2d').getImageData(2,2,1,1).data).join(',')"
        )

    async def changed(canvas, prior):
        async def check():
            current = await pixel(canvas)
            return current if current != prior and not current.endswith(",0") else None

        return await until(check, timeout=20)

    async def cleared(panel, canvas):
        await expect(
            panel.get_by_text(
                "Private view unavailable or disconnected. Rediscover the browser and reopen its view.",
                exact=True,
            )
        ).to_be_visible(timeout=15000)
        assert (await pixel(canvas)).endswith(",0")

    try:
        start()
        async with httpx.AsyncClient(
            base_url=origin,
            verify=tls,
            auth=("operator", config["password"]),
            trust_env=False,
            timeout=180,
        ) as client:
            await until(lambda: healthy(client))
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    args=[f"--ignore-certificate-errors-spki-list={pin}"]
                )
                context = await browser.new_context(
                    http_credentials={
                        "username": "operator",
                        "password": config["password"],
                        "origin": origin,
                    }
                )
                panel = await context.new_page()
                invocation = track(
                    client.post(
                        "/api/run",
                        json={
                            "session_id": SESSION,
                            "prompt": "Show the synthetic portal, ask me, then propose the exact demo mutation.",
                            "limits": {"max_elapsed_seconds": 300},
                        },
                    )
                )
                await stage("before", invocation)
                await panel.goto(origin + "/operator/sessions/" + SESSION)
                canvas = await open_view(panel)
                initial = await pixel(canvas)
                before = json.loads((root / "before.json").read_text())
                assert (
                    await panel.get_by_test_id("selected-browser-identity").inner_text()
                    == before["session_id"]
                )
                allocation_fingerprint = await panel.get_by_test_id(
                    "selected-browser-allocation"
                ).inner_text()
                # Capture one unused ticket privately; a fresh owner must reject it.
                session = await client.post("/api/browser-control/operator-session")
                client.headers["X-Cayu-Browser-Operator"] = session.json()["operator_session_token"]
                record = (await client.get(f"/api/browser-control/sessions/{SESSION}")).json()[
                    "browsers"
                ][0]
                pages = (
                    await client.post(
                        "/api/browser-control/pages",
                        json={
                            "identity": record["identity"],
                            "expected_record_revision": record["revision"],
                        },
                    )
                ).json()["pages"]
                ticket_response = await client.post(
                    "/api/browser-control/view-ticket",
                    json={
                        "identity": record["identity"],
                        "expected_record_revision": record["revision"],
                        "page": pages[0],
                    },
                )
                assert ticket_response.status_code == 200
                old_ticket = ticket_response.json()["ticket"]
                (root / "before.continue").touch()
                await stage("before-changed", invocation)
                # Model mutations invalidate page authority. Reauthorize the
                # exact changed page through the existing UI before capturing.
                canvas = await open_view(panel)
                await changed(canvas, initial)
                (root / "before-changed.continue").touch()
                await finish(invocation)
                await review(client, "user_input")
                await cleared(panel, canvas)
                allocation = json.loads((root / "allocation.json").read_text())["identity"]
                assert json.loads(docker("inspect", allocation["container_id"]))[0]["State"][
                    "Running"
                ]
                print(
                    "Live changing frames verified; durable human-input pause retained the browser.",
                    flush=True,
                )
                old_pid = json.loads((root / "worker.json").read_text())["pid"]
                stop_worker()
                start()
                await until(lambda: healthy(client))
                assert json.loads((root / "worker.json").read_text())["pid"] != old_pid
                client.headers.pop("X-Cayu-Browser-Operator")
                view = await review(client, "user_input")
                resume = track(
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
                await stage("after", resume)
                after = json.loads((root / "after.json").read_text())
                assert (
                    after["page_id"] == before["page_id"]
                    and after["session_id"] == before["session_id"]
                )
                assert after["control_epoch"] > before["control_epoch"]
                canvas = await open_view(panel)
                assert (
                    await panel.get_by_test_id("selected-browser-allocation").inner_text()
                    == allocation_fingerprint
                )
                async with connect(
                    origin.replace("https:", "wss:") + "/api/browser-control/viewer",
                    ssl=tls,
                    origin=Origin(origin),
                    subprotocols=[Subprotocol("cayu.browser-view.v1")],
                    additional_headers={
                        "Authorization": "Basic "
                        + base64.b64encode(f"operator:{config['password']}".encode()).decode()
                    },
                    proxy=None,
                    compression=None,
                ) as socket:
                    await socket.send(old_ticket)
                    try:
                        await asyncio.wait_for(socket.recv(), 5)
                        raise AssertionError("Old viewer ticket was accepted.")
                    except ConnectionClosed:
                        pass
                # Authentication and a live view still grant no native input.
                session = await client.post("/api/browser-control/operator-session")
                client.headers["X-Cayu-Browser-Operator"] = session.json()["operator_session_token"]
                record = (await client.get(f"/api/browser-control/sessions/{SESSION}")).json()[
                    "browsers"
                ][0]
                pages = (
                    await client.post(
                        "/api/browser-control/pages",
                        json={
                            "identity": record["identity"],
                            "expected_record_revision": record["revision"],
                        },
                    )
                ).json()["pages"]
                now = time.time_ns() // 1_000_000
                denied = await client.post(
                    "/api/browser-control/takeover",
                    json={
                        "request_id": "bt_" + "a" * 32,
                        "identity": record["identity"],
                        "expected_record_revision": record["revision"],
                        "expected_control_epoch": record["control_epoch"],
                        "pages": pages,
                        "purpose_code": "demo",
                        "requested_at_ms": now,
                        "expires_at_ms": now + 30_000,
                        "maximum_until_ms": now + 60_000,
                        "checkpoint_consent": "deny",
                    },
                )
                assert denied.status_code == 403
                (root / "revoke-view").touch()
                await cleared(panel, canvas)
                (root / "revoke-view").unlink()
                canvas = await open_view(panel)
                current = await pixel(canvas)
                (root / "after.continue").touch()
                await stage("after-changed", resume)
                canvas = await open_view(panel)
                await changed(canvas, current)
                (root / "after-changed.continue").touch()
                await finish(resume)
                await cleared(panel, canvas)
                view = await review(client, "tool_approval")
                assert view["fields"] == [
                    {"label": "Proposal", "text": "Commit demo-proposal with value approved."}
                ]
                assert not (root / "portal.sqlite").exists()
                decision = {
                    "session_id": SESSION,
                    "approval_id": view["interaction_id"],
                    "tool_round_id": view["tool_round_id"],
                    "tool_call_id": view["tool_call_id"],
                    "decision": "approve",
                    "review_reference": view["reference"],
                }
                await finish(track(client.post("/api/tool-approvals/resolve", json=decision)))
                with sqlite3.connect(root / "portal.sqlite") as portal:
                    assert portal.execute("SELECT * FROM receipts").fetchall() == [
                        ("demo-proposal", "approved")
                    ]
                assert json.loads((root / "business-receipt.json").read_text())["committed"]
                await browser.close()
                assert json.loads(docker("inspect", control))[0]["State"]["Running"]
                for kind, identifier in [
                    ("container", allocation["container_id"]),
                    ("network", allocation["network_id"]),
                ]:
                    assert (
                        not docker("inspect", "--type", kind, identifier, check=False)
                        or docker("inspect", "--type", kind, identifier, check=False) == "[]"
                    )
                journal = json.loads(
                    (root / "ownership" / f"{allocation['allocation_id']}.json").read_text()
                )
                assert journal["state"] == "disposed"
                evidence.update(
                    same_browser_container=True,
                    same_page=True,
                    fresh_control_epoch=True,
                    changing_frames_before=True,
                    changing_frames_after=True,
                    truthful_disconnect=True,
                    old_ticket_rejected=True,
                    authorization_revocation=True,
                    view_only_denies_takeover=True,
                    protected_human_input=True,
                    protected_tool_approval=True,
                    independent_mutation_count=1,
                    allocation_cleanup=True,
                    application_container_survived=True,
                    browser_container_id=allocation["container_id"],
                    page_id=before["page_id"],
                )
                (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
                qualified = True
                print(
                    "PASS: same browser/page, fresh protected live view, one approved mutation, exact cleanup.",
                    flush=True,
                )
    finally:
        for request in requests:
            if not request.done():
                request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        if not qualified:
            # Stop the exact fixture container before emergency disposal so no
            # worker can race new Docker work against the harness's cleanup.
            docker("stop", "--time", "1", control, check=False)
        # Emergency cleanup is restricted to the private directory created above,
        # exact journal IDs, and this harness's application container.
        for path in (root / "ownership").glob("*.json"):
            journal = json.loads(path.read_text())
            for identifier in (
                journal.get("identity", {}).get("container_id"),
                journal.get("sidecar_id"),
            ):
                if identifier:
                    docker("rm", "-f", identifier, check=False)
            network = journal.get("network_id")
            if network:
                docker("network", "disconnect", "--force", network, control, check=False)
                docker("network", "rm", network, check=False)
        if not keep_server:
            docker("rm", "-f", control, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="New private directory on this Docker host, mounted at the same absolute path.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep the owned application container after verifying allocation cleanup.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.state_dir.absolute(), keep_server=args.keep_server))

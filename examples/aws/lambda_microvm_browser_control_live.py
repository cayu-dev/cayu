"""Live operator view, takeover, and recording on an AWS Lambda MicroVM.

Run this from a host inside the egress connector's VPC, such as an ECS task,
with FFmpeg installed: recordings are encoded on the host. The host runs the
protected Cayu server with TLS on its private address, a Cayu CONNECT proxy,
and a deterministic fixture origin that the proxy maps to ``docs.browser.test``.

The browser MicroVM is created by ``VirtualEgressEnvironmentFactory`` from the
browser variant of the first-party image with ``browser_workload=True`` and a
``LambdaMicroVMBrowserControlRelay`` naming this host. Before any tool runs, the
MicroVM proves the pinned worker, launches Chromium with its sandbox, installs
the control CA, and completes a TLS handshake with this server through the
sidecar relay as the agent user. The guest dials only
``wss://cayu-control:18443/...``; TLS is verified end to end with a certificate
for ``cayu-control``. The scenario proves that:

- an operator receives a live view frame of the admitted page;
- an operator takes over, enters private input into the page, and hands back,
  after which the agent observes the same page again without receiving that input;
- the session is recorded and finalized into a WebM video that FFmpeg decodes;
- Chromium keeps its sandbox as UID 1000 throughout.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import http.server
import ipaddress
import json
import os
import secrets
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from examples._live_checks import require
from examples.aws.lambda_microvm_browser_live import (
    _CountingUpstream,
    _last_browser_state,
    _private_address,
    _SandboxProbeTool,
    _tool_results,
)
from pydantic import SecretBytes, SecretStr

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._browser_recording_store import BrowserRecordingStore
from cayu.artifacts.local import LocalArtifactStore
from cayu.browser_recording import BrowserRecordingPolicy
from cayu.egress.aws_lambda_microvm_adapter import (
    LambdaMicroVMBrowserControlRelay,
    LambdaMicroVMEgressAdapter,
)
from cayu.egress.broker import HttpxUpstream
from cayu.egress.destinations import ApprovedEgressDestination
from cayu.egress.policy import BrowserEgressPolicy
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.egress.runtime import VirtualEgressEnvironmentFactory
from cayu.environments.base import EnvironmentSpec
from cayu.evals.internal.browser_acceptance_operator import perform_fixture_handoff
from cayu.events import EventType
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners.workloads import PINNED_BROWSER_FETCH_IMAGE, PINNED_BROWSER_SESSION_WORKLOAD
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import BasicAuth, BrowserControlServerConfig, ServerConfig, create_server
from cayu.server.browser_recording import BrowserRecordingServer
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import WebBridge
from cayu.tools.browser_control import (
    BrowserControlPolicy,
    BrowserControlPolicyResult,
    BrowserOperatorPurpose,
)
from cayu.tools.browser_control_config import BrowserControlConfig
from cayu.tools.browser_session import BrowserSessionTool

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_HOST = "docs.browser.test"
_ORIGIN = f"https://{_HOST}"
_SESSION_ID = "lambda-browser-control-live"
_MAXIMUM_DURATION_SECONDS = 900
_SERVER_PORT = 8443
_OPERATOR_ORIGIN = "https://operator.test"
_GUEST_CONTROL = "wss://cayu-control:18443/api/browser-control/guest"
_GUEST_RECORDING = "wss://cayu-control:18443/api/browser-recordings/guest"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBM_MAGIC = b"\x1aE\xdf\xa3"
_PRIVATE_ENTRY = "lambda-operator-private-entry"
_PURPOSE = BrowserOperatorPurpose(code="lambda_live", expected_origins=(_ORIGIN,))


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/form":
            self.send_response(404)
            self.end_headers()
            return
        body = b"""<!doctype html>
<html><head><title>Lambda operator fixture</title></head>
<body style="background:#0b5;font:28px sans-serif">
<h1>Operator handoff fixture</h1>
<label>Private entry <input id="entry" type="text" autocomplete="off"></label>
</body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _certificates(address: str) -> tuple[bytes, bytes, bytes]:
    """A disposable CA and a server certificate for ``cayu-control`` and ``address``."""

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cayu live control CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    server = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cayu-control")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=2))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("cayu-control"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    x509.IPAddress(ipaddress.IPv4Address(address)),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    return (
        ca.public_bytes(serialization.Encoding.PEM),
        server.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


class _OperatorPolicy(BrowserControlPolicy):
    identity = "examples.lambda-browser-control:v1"

    async def decide(self, request) -> BrowserControlPolicyResult:
        return BrowserControlPolicyResult(
            allowed=(
                request.principal.subject == "operator"
                and request.identity.session_id == _SESSION_ID
                and request.identity.operator_purpose == _PURPOSE
            )
        )


class _Coordination:
    def __init__(self) -> None:
        self.navigated = asyncio.Event()
        self.operator_done = asyncio.Event()
        self.state: dict | None = None


class _OperatorModel(ModelProvider):
    """Navigate, wait for the operator's handoff, then observe and close."""

    name = "lambda-browser-control-script"

    def __init__(self, coordination: _Coordination) -> None:
        self.coordination = coordination

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples.lambda_microvm_browser_control.model",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        messages = [message.model_dump(mode="json") for message in request.messages]
        completed = sum(1 for message in messages if message.get("role") == "tool")
        if completed == 0:
            call = (
                "browser_session",
                {"operation": "navigate", "url": f"{_ORIGIN}/form", "operation_id": "nav-1"},
            )
        elif completed == 1:
            self.coordination.state = _last_browser_state(messages)
            call = ("chromium_sandbox_probe", {})
        elif completed == 2:
            self.coordination.navigated.set()
            # The operator views, takes over, types, and hands back meanwhile.
            async with asyncio.timeout(300):
                await self.coordination.operator_done.wait()
            state = self.coordination.state
            assert state is not None
            call = (
                "browser_session",
                {
                    "operation": "observe",
                    "session_id": state["session_id"],
                    "page_id": state["page_id"],
                    "operation_id": "observe-1",
                },
            )
        elif completed == 3:
            state = _last_browser_state(messages)
            call = (
                "browser_session",
                {
                    "operation": "close",
                    "session_id": state["session_id"],
                    "operation_id": "close-1",
                },
            )
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id=f"call-{completed}", name=call[0], arguments=call[1])
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


def _configuration() -> dict[str, str]:
    if os.environ.get("CAYU_LAMBDA_MICROVM_BROWSER_CONTROL_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_BROWSER_CONTROL_LIVE=1 to run this contract.")
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    connector = os.environ.get("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    if not image.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE to a browser-variant image ARN.")
    if not connector.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR to the VPC egress connector.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("Install FFmpeg on this host; recordings are encoded here.")
    return {"image": image, "connector": connector, "region": region}


async def _view_one_frame(client, tls: ssl.SSLContext, password: str) -> bytes:
    """Authorize and receive one live frame of the agent's page, then retire the view."""

    from websockets.asyncio.client import connect
    from websockets.typing import Origin, Subprotocol

    session = (await client.post("/api/browser-control/operator-session")).json()
    headers = {"X-Cayu-Browser-Operator": session["operator_session_token"]}
    async with asyncio.timeout(60):
        while True:
            records = (
                await client.get(f"/api/browser-control/sessions/{_SESSION_ID}", headers=headers)
            ).json()["browsers"]
            if records and records[0].get("state") == "agent_controlled":
                record = records[0]
                break
            await asyncio.sleep(0.25)
    pages = (
        await client.post(
            "/api/browser-control/pages",
            headers=headers,
            json={"identity": record["identity"], "expected_record_revision": record["revision"]},
        )
    ).json()["pages"]
    ticket = await client.post(
        "/api/browser-control/view-ticket",
        headers=headers,
        json={
            "identity": record["identity"],
            "expected_record_revision": record["revision"],
            "page": pages[0],
        },
    )
    require(ticket.status_code == 200, f"view ticket was refused: {ticket.status_code}")
    async with connect(
        f"wss://127.0.0.1:{_SERVER_PORT}/api/browser-control/viewer",
        ssl=tls,
        origin=Origin(_OPERATOR_ORIGIN),
        subprotocols=[Subprotocol("cayu.browser-view.v1")],
        additional_headers={
            "Authorization": "Basic " + base64.b64encode(f"operator:{password}".encode()).decode()
        },
        proxy=None,
        compression=None,
        max_size=4 * 1024 * 1024,
    ) as viewer:
        await viewer.send(ticket.json()["ticket"])
        require(await viewer.recv() == "ready", "operator viewer was not admitted")
        await viewer.send("frame")
        frame = await asyncio.wait_for(viewer.recv(), 30)
        if not isinstance(frame, bytes):
            raise SystemExit(f"operator viewer returned {frame!r}")
        await viewer.send("retire")
        purge = await asyncio.wait_for(viewer.recv(), 10)
        if isinstance(purge, str) and purge.startswith("purge:"):
            await viewer.send("purged:" + purge.removeprefix("purge:"))
            await asyncio.wait_for(viewer.recv(), 10)
    return frame


def _decoded_frames(video: bytes, directory: Path) -> int:
    path = directory / "recording.webm"
    path.write_bytes(video)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    (stream,) = json.loads(probe.stdout)["streams"]
    require(stream["codec_name"] in {"vp8", "vp9", "av1"}, f"unexpected codec: {stream}")
    return int(stream["nb_read_frames"])


async def main() -> None:
    import httpx
    import uvicorn

    config = _configuration()
    started = time.monotonic()
    host_address = _private_address()
    workdir = tempfile.TemporaryDirectory(prefix="cayu-lambda-browser-control-")
    root = Path(workdir.name)
    ca, certificate, key = _certificates(host_address)
    (root / "server.pem").write_bytes(certificate + ca)
    (root / "server.key").write_bytes(key)
    tls = ssl.create_default_context(cadata=ca.decode("ascii"))
    password = secrets.token_urlsafe(24)

    origin = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _FixtureHandler)
    origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
    origin_thread.start()
    upstream = _CountingUpstream(
        HttpxUpstream(
            routes={_HOST: f"http://{host_address}:{origin.server_port}"},
            max_response_bytes=1024 * 1024,
        )
    )
    recordings = BrowserRecordingStore(root / "recordings.sqlite")
    recording = await recordings.authorize_capture(
        session_id=_SESSION_ID,
        policy=BrowserRecordingPolicy(
            scope="lambda-live",
            allowed_origins=(_ORIGIN,),
            retention_seconds=3600,
            max_duration_seconds=600,
        ),
        guest_endpoint=_GUEST_RECORDING,
    )
    artifact_store = LocalArtifactStore(root / "artifacts", store_id="lambda-browser-control-live")
    factory = VirtualEgressEnvironmentFactory(
        policies={"browser-live": BrowserEgressPolicy(name="browser-live", allowed_hosts=(_HOST,))},
        approved_destinations=[
            ApprovedEgressDestination(destination=_HOST, policy_name="browser-live")
        ],
        credentials=[],
        adapter=LambdaMicroVMEgressAdapter(
            region_name=config["region"],
            egress_network_connector_arn=config["connector"],
            exposure=VpcTaskProxyExposure(host_address),
            browser_workload=True,
            browser_control_relay=LambdaMicroVMBrowserControlRelay(
                host=host_address, port=_SERVER_PORT, ca_certificate_pem=ca
            ),
            runner_options={"maximum_duration_in_seconds": _MAXIMUM_DURATION_SECONDS},
        ),
        image=config["image"],
        artifact_store=artifact_store,
        upstream=upstream,
    )
    coordination = _Coordination()
    app = CayuApp(
        session_store=SQLiteSessionStore(root / "sessions.sqlite"),
        enable_logging=False,
        browser_control=BrowserControlConfig(
            policy=_OperatorPolicy(), purpose=_PURPOSE, guest_endpoint=_GUEST_CONTROL
        ),
    )
    session_bridge = WebBridge.sandboxed_browser(
        environment=factory, browser_image=PINNED_BROWSER_FETCH_IMAGE, interactive=True
    )
    (bridge_tool,) = session_bridge.tools
    app.register_provider(_OperatorModel(coordination), default=True)
    app.register_environment_factory(EnvironmentSpec(name="browser"), factory, default=True)
    app.register_agent(
        AgentSpec(name="browser-agent", model="scripted"),
        tools=[
            BrowserSessionTool(
                expected_runner_candidate="lambda-microvm",
                expected_environment_authority=bridge_tool.expected_environment_authority,
                expected_workload_authority=PINNED_BROWSER_SESSION_WORKLOAD,
                expected_artifact_store_id=bridge_tool.expected_artifact_store_id,
                recording=recording,
            ),
            _SandboxProbeTool(),
        ],
        execution_requirements=session_bridge.execution_requirements,
    )

    async def recording_access(principal, manifest, purpose) -> bool:
        del purpose
        return principal.subject == "operator" and manifest.identity.session_id == _SESSION_ID

    server = uvicorn.Server(
        uvicorn.Config(
            create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password=password),
                    browser_control=BrowserControlServerConfig(
                        operator_origin=_OPERATOR_ORIGIN,
                        signing_key=SecretBytes(secrets.token_bytes(32)),
                    ),
                ),
                browser_recordings=BrowserRecordingServer(
                    store=recordings, authorize=recording_access
                ),
            ),
            host="0.0.0.0",
            port=_SERVER_PORT,
            ssl_certfile=str(root / "server.pem"),
            ssl_keyfile=str(root / "server.key"),
            ws="websockets-sansio",
            ws_max_size=2 * 1024 * 1024 + 4096,
            ws_per_message_deflate=False,
            access_log=False,
            log_level="warning",
            timeout_graceful_shutdown=5,
        )
    )
    server_task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if server_task.done():
                    await server_task
                    raise SystemExit("control server exited before readiness")
                await asyncio.sleep(0.05)

        async def run() -> list:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="browser-agent",
                        session_id=_SESSION_ID,
                        messages=[Message.text("user", "Open the form for the operator.")],
                    )
                )
            ]

        run_task = asyncio.create_task(run())
        async with httpx.AsyncClient(
            base_url=f"https://127.0.0.1:{_SERVER_PORT}",
            verify=tls,
            auth=httpx.BasicAuth("operator", password),
            trust_env=False,
        ) as client:
            waiter = asyncio.create_task(coordination.navigated.wait())
            done, _pending = await asyncio.wait(
                {waiter, run_task}, timeout=600, return_when=asyncio.FIRST_COMPLETED
            )
            if waiter not in done:
                waiter.cancel()
                events = run_task.result() if run_task.done() else []
                failed = [
                    event.payload for event in events if event.type is EventType.SESSION_FAILED
                ]
                raise SystemExit(f"browser session never opened: {failed or 'timeout'}")
            state = coordination.state
            if state is None:
                raise SystemExit("navigate did not report browser state")
            frame = await _view_one_frame(client, tls, password)
            require(frame.startswith(_PNG_MAGIC), "operator view frame is not a PNG")
            await perform_fixture_handoff(
                client=client,
                input_endpoint=f"wss://127.0.0.1:{_SERVER_PORT}/api/browser-control/input",
                tls=tls,
                operator_origin=_OPERATOR_ORIGIN,
                session_id=_SESSION_ID,
                browser_session_id=state["session_id"],
                private_text=SecretStr(_PRIVATE_ENTRY),
            )
            coordination.operator_done.set()
            events = await asyncio.wait_for(run_task, 600)
        failed = next((event for event in events if event.type is EventType.SESSION_FAILED), None)
        require(failed is None, f"session failed: {failed.payload if failed else None}")
        require(
            EventType.SESSION_COMPLETED in {event.type for event in events},
            f"session did not complete: {[event.type.value for event in events[-6:]]}",
        )
        results = _tool_results(events)
        require(len(results) == 4, f"expected four tool results, got {len(results)}")
        for index, result in enumerate(results):
            require(
                not result["is_error"],
                f"tool call {index} failed: {result['content'][:800]} "
                f"{json.dumps(result.get('structured'), sort_keys=True)[:2000]}",
            )
        _navigated, probe, observed, _closed = results
        sandbox = probe["structured"]
        require(
            sandbox["processes"] > 0
            and sandbox["no_sandbox_flags"] == 0
            and sandbox["all_uid_1000"]
            and sandbox["all_no_new_privs"]
            and sandbox["renderers"] == sandbox["renderers_seccomp_filtered"] > 0,
            f"Chromium was not sandboxed as the agent: {sandbox}",
        )
        # After handback the agent observes the same page again, but the
        # operator's private entry is withheld from the model.
        require(
            f"URL: {_ORIGIN}/form" in observed["content"],
            f"the agent's fresh observation lost the page: {observed['content'][:800]}",
        )
        require(
            _PRIVATE_ENTRY not in observed["content"]
            and _PRIVATE_ENTRY not in json.dumps(observed.get("structured")),
            "the operator's private input reached the model",
        )
        require(await app.drain_environment_cleanups(timeout_s=120), "allocation cleanup unsettled")

        (recording_id,) = await recordings.recordings_for_session(_SESSION_ID)
        async with asyncio.timeout(120):
            while (manifest := await recordings.manifest(recording_id)).status == "recording":
                await asyncio.sleep(0.5)
        require(
            manifest.status in {"complete", "partial"},
            f"recording did not finalize: {manifest.status} {manifest.reason}",
        )
        require(bool(manifest.segments), "recording has no segments")
        video = await recordings.video(recording_id)
        require(video.startswith(_WEBM_MAGIC), "recording is not a WebM file")
        frames = _decoded_frames(video, root)
        require(frames > 0, "FFmpeg decoded no recorded frames")

        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "lambda-microvm",
                    "region": config["region"],
                    "control_path": "wss://cayu-control:18443 via sidecar relay, TLS verified",
                    "operator_view_frame_png_bytes": len(frame),
                    "takeover": "sensitive tab+text settled, handed back",
                    "fresh_observation_after_handback": "verified",
                    "operator_private_input_withheld_from_model": True,
                    "recording_status": manifest.status,
                    "recording_reason": manifest.reason,
                    "recording_segments": len(manifest.segments),
                    "recording_webm_bytes": len(video),
                    "recording_decoded_frames": frames,
                    "chromium_sandbox": "enabled",
                    "renderers_seccomp_filtered": sandbox["renderers_seccomp_filtered"],
                    "agent_uid": 1000,
                    "proxy_requests": len(upstream.paths),
                    "seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            )
        )
    finally:
        with contextlib.suppress(Exception):
            await app.drain_environment_cleanups(timeout_s=120)
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server_task, 15)
        origin.shutdown()
        origin.server_close()
        origin_thread.join(timeout=5)
        workdir.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

"""Live browser contract on an AWS Lambda MicroVM through the Cayu proxy.

Run this from a host inside the egress connector's VPC, such as an ECS task:
the MicroVM reaches the Cayu CONNECT proxy on this host's private address and
nothing else. The host also serves a deterministic fixture origin that the
proxy maps to ``docs.browser.test``.

One ``CayuApp`` session runs on a MicroVM that ``VirtualEgressEnvironmentFactory``
allocates from the browser variant of the first-party image with
``browser_workload=True``. The scenario proves that:

- the factory binds the pinned browser workload before any allocation, and the
  created MicroVM proves it before any tool is exposed: a trusted probe of its
  worker sources, then one sandboxed Chromium launch as the agent user;
- ``web_fetch`` renders a JavaScript page in Chromium and ``screenshot_page``
  stores a PNG, with every request crossing the Cayu proxy under the session
  CA;
- an interactive browser session navigates, screenshots, and closes, and while
  it is open every Chromium process runs as UID 1000 with renderers confined
  by Chromium's own seccomp sandbox and no ``--no-sandbox`` flag.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import socket
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from examples._live_checks import require

from cayu import AgentSpec, CayuApp, ExecCommand, Message, RunRequest
from cayu.artifacts.local import LocalArtifactStore
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.broker import (
    CapturedRequest,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpxUpstream,
)
from cayu.egress.destinations import ApprovedEgressDestination
from cayu.egress.policy import BrowserEgressPolicy
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.egress.runtime import VirtualEgressEnvironmentFactory
from cayu.environments.base import EnvironmentSpec
from cayu.events import EventType
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners.workloads import PINNED_BROWSER_FETCH_IMAGE
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools import WebBridge
from cayu.tools.base import Tool, ToolResult, ToolSpec

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_HOST = "docs.browser.test"
_SESSION_ID = "lambda-browser-live"
_MAXIMUM_DURATION_SECONDS = 900
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Runs as the agent while the interactive session is open. It reports only
# counts and booleans about Chromium processes owned by the agent user.
_SANDBOX_PROBE = r"""
import json, os
chrome = []
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            cmdline = handle.read().replace(b'\0', b' ')
        with open(f'/proc/{pid}/status') as handle:
            status = dict(line.split(':\t', 1) for line in handle.read().splitlines() if ':\t' in line)
    except OSError:
        continue
    # Chromium rewrites each child's command line into one string, so the
    # process type is found anywhere in it.
    if b'headless_shell' not in cmdline.split(b' ', 1)[0]:
        continue
    chrome.append({
        'renderer': b'--type=renderer' in cmdline,
        'no_sandbox': b'--no-sandbox' in cmdline,
        'uids': status.get('Uid', '').split(),
        'seccomp': status.get('Seccomp', '').strip(),
        'no_new_privs': status.get('NoNewPrivs', '').strip(),
    })
print(json.dumps({
    'processes': len(chrome),
    'renderers': sum(1 for item in chrome if item['renderer']),
    'renderers_seccomp_filtered': sum(
        1 for item in chrome if item['renderer'] and item['seccomp'] == '2'
    ),
    'no_sandbox_flags': sum(1 for item in chrome if item['no_sandbox']),
    'all_uid_1000': all(set(item['uids']) == {'1000'} for item in chrome),
    'all_no_new_privs': all(item['no_new_privs'] == '1' for item in chrome),
}))
"""


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/page":
            body = b"""<!doctype html>
<html><head><title>Lambda fixture before script</title></head>
<body><main id="content">not rendered</main>
<script src="https://docs.browser.test/render.js"></script>
</body></html>"""
            self._send("text/html; charset=utf-8", body)
            return
        if self.path == "/render.js":
            self._send(
                "text/javascript; charset=utf-8",
                b"document.title='Lambda browser fixture';"
                b"document.querySelector('#content').textContent="
                b"'JavaScript rendered inside a Lambda MicroVM';",
            )
            return
        self.send_response(404)
        self.end_headers()

    def _send(self, content_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _CountingUpstream:
    def __init__(self, inner: HttpxUpstream) -> None:
        self.inner = inner
        self.paths: list[str] = []

    def prepare(
        self, request: CapturedRequest, *, limits: EgressUpstreamLimits
    ) -> EgressUpstreamOperation:
        self.paths.append(request.path)
        return self.inner.prepare(request, limits=limits)


def _private_address() -> str:
    configured = os.environ.get("CAYU_LAMBDA_MICROVM_PROXY_HOST")
    if configured:
        return configured
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        return str(probe.getsockname()[0])


def _configuration() -> dict[str, str]:
    if os.environ.get("CAYU_LAMBDA_MICROVM_BROWSER_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_BROWSER_LIVE=1 to run this contract.")
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    connector = os.environ.get("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    if not image.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE to a browser-variant image ARN.")
    if not connector.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR to the VPC egress connector.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")
    return {"image": image, "connector": connector, "region": region}


class _SandboxProbeTool(Tool):
    """Report Chromium's process confinement from inside the agent profile."""

    spec = ToolSpec(
        name="chromium_sandbox_probe",
        description="Report the confinement of running Chromium processes.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def run(self, ctx, args) -> ToolResult:
        del args
        result = await ctx.runner.exec(
            ExecCommand.process("python3", "-c", _SANDBOX_PROBE), timeout_s=30
        )
        if result.exit_code != 0:
            return ToolResult(content=result.stderr[:500], is_error=True)
        return ToolResult(content=result.stdout, structured=json.loads(result.stdout))


class _BrowserModel(ModelProvider):
    """A stateless script that reads browser state from its own tool results."""

    name = "lambda-browser-script"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples.lambda_microvm_browser.model",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        messages = [message.model_dump(mode="json") for message in request.messages]
        completed = sum(1 for message in messages if message.get("role") == "tool")
        url = f"https://{_HOST}/page"
        if completed == 0:
            call = ("web_fetch", {"url": url})
        elif completed == 1:
            call = ("screenshot_page", {"url": url})
        elif completed == 2:
            call = (
                "browser_session",
                {"operation": "navigate", "url": url, "operation_id": "nav-1"},
            )
        elif completed == 3:
            call = ("chromium_sandbox_probe", {})
        elif completed in {4, 5}:
            state = _last_browser_state(messages)
            call = (
                "browser_session",
                {
                    "operation": "screenshot",
                    "session_id": state["session_id"],
                    "page_id": state["page_id"],
                    "expected_revision": state["expected_revision"],
                    "expected_control_epoch": state["expected_control_epoch"],
                    "operation_id": "shot-1",
                }
                if completed == 4
                else {
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


def _last_browser_state(messages: list) -> dict:
    texts: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(messages)
    states = [
        match
        for text in texts
        for match in re.findall(r"<cayu_browser_state>(.*?)</cayu_browser_state>", text)
    ]
    require(bool(states), "browser_session did not report its state")
    return json.loads(states[-1])


def _tool_results(events) -> list[dict]:
    return [
        event.payload["result"]
        for event in events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]


async def main() -> None:
    config = _configuration()
    started = time.monotonic()
    host_address = _private_address()
    workdir = tempfile.TemporaryDirectory(prefix="cayu-lambda-browser-")
    artifact_store = LocalArtifactStore(
        Path(workdir.name) / "artifacts", store_id="lambda-browser-live"
    )
    origin = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _FixtureHandler)
    origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
    origin_thread.start()
    upstream = _CountingUpstream(
        HttpxUpstream(
            routes={_HOST: f"http://{host_address}:{origin.server_port}"},
            max_response_bytes=1024 * 1024,
        )
    )
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
            runner_options={"maximum_duration_in_seconds": _MAXIMUM_DURATION_SECONDS},
        ),
        image=config["image"],
        artifact_store=artifact_store,
        upstream=upstream,
    )
    try:
        # Construction admits the factory before any MicroVM exists.
        # A MicroVM fetches its root filesystem lazily. The first-party browser
        # image reads Chromium before its snapshot, but a first start on a fresh
        # MicroVM still measured from a few seconds to about a minute.
        fetch_bridge = WebBridge.sandboxed_browser(
            environment=factory,
            browser_image=PINNED_BROWSER_FETCH_IMAGE,
            fetch_options={"timeout_seconds": 120},
            screenshot_options={"timeout_seconds": 120},
        )
        session_bridge = WebBridge.sandboxed_browser(
            environment=factory, browser_image=PINNED_BROWSER_FETCH_IMAGE, interactive=True
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(_BrowserModel(), default=True)
        app.register_environment_factory(EnvironmentSpec(name="browser"), factory, default=True)
        app.register_agent(
            AgentSpec(name="browser-agent", model="scripted"),
            tools=[*fetch_bridge.tools, *session_bridge.tools, _SandboxProbeTool()],
            execution_requirements=session_bridge.execution_requirements,
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="browser-agent",
                    session_id=_SESSION_ID,
                    messages=[Message.text("user", "Read and capture the fixture page.")],
                )
            )
        ]
        failed = next((event for event in events if event.type is EventType.SESSION_FAILED), None)
        require(failed is None, f"browser session failed: {failed.payload if failed else None}")
        require(
            EventType.SESSION_COMPLETED in {event.type for event in events},
            f"browser session did not complete: {[event.type.value for event in events[-6:]]}",
        )
        results = _tool_results(events)
        require(len(results) == 6, f"expected six tool results, got {len(results)}")
        for index, result in enumerate(results):
            require(
                not result["is_error"],
                f"tool call {index} failed: {result['content'][:800]} "
                f"{json.dumps(result.get('structured'), sort_keys=True)[:2000]}",
            )
        fetched, screenshot, opened, probe, session_shot, closed = results
        require(
            "JavaScript rendered inside a Lambda MicroVM" in fetched["content"],
            f"web_fetch did not render JavaScript: {fetched['content'][:800]}",
        )
        stored = await artifact_store.read_bytes(screenshot["structured"]["artifact_id"])
        require(stored.content.startswith(_PNG_MAGIC), "screenshot artifact is not a PNG")
        require(
            "Lambda browser fixture" in opened["content"],
            f"browser session did not render the page: {opened['content'][:800]}",
        )
        sandbox = probe["structured"]
        require(sandbox["processes"] > 0, "no Chromium processes were observed")
        require(sandbox["no_sandbox_flags"] == 0, "Chromium ran with --no-sandbox")
        require(sandbox["all_uid_1000"], "a Chromium process ran outside UID 1000")
        require(sandbox["all_no_new_privs"], "a Chromium process could gain privileges")
        require(
            sandbox["renderers"] > 0
            and sandbox["renderers_seccomp_filtered"] == sandbox["renderers"],
            f"renderers were not seccomp-sandboxed: {sandbox}",
        )
        require(bool(session_shot["structured"].get("artifacts")), "no session screenshot")
        require(closed["structured"] is not None, "browser session close was not published")
        require(
            {"/page", "/render.js"} <= set(upstream.paths),
            f"browser traffic did not cross the Cayu proxy: {upstream.paths}",
        )

        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "lambda-microvm",
                    "region": config["region"],
                    "workload": "verified_worker_sources+playwright-1.62.0",
                    "web_fetch_javascript": "verified",
                    "screenshot_png_bytes": len(stored.content),
                    "browser_session": "navigate+screenshot+close",
                    "proxy_requests": len(upstream.paths),
                    "chromium_processes": sandbox["processes"],
                    "chromium_sandbox": "enabled",
                    "renderers_seccomp_filtered": sandbox["renderers_seccomp_filtered"],
                    "agent_uid": 1000,
                    "seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            )
        )
    finally:
        origin.shutdown()
        origin.server_close()
        origin_thread.join(timeout=5)
        workdir.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

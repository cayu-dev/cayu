"""Real Chromium proof for runner-backed ``web_fetch`` virtual egress."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    BrowserWebFetchAdapter,
    ExecCommand,
    ToolContext,
    WebFetchTool,
)
from cayu.egress import CapturedRequest, CapturedResponse, HttpxUpstream
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners.docker import DockerRunner
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import SecretRedactor

_BROWSER_IMAGE = os.environ.get(
    "CAYU_BROWSER_FETCH_IMAGE",
    "cayu-browser-fetch:2-playwright-1.62.0-test",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SECCOMP_PROFILE = _REPOSITORY_ROOT / "examples" / "browser_fetch" / "seccomp_profile.json"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


_DOCKER_AVAILABLE = _docker_available()
if os.environ.get("CAYU_REQUIRE_DOCKER_EGRESS") == "1" and not _DOCKER_AVAILABLE:
    raise RuntimeError("CAYU_REQUIRE_DOCKER_EGRESS=1 but the Docker daemon is unavailable.")

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        not _DOCKER_AVAILABLE,
        reason="Docker daemon not available for browser-fetch E2E.",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def browser_fetch_image() -> None:
    if os.environ.get("CAYU_BROWSER_FETCH_IMAGE"):
        return
    subprocess.run(
        [
            "docker",
            "build",
            "--file",
            "examples/browser_fetch/Dockerfile",
            "--tag",
            _BROWSER_IMAGE,
            ".",
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        timeout=600,
    )


def _host_private_address() -> str:
    """Return the host address used by the broker's deterministic HTTP origin."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        address = str(probe.getsockname()[0])
    if address.startswith("127.") or address == "0.0.0.0":
        raise RuntimeError("A non-loopback host address is required for the browser E2E.")
    return address


class _BrowserFixtureHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", "/guide")
            self.end_headers()
            return
        if self.path == "/guide":
            body = b"""<!doctype html>
<html><head><title>Fixture before script</title></head>
<body><main id="content">not rendered</main>
<script src="https://static.browser.test/render.js"></script>
</body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/render.js":
            body = (
                b"document.title='Rendered fixture';"
                b"document.querySelector('#content').textContent="
                b"'JavaScript-rendered browser content';"
            )
            self._send(200, "text/javascript; charset=utf-8", body)
            return
        if self.path == "/structured":
            body = b"""<!doctype html>
<html><head><title>Deployment console</title></head>
<body><main>
<h1>Deployment overview</h1>
<table aria-label="Release status">
<thead><tr><th>Environment</th><th>Status</th></tr></thead>
<tbody><tr><td>Production</td><td>Ready</td></tr></tbody>
</table>
<form aria-label="Deployment controls">
<label for="target">Target</label>
<select id="target"><option>Production</option></select>
<button type="button">Deploy</button>
</form>
</main></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/navigation":
            body = b"""<!doctype html>
<html><head><title>Navigation</title></head>
<body><a href="/structured">Open deployment console</a></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/shadow-controls":
            body = b"""<!doctype html>
<html><head><title>Shadow controls</title></head>
<body><div id="controls"></div><script>
const root = document.querySelector('#controls').attachShadow({mode: 'open'});
for (let index = 0; index < 8; index += 1) {
  const button = document.createElement('button');
  button.setAttribute('aria-label', `Deploy environment ${index}`);
  root.append(button);
}
</script></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/deep-accessibility":
            groups = "".join(
                (
                    f'<div id="level-{index}" role="group" '
                    f'aria-label="Level {index}"'
                    + (f' aria-owns="level-{index + 1}"></div>' if index < 39 else "></div>")
                )
                for index in range(40)
            )
            body = (
                "<!doctype html><html><head><title>Deep relationships</title></head>"
                f"<body>{groups}</body></html>"
            ).encode()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/redirect-out":
            self.send_response(302)
            self.send_header("Location", "https://blocked.browser.test/private")
            self.end_headers()
            return
        if self.path == "/redirect-http":
            self.send_response(302)
            self.send_header("Location", "http://docs.browser.test/plain")
            self.end_headers()
            return
        if self.path == "/popup":
            body = b"""<!doctype html>
<html><body><script>
window.open('https://docs.browser.test/popup-target');
</script><main>primary page</main></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/mutating-extraction":
            body = b"""<!doctype html><html><head><title>Stable inspection</title></head>
<body><button>initial control</button><script>
Object.defineProperty(document.body, 'innerText', {
  configurable: true,
  get() {
    fetch('https://static.browser.test/private/late');
    for (let index = 0; index < 100; index += 1) {
      const button = document.createElement('button');
      button.textContent = `page-controlled ${index}`;
      document.body.append(button);
    }
    return 'page-controlled replacement';
  }
});
setTimeout(() => {
  for (let index = 0; index < 100; index += 1) {
    const button = document.createElement('button');
    button.textContent = `frame-timer ${index}`;
    document.body.append(button);
  }
}, 500);
</script></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/framed":
            body = b"""<!doctype html><html><head><title>Framed application</title></head>
<body><main>Parent overview</main><iframe src="https://static.browser.test/framed-controls"></iframe></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/framed-controls":
            body = b"""<!doctype html><html><head><title>Frame controls</title></head>
<body><form><input aria-label="Target"><button>Deploy</button></form><script>
Object.defineProperty(document.body, 'innerText', {
  configurable: true,
  get() {
    fetch('https://static.browser.test/private/frame-late');
    for (let index = 0; index < 100; index += 1) {
      const button = document.createElement('button');
      button.textContent = `frame-controlled ${index}`;
      document.body.append(button);
    }
    return 'frame-controlled replacement';
  }
});
</script></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/mixed-frames":
            body = b"""<!doctype html><html><head><title>Mixed frames</title></head><body><main>Parent frame shell</main><iframe hidden src="https://static.browser.test/hidden-controls"></iframe><iframe src="https://static.browser.test/visible-article"></iframe></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/hidden-controls":
            body = b"""<!doctype html><html><head><title>Hidden controls</title></head><body><form><input aria-label="Hidden target"><button>Hidden deploy</button></form></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/visible-article":
            body = b"""<!doctype html><html><head><title>Visible article</title></head><body><article>Visible frame article</article></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/redirect-then-denied":
            self.send_response(302)
            self.send_header("Location", "/page-with-denied-subresource")
            self.end_headers()
            return
        if self.path == "/page-with-denied-subresource":
            body = b"""<!doctype html>
<html><head><title>Denied subresource probe</title></head>
<body><script src="https://static.browser.test/private/subresource.js"></script>
<main>page after allowed redirect</main></body></html>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/popup-target":
            self._send(200, "text/html; charset=utf-8", b"<p>secondary page</p>")
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _RecordingUpstream:
    def __init__(self, inner: HttpxUpstream) -> None:
        self.inner = inner
        self.requests: list[CapturedRequest] = []

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.requests.append(request.model_copy(deep=True))
        return await self.inner.send(request)


async def _drive_browser_fetch() -> dict[str, Any]:
    endpoint = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _BrowserFixtureHandler)
    endpoint_thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    endpoint_thread.start()
    origin = f"http://{_host_private_address()}:{endpoint.server_port}"
    hosts = ("docs.browser.test", "static.browser.test")
    upstream = _RecordingUpstream(
        HttpxUpstream(
            routes={host: origin for host in hosts},
            max_response_bytes=1024 * 1024,
        )
    )
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "browser-fixture": BrowserEgressPolicy(
                name="browser-fixture",
                allowed_hosts=hosts,
                denied_prefixes=("/private",),
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination=host,
                policy_name="browser-fixture",
            )
            for host in hosts
        ],
        credentials=[],
        adapter=DockerEgressAdapter(seccomp_profile=str(_SECCOMP_PROFILE)),
        image=_BROWSER_IMAGE,
        upstream=upstream,
    )
    result = None
    try:
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="browser-fetch-e2e",
                agent_name="agent",
                environment_name="browser",
            )
        )
        runner = result.environment.runner
        binding = result.environment.binding
        assert runner is not None
        assert binding is not None
        handle = InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
        )
        tool = WebFetchTool(
            adapter=BrowserWebFetchAdapter(),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        )
        success = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/start"},
        )
        structured = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/structured"},
        )
        navigation = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/navigation"},
        )
        shadow_controls = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/shadow-controls"},
        )
        deep_accessibility = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/deep-accessibility"},
        )
        oversized_dom = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=5),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/shadow-controls"},
        )
        denied = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/redirect-out"},
        )
        locally_denied_redirect = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/redirect-http"},
        )
        popup = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/popup"},
        )
        stable_extraction = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=5),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/mutating-extraction"},
        )
        framed = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=11),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/framed"},
        )
        framed_oversized = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=10),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/framed"},
        )
        mixed_frames = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=13),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/mixed-frames"},
        )
        mixed_frames_oversized = await WebFetchTool(
            adapter=BrowserWebFetchAdapter(max_dom_nodes=12),
            max_response_bytes=1024 * 1024,
            max_content_bytes=16 * 1024,
            timeout_seconds=30,
            max_redirects=3,
        ).run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/mixed-frames"},
        )
        denied_subresource = await tool.run(
            ToolContext(session_id="browser-fetch-e2e", runner=handle),
            {"url": "https://docs.browser.test/redirect-then-denied"},
        )
        worker_integrity_probe = await handle.exec(
            ExecCommand.process(
                "python",
                "-c",
                (
                    "import os; from pathlib import Path; "
                    "path=Path('/opt/cayu-browser/worker.py'); "
                    "\ntry: os.chmod(path, 0o755)\n"
                    "except OSError: pass\n"
                    "else: raise SystemExit('worker permissions are mutable')\n"
                    "try: path.open('ab').close()\n"
                    "except OSError: print('worker-immutable')\n"
                    "else: raise SystemExit('worker content is mutable')"
                ),
            ),
            timeout_s=10,
        )
        network_probe = await handle.exec(
            ExecCommand.process(
                "python",
                "-c",
                (
                    "import socket; "
                    "sock=socket.socket(); sock.settimeout(2); "
                    "\ntry: sock.connect(('1.1.1.1',443))\n"
                    "except OSError: print('direct-network-denied')\n"
                    "else: sock.close(); raise SystemExit('direct network reachable')"
                ),
            ),
            timeout_s=10,
        )
        return {
            "success": success,
            "structured": structured,
            "navigation": navigation,
            "shadow_controls": shadow_controls,
            "deep_accessibility": deep_accessibility,
            "oversized_dom": oversized_dom,
            "denied": denied,
            "locally_denied_redirect": locally_denied_redirect,
            "popup": popup,
            "stable_extraction": stable_extraction,
            "framed": framed,
            "framed_oversized": framed_oversized,
            "mixed_frames": mixed_frames,
            "mixed_frames_oversized": mixed_frames_oversized,
            "denied_subresource": denied_subresource,
            "worker_integrity_probe": worker_integrity_probe,
            "network_probe": network_probe,
            "requests": tuple((request.host, request.path) for request in upstream.requests),
        }
    finally:
        if result is not None:
            runner = result.environment.runner
            binding = result.environment.binding
            if runner is not None and binding is not None:
                bound = await binding.bind(None, runner, session_id="browser-fetch-e2e")
                await binding.finalize(bound, outcome="completed")
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser_fetch_results(browser_fetch_image: None) -> dict[str, Any]:
    del browser_fetch_image
    return asyncio.run(_drive_browser_fetch())


def test_browser_fetch_renders_javascript_through_managed_virtual_egress(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["success"]
    assert result.is_error is False
    assert result.structured == {
        "requested_url": "https://docs.browser.test/start",
        "final_url": "https://docs.browser.test/guide",
        "title": "Rendered fixture",
        "representation": "text",
        "content": "JavaScript-rendered browser content",
        "redirects": [
            {
                "status_code": 302,
                "from_url": "https://docs.browser.test/start",
                "to_url": "https://docs.browser.test/guide",
            }
        ],
        "truncated": False,
        "truncation_reasons": [],
    }
    assert ("docs.browser.test", "/start") in browser_fetch_results["requests"]
    assert ("docs.browser.test", "/guide") in browser_fetch_results["requests"]
    assert ("static.browser.test", "/render.js") in browser_fetch_results["requests"]


def test_browser_fetch_preserves_structured_page_relationships_as_accessibility_evidence(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["structured"]

    assert result.is_error is False
    assert result.structured["requested_url"] == "https://docs.browser.test/structured"
    assert result.structured["final_url"] == "https://docs.browser.test/structured"
    assert result.structured["title"] == "Deployment console"
    assert result.structured["representation"] == "accessibility"
    assert result.structured["truncated"] is False
    assert result.structured["truncation_reasons"] == []
    accessibility = result.structured["content"]
    assert type(accessibility) is str
    assert "Release status" in accessibility
    assert "Production" in accessibility
    assert "Ready" in accessibility
    assert "Deployment controls" in accessibility
    assert "Deploy" in accessibility
    assert ("docs.browser.test", "/structured") in browser_fetch_results["requests"]


def test_browser_fetch_preserves_native_navigation_roles_and_destinations(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["navigation"]

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    accessibility = result.structured["content"]
    assert type(accessibility) is str
    assert 'link "Open deployment console"' in accessibility
    assert "/url:" in accessibility
    assert "/structured" in accessibility


def test_browser_fetch_discovers_and_bounds_open_shadow_dom_controls(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["shadow_controls"]

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    accessibility = result.structured["content"]
    assert type(accessibility) is str
    assert 'button "Deploy environment 0"' in accessibility
    assert 'button "Deploy environment 7"' in accessibility


def test_browser_fetch_reports_accessibility_tree_depth_truncation(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["deep_accessibility"]

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["content"]
    assert "Truncated: true" in result.content
    assert "Truncation reasons: content" in result.content
    accessibility = result.structured["content"]
    assert type(accessibility) is str
    assert 'group "Level 32"' in accessibility
    assert 'group "Level 33"' not in accessibility


def test_browser_fetch_rejects_pages_above_the_configured_dom_node_ceiling(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["oversized_dom"]

    assert result.is_error is True
    assert result.structured == {"error": "oversized_response"}


def test_browser_fetch_denies_unapproved_redirect_and_direct_network(
    browser_fetch_results: dict[str, Any],
) -> None:
    denied = browser_fetch_results["denied"]
    assert denied.is_error is True
    assert denied.structured == {"error": "redirect_denied"}
    locally_denied = browser_fetch_results["locally_denied_redirect"]
    assert locally_denied.is_error is True
    assert locally_denied.structured == {"error": "redirect_denied"}
    assert ("docs.browser.test", "/plain") not in browser_fetch_results["requests"]
    network_probe = browser_fetch_results["network_probe"]
    assert network_probe.exit_code == 0
    assert network_probe.timed_out is False
    assert network_probe.stdout.strip() == "direct-network-denied"


def test_browser_fetch_rejects_untracked_secondary_pages(
    browser_fetch_results: dict[str, Any],
) -> None:
    popup = browser_fetch_results["popup"]
    assert popup.is_error is True
    assert popup.structured == {"error": "fetch_failed"}


def test_browser_fetch_freezes_and_isolates_page_controlled_extraction(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["stable_extraction"]

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    assert result.structured["truncated"] is False
    assert result.structured["content"] == '- button "initial control"'
    assert "page-controlled" not in result.content
    assert ("static.browser.test", "/private/late") not in browser_fetch_results["requests"]


def test_browser_fetch_aggregates_admitted_frames_under_shared_limits(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["framed"]

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    assert result.structured["truncated"] is False
    content = result.structured["content"]
    assert type(content) is str
    assert "[Main frame]" in content
    assert "URL: https://docs.browser.test/framed" in content
    assert "Parent overview" in content
    assert "[Frame 1]" in content
    assert "URL: https://static.browser.test/framed-controls" in content
    assert "Parent frame: 0" in content
    assert 'textbox "Target"' in content
    assert 'button "Deploy"' in content
    assert "frame-controlled" not in content
    assert "frame-timer" not in content
    assert ("static.browser.test", "/private/frame-late") not in browser_fetch_results["requests"]

    oversized = browser_fetch_results["framed_oversized"]
    assert oversized.is_error is True
    assert oversized.structured == {"error": "oversized_response"}


def test_browser_fetch_excludes_hidden_frames_but_counts_their_nodes(
    browser_fetch_results: dict[str, Any],
) -> None:
    result = browser_fetch_results["mixed_frames"]

    assert result.is_error is False
    assert result.structured["representation"] == "text"
    assert result.structured["truncated"] is False
    content = result.structured["content"]
    assert type(content) is str
    assert "Parent frame shell" in content
    assert "[Frame 1]" in content
    assert "[Frame 2]" not in content
    assert "URL: https://static.browser.test/visible-article" in content
    assert "Visible frame article" in content
    assert "hidden-controls" not in content
    assert "Hidden controls" not in content
    assert "Hidden target" not in content
    assert "Hidden deploy" not in content

    oversized = browser_fetch_results["mixed_frames_oversized"]
    assert oversized.is_error is True
    assert oversized.structured == {"error": "oversized_response"}


def test_browser_fetch_does_not_misclassify_denied_subresource_after_redirect(
    browser_fetch_results: dict[str, Any],
) -> None:
    denied = browser_fetch_results["denied_subresource"]
    assert denied.is_error is True
    assert denied.structured == {"error": "destination_denied"}
    assert ("static.browser.test", "/private/subresource.js") not in browser_fetch_results[
        "requests"
    ]


def test_browser_fetch_worker_is_immutable_to_the_runtime_user(
    browser_fetch_results: dict[str, Any],
) -> None:
    probe = browser_fetch_results["worker_integrity_probe"]
    assert probe.exit_code == 0
    assert probe.stdout.strip() == "worker-immutable"


def test_browser_fetch_worker_crash_reaps_profile_guardian(
    browser_fetch_image: None,
) -> None:
    del browser_fetch_image

    async def exercise() -> None:
        runner = await DockerRunner.create(
            f"cayu-browser-guardian-{os.getpid()}",
            image=_BROWSER_IMAGE,
            close_action="remove",
            seccomp_profile=str(_SECCOMP_PROFILE),
        )
        marker = "/tmp/cayu-browser-guardian-crash.json"
        worker_task: asyncio.Task[Any] | None = None
        try:
            worker_script = """
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cayu_browser_worker_crash_probe",
    "/opt/cayu-browser/worker.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("browser worker module is unavailable")
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)

async def main():
    owner = await worker._start_temporary_profile_owner(timeout_seconds=1.0)
    Path(sys.argv[1]).write_text(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "guardian_pid": owner.pid,
                "profile": str(owner.home),
            }
        ),
        encoding="utf-8",
    )
    await asyncio.Event().wait()

asyncio.run(main())
"""
            worker_task = asyncio.create_task(
                runner.exec(
                    ExecCommand.process("python", "-c", worker_script, marker),
                    timeout_s=20,
                )
            )

            metadata: dict[str, Any] | None = None
            deadline = asyncio.get_running_loop().time() + 5
            while metadata is None and asyncio.get_running_loop().time() < deadline:
                marker_probe = await runner.exec(
                    ExecCommand.process("cat", marker),
                    timeout_s=5,
                )
                if marker_probe.exit_code == 0:
                    metadata = json.loads(marker_probe.stdout)
                    break
                await asyncio.sleep(0.05)
            assert metadata is not None
            worker_pid = metadata["worker_pid"]
            guardian_pid = metadata["guardian_pid"]
            profile = metadata["profile"]
            assert type(worker_pid) is int
            assert type(guardian_pid) is int
            assert type(profile) is str

            killed = await runner.exec(
                ExecCommand.process("kill", "-KILL", str(worker_pid)),
                timeout_s=5,
            )
            assert killed.exit_code == 0
            worker_result = await asyncio.wait_for(worker_task, timeout=5)
            assert worker_result.exit_code != 0

            last_state: dict[str, Any] | None = None
            probe_script = """
import json
import sys
from pathlib import Path

process = Path("/proc") / sys.argv[1]
status = None
if process.exists():
    try:
        status = next(
            line for line in (process / "status").read_text().splitlines()
            if line.startswith("State:")
        )
    except (OSError, StopIteration):
        status = "unknown"
print(
    json.dumps(
        {
            "process_exists": process.exists(),
            "profile_exists": Path(sys.argv[2]).exists(),
            "status": status,
        }
    )
)
"""
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                state_probe = await runner.exec(
                    ExecCommand.process(
                        "python",
                        "-c",
                        probe_script,
                        str(guardian_pid),
                        profile,
                    ),
                    timeout_s=5,
                )
                assert state_probe.exit_code == 0
                last_state = json.loads(state_probe.stdout)
                if not last_state["process_exists"] and not last_state["profile_exists"]:
                    break
                await asyncio.sleep(0.05)
            assert last_state == {
                "process_exists": False,
                "profile_exists": False,
                "status": None,
            }
        finally:
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            await runner.close()

    asyncio.run(exercise())

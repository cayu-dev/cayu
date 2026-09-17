"""Synthetic public routing through the real broker and pinned isolated browser."""

from __future__ import annotations

import asyncio
import http.server
import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from cayu import BrowserSessionTool, PublicWebEgressPolicy, ToolContext
from cayu.egress import HttpxUpstream
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.runtime import VirtualEgressEnvironmentFactory
from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest
from cayu.evals.browser_acceptance_fixture import _fixture_address
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import SecretRedactor

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_PUBLIC_WEB_ACCEPTANCE") != "1",
        reason="Requires installed pinned Docker browser image.",
    ),
]


def test_public_web_discovery_and_private_redirect(tmp_path, monkeypatch):
    observed = []

    class Fixture(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append((self.headers["Host"], self.path))
            if self.path in {"/", "/private"}:
                host = "next.research.test" if self.path == "/" else "private.research.test"
                self.send_response(302)
                self.send_header("Location", f"https://{host}/document")
                self.end_headers()
                return
            body = (
                b"document.getElementById('result').textContent='Discovered resource loaded'"
                if self.path == "/script.js"
                else b"<html><p id='result'>Loading</p><script src='https://static.research.test/script.js'></script></html>"
            )
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/javascript" if self.path == "/script.js" else "text/html",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    async def scenario():
        address = _fixture_address()
        server = http.server.ThreadingHTTPServer((address, 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        original_target = HttpxUpstream._target

        async def resolver(host, port):
            assert port == 443
            return ("127.0.0.1",) if host == "private.research.test" else ("93.184.216.34",)

        async def synthetic_target(self, request):
            # Exercise production DNS/IP admission first. Only its admitted
            # synthetic public endpoint is routed to the local fixture server.
            # No production route/transport escape hatch is enabled.
            target = await original_target(self, request)
            return replace(
                target,
                url=f"http://{address}:{server.server_port}{request.path}",
                sni_hostname=None,
            )

        monkeypatch.setattr(HttpxUpstream, "_target", synthetic_target)
        repo = Path(__file__).resolve().parents[2]
        factory = VirtualEgressEnvironmentFactory(
            policies={"research": PublicWebEgressPolicy(name="research")},
            public_web_policy="research",
            adapter=DockerEgressAdapter(
                seccomp_profile=str(repo / "examples/browser_fetch/seccomp_profile.json"),
                reconnect_state_dir=tmp_path / "ownership",
            ),
            image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            upstream=HttpxUpstream(destination_resolver=resolver),
        )
        allocation = None
        try:
            allocation = await factory.create(
                EnvironmentFactoryRequest(
                    session_id="public-web", agent_name="agent", environment_name="browser"
                )
            )
            context = ToolContext(
                session_id="public-web",
                agent_name="agent",
                environment_name="browser",
                runner=InvocationRunnerHandle(
                    allocation.environment.runner,
                    redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                        0, SecretRedactor()
                    ),
                ),
            )
            tool = BrowserSessionTool(expected_runner_candidate="docker")
            opened = await tool.run(
                context,
                {
                    "operation": "navigate",
                    "operation_id": "open",
                    "url": "https://start.research.test/",
                },
            )
            assert not opened.is_error, opened
            assert "Discovered resource loaded" in opened.content
            assert {host for host, _ in observed} >= {
                "start.research.test",
                "next.research.test",
                "static.research.test",
            }
            denied = await tool.run(
                context,
                {
                    "operation": "navigate",
                    "operation_id": "private",
                    "url": "https://start.research.test/private",
                },
            )
            assert denied.is_error, denied
            assert not any(host == "private.research.test" for host, _ in observed)
            metadata = allocation.reconnect_metadata
            assert metadata
            await allocation.environment.runner.finalize(outcome="interrupted")
            reconnect_request = EnvironmentFactoryRequest(
                session_id="public-web",
                agent_name="agent",
                environment_name="browser",
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata=metadata,
            )
            changed = VirtualEgressEnvironmentFactory(
                policies={"research": PublicWebEgressPolicy(name="research")},
                public_web_policy="research",
                adapter=DockerEgressAdapter(
                    seccomp_profile=str(repo / "examples/browser_fetch/seccomp_profile.json"),
                    reconnect_state_dir=tmp_path / "ownership",
                ),
                image=PINNED_BROWSER_SESSION_WORKLOAD.image,
                upstream=HttpxUpstream(destination_resolver=resolver),
                egress_policy_version="changed",
            )
            with pytest.raises(Exception, match="(?i)(authority|fingerprint|configuration)"):
                await changed.create(reconnect_request)
            allocation = await factory.create(reconnect_request)
            assert allocation.reconnect_metadata == metadata
            resumed = context.model_copy(
                update={
                    "runner": InvocationRunnerHandle(
                        allocation.environment.runner,
                        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                            0, SecretRedactor()
                        ),
                    )
                }
            )
            reopened = await BrowserSessionTool(expected_runner_candidate="docker").run(
                resumed,
                {
                    "operation": "navigate",
                    "operation_id": "reconnected-open",
                    "url": "https://fresh.research.test/document",
                },
            )
            assert not reopened.is_error, reopened
            assert "Discovered resource loaded" in reopened.content

        finally:
            if allocation is not None:
                environment = allocation.environment
                if environment.runner is not None and environment.binding is not None:
                    bound = await environment.binding.bind(
                        None, environment.runner, session_id="public-web"
                    )
                    await environment.binding.finalize(bound, outcome="completed")
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    asyncio.run(scenario())

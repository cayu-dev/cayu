"""Public tool regressions for delayed upload reads and committed POST history."""

from __future__ import annotations

import asyncio
import http.server
import os
import shlex
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    ApprovedEgressDestination,
    ArtifactScope,
    BrowserSessionTool,
    LocalArtifactStore,
    ToolContext,
)
from cayu.egress import HttpEgressPolicy, HttpxUpstream
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryRequest
from cayu.evals.browser_acceptance import BrowserAcceptanceFaultScenario
from cayu.evals.browser_acceptance_fixture import _fixture_address
from cayu.evals.internal.browser_acceptance import (
    _BROWSER_PAGE_FAULT_CONTROL_SCRIPT,
    _FaultControl,
    _install_browser_crash_fault,
)
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD, ExecCommand
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import SecretRedactor

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_BROWSER_OPERATIONS_ACCEPTANCE") != "1",
        reason="Requires the already installed pinned Docker browser image.",
    ),
]


@pytest.mark.parametrize(
    "case",
    [
        "delayed-upload",
        "post-204",
        "reload-race",
        "get-history",
        "upload-disconnection",
        "upload-ack-loss",
    ],
)
def test_public_browser_upload_and_committed_navigation(tmp_path: Path, case: str) -> None:
    posts: list[bytes] = []
    no_content_seen = threading.Event()
    selections: list[str] = []

    class Fixture(http.server.BaseHTTPRequestHandler):
        def reply(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/effect/upload-selected":
                selections.append(self.path)
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/no-content":
                no_content_seen.set()
                self.send_response(204)
                self.end_headers()
                return
            body = b"""<!doctype html><title>Operations</title>
                <label>File <input id="file" type="file"></label>
                <button onclick="file.files[0].text().then(t=>document.getElementById('result').textContent=t)">Read later</button>
                <p id="result">Not read</p>
                <a href="/get-next">Next GET</a>
                <form action="/post" method="post"><button>Submit POST</button></form>"""
            if case in {"upload-disconnection", "upload-ack-loss"}:
                body += b"<script>file.onchange=()=>fetch('/effect/upload-selected')</script>"
            self.reply(body)

        def do_POST(self) -> None:
            posts.append(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            self.reply(b"""<!doctype html><title>POST document</title>
                <p>POST committed</p><a href="/no-content">No content</a>""")

        def log_message(self, _format: str, *_args: object) -> None:
            return

    async def scenario() -> None:
        address = _fixture_address()
        server = http.server.ThreadingHTTPServer((address, 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        repo = Path(__file__).resolve().parents[2]
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="operation-artifacts")
        await store.put_bytes(
            b"Delayed upload contents",
            artifact_id="art_11111111111111111111111111111111",
            filename="later.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="operations-e2e",
        )
        factory = VirtualEgressEnvironmentFactory(
            policies={
                "operations": HttpEgressPolicy(
                    name="operations",
                    allowed_hosts=("operations.browser.test",),
                    allowed_endpoints=(
                        ("GET", "/"),
                        ("GET", "/favicon.ico"),
                        ("GET", "/no-content"),
                        ("GET", "/get-next"),
                        ("GET", "/effect/upload-selected"),
                        ("POST", "/post"),
                    ),
                )
            },
            approved_destinations=(
                ApprovedEgressDestination(
                    destination="operations.browser.test",
                    policy_name="operations",
                ),
            ),
            credentials=[],
            adapter=DockerEgressAdapter(
                seccomp_profile=str(repo / "examples/browser_fetch/seccomp_profile.json")
            ),
            image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            artifact_store=store,
            # Exercise checkout code without downloading/rebuilding a browser.
            host_workspace_path=str(repo / "src/cayu/tools"),
            setup_commands=(
                f"install -o root -g root -m 0555 {shlex.quote(str(repo / 'src/cayu/tools/_browser_guest.py'))} /opt/cayu-browser/worker.py "
                f"&& install -o root -g root -m 0555 {shlex.quote(str(repo / 'src/cayu/tools/_browser_visual_guest.py'))} /opt/cayu-browser/_browser_visual_guest.py",
            ),
            upstream=HttpxUpstream(
                routes={"operations.browser.test": f"http://{address}:{server.server_port}"}
            ),
        )
        allocation = None
        try:
            allocation = await factory.create(
                EnvironmentFactoryRequest(
                    session_id="operations-e2e",
                    agent_name="agent",
                    environment_name="browser",
                )
            )
            environment = allocation.environment
            assert environment.runner is not None
            context = ToolContext(
                session_id="operations-e2e",
                agent_name="agent",
                environment_name="browser",
                runner=InvocationRunnerHandle(
                    environment.runner,
                    redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                        0, SecretRedactor()
                    ),
                ),
                artifact_store=store,
                artifact_store_id=store.id,
            )
            tool = BrowserSessionTool(expected_runner_candidate="docker", max_wait_ms=1000)
            control = None
            backend_evidence: list[tuple[str, str | None, str]] = []
            if case == "reload-race":
                # Test-only guest injection: commit a POST after the production
                # method check, immediately before Chromium receives Page.reload.
                source = (repo / "src/cayu/evals/internal/_browser_page_fault_guest.py").read_text()
                source = source.replace(
                    '    daemon_type = worker["_InteractiveDaemon"]',
                    """    daemon_type = worker["_InteractiveDaemon"]
    original_reload = worker["_interactive_safe_reload"]
    async def raced_reload(state, request):
        original_send = state.cdp.send
        async def send(method, params=None):
            if method == "Page.reload":
                assert params["loaderId"]
                async with state.page.expect_navigation(wait_until="load"):
                    await state.page.evaluate("document.querySelector('form').requestSubmit()")
            return await original_send(method, params)
        state.cdp.send = send
        try:
            return await original_reload(state, request)
        finally:
            state.cdp.send = original_send
    original_reload.__globals__["_interactive_safe_reload"] = raced_reload""",
                )
                original_execute = tool._backend.execute
                launched = False

                async def race_backend(ctx: Any, request: dict[str, Any]) -> Any:
                    nonlocal launched
                    if not launched:
                        result = await environment.runner.exec_system(
                            ExecCommand.process(
                                "/usr/local/bin/python",
                                "-I",
                                "-c",
                                _BROWSER_PAGE_FAULT_CONTROL_SCRIPT,
                                "launch",
                                request["session_id"],
                            ),
                            stdin=source,
                            timeout_s=15,
                            output_limit_bytes=1024,
                        )
                        assert result.exit_code == 0, result
                        launched = True
                    return await original_execute(ctx, request)

                tool._backend.execute = race_backend
            if case in {"upload-disconnection", "upload-ack-loss"}:
                backend = tool._backend
                original_execute = backend.execute

                async def record_backend(ctx: Any, request: dict[str, Any]) -> Any:
                    try:
                        response = await original_execute(ctx, request)
                    except Exception as exc:
                        backend_evidence.append(
                            (request["operation"], type(exc).__name__, str(exc))
                        )
                        raise
                    backend_evidence.append(
                        (
                            request["operation"],
                            None if response.failure is None else response.failure.code,
                            response.allocation_disposition,
                        )
                    )
                    return response

                backend.execute = record_backend
                control = _FaultControl(
                    scenario=(
                        BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_DISCONNECTION
                        if case == "upload-disconnection"
                        else BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_ACKNOWLEDGEMENT_LOSS
                    ),
                    marker_path=tmp_path / "fault-observed",
                    target_operation_number=2,
                )
                _install_browser_crash_fault(types.SimpleNamespace(tools=(tool,)), control)
            opened = await tool.run(
                context,
                {
                    "operation": "navigate",
                    "operation_id": "open",
                    "url": "https://operations.browser.test/",
                },
            )
            assert not opened.is_error, opened
            state = dict(opened.structured or {})
            counter = 0

            async def action(operation: str, name: str | None = None, **extra: Any) -> Any:
                nonlocal state, counter
                counter += 1
                args = {
                    "operation": operation,
                    "operation_id": f"action-{counter}",
                    "session_id": state["session_id"],
                    "page_id": state["page_id"],
                    "expected_revision": state["revision"],
                    "expected_control_epoch": state["control_epoch"],
                    **extra,
                }
                if name is not None:
                    refs = [item for item in state["refs"] if item["name"] == name]
                    assert len(refs) == 1, state
                    args["ref"] = refs[0]["ref"]
                result = await tool.run(context, args)
                if not result.is_error:
                    state = dict(result.structured or {})
                return result

            if control is not None:
                failed = await action(
                    "upload", "File", artifact_ids=["art_11111111111111111111111111111111"]
                )
                expected = (
                    "browser_crash" if case == "upload-disconnection" else "outcome_ambiguous"
                )
                assert failed.structured["error"] == expected, (
                    failed,
                    backend_evidence,
                    selections,
                )
                assert failed.structured["allocation_disposition"] != "live", failed
                assert control.observed, (failed, backend_evidence, selections)
                assert backend_evidence[-1] == ("upload", expected, "uncertain")
                assert len(selections) == 1
            elif case == "delayed-upload":
                selected = await action(
                    "upload", "File", artifact_ids=["art_11111111111111111111111111111111"]
                )
                assert not selected.is_error, selected
                read = await action("click", "Read later")
                assert not read.is_error, read
                observed = await tool.run(
                    context,
                    {
                        "operation": "observe",
                        "operation_id": "read-result",
                        "session_id": state["session_id"],
                        "page_id": state["page_id"],
                    },
                )
                assert "Delayed upload contents" in observed.content, observed
            elif case == "reload-race":
                refused = await action("reload")
                assert refused.structured["error"] == "unsafe_reload", refused
                assert refused.structured["allocation_disposition"] == "live", refused
                assert len(posts) == 1
            elif case == "get-history":
                assert not (await action("click", "Next GET")).is_error
                for operation in ("back", "forward", "reload"):
                    result = await action(operation)
                    assert not result.is_error, result
                assert not posts
            else:
                posted = await action("click", "Submit POST")
                assert not posted.is_error, posted
                assert len(posts) == 1
                await action("click", "No content")
                assert no_content_seen.is_set()
                # Refresh observation even if the noncommitting navigation timed out.
                observed = await tool.run(
                    context,
                    {
                        "operation": "observe",
                        "operation_id": "post-observation",
                        "session_id": state["session_id"],
                        "page_id": state["page_id"],
                    },
                )
                assert not observed.is_error, observed
                state = dict(observed.structured or {})
                refused = await action("reload")
                assert refused.structured["error"] == "unsafe_reload", refused
                assert len(posts) == 1
            closed = await tool.run(
                context,
                {
                    "operation": "close",
                    "operation_id": "close",
                    "session_id": state["session_id"],
                },
            )
            assert not closed.is_error, closed
        finally:
            if allocation is not None:
                environment = allocation.environment
                if environment.runner is not None and environment.binding is not None:
                    bound = await environment.binding.bind(
                        None, environment.runner, session_id="operations-e2e"
                    )
                    await environment.binding.finalize(bound, outcome="completed")
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    asyncio.run(scenario())

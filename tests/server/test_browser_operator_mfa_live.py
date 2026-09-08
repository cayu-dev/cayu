"""Designated disposable login/TOTP acceptance; no external account or API key.

The model and allocation adapter are test doubles. CayuApp, HTTPS/WSS, the
compiled operator dashboard, native Chromium and password/TOTP verification are
real. Production Docker/egress and profile-consent proofs are separate tests.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from pydantic import SecretBytes
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.core.test_browser_session import _WireRunner
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider
from tests.server._browser_control_tls_server import browser_control_tls_server
from tests.server._browser_operator_mfa_fixture import LocalMfa, totp

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    run_to_completion,
)
from cayu.artifacts import LocalArtifactStore
from cayu.runners import ExecResult, Runner
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime.browser_control import (
    BrowserControlPolicy,
    BrowserControlPolicyResult,
    BrowserOperatorPurpose,
)
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.server import (
    BasicAuth,
    BrowserControlServerConfig,
    DashboardConfig,
    ServerConfig,
    create_server,
)
from cayu.tools import _browser_guest
from cayu.tools._browser_control_transport import open_guest_control_channel
from cayu.tools.browser_session import BrowserSessionTool

control_tls = _control_tls
pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1"
    or not os.environ.get("CAYU_BROWSER_DASHBOARD_BUILD"),
    reason="Opt-in native browser acceptance requiring a compiled dashboard.",
)


@pytest.mark.parametrize("persistent", [False, True])
def test_designated_account_login_mfa_and_private_handback(
    tmp_path, monkeypatch, control_tls, caplog, capfd, recwarn, persistent
):
    from playwright.async_api import async_playwright

    fixture = LocalMfa()
    # RFC 6238 SHA-1 vector, six-digit projection.
    assert totp(b"12345678901234567890", 1) == "287082"
    caplog.set_level(logging.INFO)

    async def scenario():
        _, tls = control_tls
        cert = x509.load_pem_x509_certificate((tmp_path / "certificate.pem").read_bytes())
        pin = base64.b64encode(
            hashlib.sha256(
                cert.public_key().public_bytes(
                    serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            ).digest()
        ).decode()
        password = secrets.token_urlsafe(32)
        fixture.private_values.append(password)
        session_id = "local-mfa-" + secrets.token_hex(8)
        store = (
            SQLiteSessionStore(tmp_path / "session.sqlite")
            if persistent
            else InMemorySessionStore()
        )
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        daemons = {}
        api = None

        async def network_app(scope, receive, send):
            target = fixture.api if scope["path"].startswith("/fixture/") else api
            assert target is not None, "Operator server is not initialized"
            await target(scope, receive, send)

        async def connect(*, endpoint, credential):
            return await open_guest_control_channel(
                endpoint=endpoint, credential=credential, tls=tls
            )

        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"),
                    headless=True,
                    args=["--ignore-certificate-errors-spki-list=" + pin],
                )
                try:

                    class NativeRunner(_WireRunner, Runner):
                        preflight_exec = Runner.preflight_exec

                        async def exec(self, command, **kwargs):
                            raw = json.loads(kwargs["stdin"])
                            if raw.get("protocol_version") == "cayu.browser-control-bootstrap.v1":
                                daemon = daemons[raw.pop("session_id")]
                                raw.pop("protocol_version")
                                result = await daemon.bootstrap_operator_channel(raw)
                            else:
                                request = _browser_guest._interactive_request_from_json(raw)
                                daemon = daemons.get(request.session_id)
                                if daemon is None:
                                    daemon = _browser_guest._InteractiveDaemon(request.session_id)
                                    daemon.browser = browser
                                    daemons[request.session_id] = daemon
                                    await daemon._ensure_context(None)
                                    await daemon._ensure_configuration(request)
                                    await daemon.context.route(
                                        "https://mfa.example.test/**", fixture.forward
                                    )
                                result = await daemon.execute(request)
                            return ExecResult(stdout=json.dumps(result))

                    runner = NativeRunner()

                    class Factory(_FakeRemoteFactory):
                        def _result(self, request, resource, *, allocation=None):
                            result = super()._result(request, resource, allocation=allocation)
                            return replace(
                                result,
                                environment=Environment(
                                    result.environment.spec, runner=runner, artifact_store=artifacts
                                ),
                            )

                    class Policy(BrowserControlPolicy):
                        identity = "local-mfa-acceptance:v1"

                        async def decide(self, request):
                            return BrowserControlPolicyResult(
                                allowed=(
                                    request.principal.subject == "operator"
                                    and request.identity.session_id == session_id
                                )
                            )

                    async with browser_control_tls_server(network_app, tmp_path) as port:
                        origin = f"https://127.0.0.1:{port}"
                        fixture.origin, fixture.tls = origin, tls
                        await fixture.check_rejections()
                        app = CayuApp(
                            session_store=store,
                            browser_control=BrowserControlConfig(
                                policy=Policy(),
                                purpose=BrowserOperatorPurpose(
                                    code="login", expected_origins=("https://mfa.example.test",)
                                ),
                                guest_endpoint=origin.replace("https:", "wss:")
                                + "/api/browser-control/guest",
                            ),
                        )

                        class Provider(ScriptedModelProvider):
                            count = 0

                            async def stream(self, request):
                                self.count += 1
                                fixture.assert_safe(repr(request))
                                if self.count == 1:
                                    args = {
                                        "operation": "navigate",
                                        "url": "https://mfa.example.test/fixture/login",
                                        "operation_id": "open",
                                    }
                                elif self.count == 2:
                                    runtime = app._browser_control_runtime
                                    assert runtime is not None
                                    owner = next(iter(runtime.service._owners.values()))
                                    bound = await asyncio.wait_for(asyncio.shield(owner.bound), 10)
                                    context = await browser.new_context(
                                        http_credentials={
                                            "username": "operator",
                                            "password": password,
                                            "origin": origin,
                                        }
                                    )
                                    try:
                                        panel = await context.new_page()
                                        await panel.goto(
                                            origin + "/operator/sessions/" + session_id
                                        )
                                        daemon = daemons[bound.record.identity.browser_session_id]
                                        await fixture.drive(
                                            panel,
                                            runtime,
                                            bound,
                                            native_page=daemon.pages[daemon.active_page_id].page,
                                        )
                                    finally:
                                        await context.close()
                                    daemon = daemons[bound.record.identity.browser_session_id]
                                    args = {
                                        "operation": "observe",
                                        "session_id": daemon.session_id,
                                        "page_id": daemon.active_page_id,
                                        "operation_id": "fresh",
                                    }
                                else:
                                    assert self.count == 3
                                    fixture.assert_authenticated(repr(request))
                                    yield ModelStreamEvent.text_delta(
                                        "Authentication confirmed after fresh observation."
                                    )
                                    yield ModelStreamEvent.completed()
                                    return
                                yield ModelStreamEvent.tool_call(
                                    id=f"mfa-{self.count}", name="browser_session", arguments=args
                                )
                                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

                        app.register_provider(Provider([]), default=True)
                        app.register_environment_factory(
                            EnvironmentSpec(name="browser"),
                            Factory(_FakeRemoteProvider()),
                            artifact_store=artifacts,
                            default=True,
                        )
                        app.register_agent(
                            AgentSpec(name="agent", model="scripted"),
                            tools=[
                                BrowserSessionTool(
                                    max_sessions=1, idle_timeout_seconds=120, max_operations=32
                                )
                            ],
                        )
                        api = create_server(
                            app,
                            config=ServerConfig.protected(
                                BasicAuth(username="operator", password=password),
                                dashboard=DashboardConfig(
                                    path="/operator",
                                    directory=Path(os.environ["CAYU_BROWSER_DASHBOARD_BUILD"]),
                                ),
                                browser_control=BrowserControlServerConfig(
                                    operator_origin=origin,
                                    signing_key=SecretBytes(secrets.token_bytes(32)),
                                ),
                            ),
                        )
                        async with asyncio.timeout(120):
                            outcome = await run_to_completion(
                                app,
                                RunRequest(
                                    agent_name="agent",
                                    session_id=session_id,
                                    messages=[
                                        Message.text(
                                            "user",
                                            "Open the designated login page and wait for private handback.",
                                        )
                                    ],
                                    limits=RunLimits(max_tool_calls=4, max_elapsed_seconds=110),
                                ),
                            )
                        assert outcome.ok and fixture.completed, (
                            "Local MFA acceptance did not complete"
                        )
                        fixture.assert_safe(
                            json.dumps(
                                [
                                    event.model_dump(mode="json")
                                    for event in await store.load_events(session_id)
                                ]
                            )
                        )
                        with browser_control_checkpoint_read_scope(session_id):
                            fixture.assert_safe(json.dumps(await store.load_checkpoint(session_id)))
                        # No pixel artifact is allowed during this sensitive journey.
                        assert not (await artifacts.list()).artifacts
                        for daemon in daemons.values():
                            assert await daemon.close(), "Native MFA cleanup did not settle"
                        runtime = app._browser_control_runtime
                        assert runtime is not None
                        assert await runtime.service.channels.drain()
                        assert await runtime.service.settle_bootstrap_retirements()
                finally:
                    await browser.close()
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                for path in (tmp_path / "session.sqlite", tmp_path / "session.sqlite-wal"):
                    if path.exists():
                        fixture.assert_safe(path.read_bytes().decode("utf-8", errors="ignore"))
            for path in (tmp_path / "artifacts").rglob("*"):
                if path.is_file():
                    fixture.assert_safe(path.read_bytes().decode("utf-8", errors="ignore"))

    try:
        asyncio.run(scenario())
    finally:
        captured = capfd.readouterr()
        fixture.assert_safe(captured.out + captured.err + caplog.text)
        fixture.assert_safe("\n".join(str(item.message) for item in recwarn))

"""Lambda MicroVM browser workload declaration, verification, and tool binding."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.core.test_browser_session import _WireRunner
from tests.egress.test_aws_lambda_microvm_adapter import (
    _broker_and_grant,
    _create_adapter_runner,
    _ExecutingPreflightLambdaRunner,
    _FakeProxyServer,
    _PrivateExposure,
)

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
from cayu.artifacts import LocalArtifactStore
from cayu.browser_recording import browser_recording_capability
from cayu.egress import (
    HttpEgressPolicy,
    TransparentEgressBroker,
    UnsupportedEgressError,
    VirtualCredentialRegistry,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
    VirtualEgressRunnerRequest,
)
from cayu.egress._remote_adapter import PROXY_TRANSPORT_TOKEN_ENV
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.destinations import ApprovedEgressDestination
from cayu.egress.policy import BrowserEgressPolicy
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.runners import ExecResult, LambdaMicroVMBrowserWorkloadError, LambdaMicroVMRunner
from cayu.runners.workloads import (
    BROWSER_FETCH_WORKLOAD_NAME,
    BROWSER_SESSION_WORKLOAD_NAME,
    PINNED_BROWSER_FETCH_IMAGE,
    PINNED_BROWSER_FETCH_WORKLOAD,
    PINNED_BROWSER_SESSION_WORKLOAD,
)
from cayu.tools import WebBridge
from cayu.tools.base import ToolContext
from cayu.tools.browser_session import (
    BrowserSessionTool,
    _browser_recording_admitted,
    _RunnerBrowserSessionBackend,
)
from cayu.vaults import SecretRef, StaticVault

_IMAGE = "arn:aws:lambda:us-east-1:123:microvm-image:cayu-browser"


def _adapter(**options: Any) -> LambdaMicroVMEgressAdapter:
    return LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=_PrivateExposure(),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
        **options,
    )


def _factory(
    adapter: LambdaMicroVMEgressAdapter, tmp_path: Path
) -> VirtualEgressEnvironmentFactory:
    return VirtualEgressEnvironmentFactory(
        policies={
            "receiver": HttpEgressPolicy(
                name="receiver",
                allowed_hosts=["receiver.example"],
                allowed_endpoints=[("POST", "/v1/actions")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="RECEIVER_TOKEN",
                secret=SecretRef(name="receiver"),
                destination="receiver.example",
                policy_name="receiver",
            )
        ],
        resolver=StaticVault({"receiver": "receiver-test-secret"}),
        adapter=adapter,
        image=_IMAGE,
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
    )


def test_adapter_declares_the_browser_workload_only_when_configured() -> None:
    plain = _adapter()
    browser = _adapter(browser_workload=True)

    assert plain.declared_workload_authority(BROWSER_SESSION_WORKLOAD_NAME, image=_IMAGE) is None
    assert browser.declared_workload_authority(BROWSER_SESSION_WORKLOAD_NAME, image=_IMAGE) is (
        PINNED_BROWSER_SESSION_WORKLOAD
    )
    assert browser.declared_workload_authority(BROWSER_FETCH_WORKLOAD_NAME, image=_IMAGE) is (
        PINNED_BROWSER_FETCH_WORKLOAD
    )
    assert browser.declared_workload_authority("cayu.other", image=_IMAGE) is None
    with pytest.raises(TypeError, match="browser_workload"):
        _adapter(browser_workload="yes")


def test_factory_binds_the_lambda_browser_workload_through_its_adapter(tmp_path: Path) -> None:
    assert _factory(_adapter(), tmp_path).workload_authority(BROWSER_FETCH_WORKLOAD_NAME) is None
    factory = _factory(_adapter(browser_workload=True), tmp_path)

    assert factory.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) == PINNED_BROWSER_FETCH_WORKLOAD
    assert factory.workload_authority(BROWSER_SESSION_WORKLOAD_NAME) == (
        PINNED_BROWSER_SESSION_WORKLOAD
    )


def test_sandboxed_browser_accepts_a_lambda_browser_factory(tmp_path: Path) -> None:
    factory = _factory(_adapter(browser_workload=True), tmp_path)

    fetch = WebBridge.sandboxed_browser(
        environment=factory, browser_image=PINNED_BROWSER_FETCH_IMAGE
    )
    session = WebBridge.sandboxed_browser(
        environment=factory, browser_image=PINNED_BROWSER_FETCH_IMAGE, interactive=True
    )

    assert {tool.spec.name for tool in fetch.tools} == {"web_fetch", "screenshot_page"}
    (session_tool,) = session.tools
    assert session_tool.expected_workload_authority == PINNED_BROWSER_SESSION_WORKLOAD
    assert session_tool.expected_runner_candidate == "lambda-microvm"


def test_sandboxed_browser_refuses_a_lambda_factory_without_the_browser_workload(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pinned browser image and worker"):
        WebBridge.sandboxed_browser(
            environment=_factory(_adapter(), tmp_path),
            browser_image=PINNED_BROWSER_FETCH_IMAGE,
        )


class _BrowserPreflightRunner(_ExecutingPreflightLambdaRunner):
    verified = 0
    fail_verification = False

    async def verify_browser_workload(self, *, timeout_s: int = 30) -> None:
        type(self).verified += 1
        if self.fail_verification:
            raise LambdaMicroVMBrowserWorkloadError("worker file worker.py differs")


def test_admission_verifies_the_browser_workload_before_returning_a_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _BrowserPreflightRunner)
    _BrowserPreflightRunner.verified = 0
    _BrowserPreflightRunner.fail_verification = False
    broker, grant = _broker_and_grant()

    runner = asyncio.run(
        _create_adapter_runner(_adapter(browser_workload=True), broker, grant, tmp_path)
    )

    assert _BrowserPreflightRunner.verified == 1
    assert runner.terminated is False


def test_admission_terminates_a_created_microvm_that_lacks_the_browser_workload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _BrowserPreflightRunner)
    _BrowserPreflightRunner.verified = 0
    _BrowserPreflightRunner.fail_verification = True
    broker, grant = _broker_and_grant()
    try:
        with pytest.raises(LambdaMicroVMBrowserWorkloadError):
            asyncio.run(
                _create_adapter_runner(_adapter(browser_workload=True), broker, grant, tmp_path)
            )
    finally:
        _BrowserPreflightRunner.fail_verification = False

    created = _BrowserPreflightRunner.last_instance
    assert created is not None
    assert created.terminated is True
    assert created.closed is True


def test_admission_skips_browser_verification_when_not_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _BrowserPreflightRunner)
    _BrowserPreflightRunner.verified = 0
    broker, grant = _broker_and_grant()

    asyncio.run(_create_adapter_runner(_adapter(), broker, grant, tmp_path))

    assert _BrowserPreflightRunner.verified == 0


class _LambdaWireRunner(_WireRunner):
    """A Lambda-candidate browser runner that proved the pinned workload."""

    def __init__(self) -> None:
        self.private_calls: list[dict[str, Any]] = []

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return ExecutionAdmissionCandidate(
            candidate="lambda-microvm",
            evidence=ExecutionCapabilityEvidence(
                subject="lambda-microvm",
                claims=tuple(
                    ExecutionCapabilityClaim.available(name)
                    for name in (
                        "deny_by_default_network",
                        "brokered_egress",
                        "confirmed_cancellation",
                        "confirmed_cleanup",
                    )
                ),
            ),
        )

    async def _exec_private_browser_control(self, command: Any, **kwargs: Any) -> ExecResult:
        self.private_calls.append(kwargs)
        raise AssertionError("control credentials must not reach a Lambda guest")


def test_operator_control_is_refused_on_lambda_before_any_credential_delivery() -> None:
    runner = _LambdaWireRunner()
    backend = BrowserSessionTool()._backend
    assert isinstance(backend, _RunnerBrowserSessionBackend)

    with pytest.raises(RuntimeError, match="no guest-to-control-plane network path"):
        asyncio.run(
            backend.bootstrap_control(
                ToolContext(session_id="parent", runner=runner),
                browser_session_id="bs_lambda",
                endpoint="wss://control.example/guest",
                credential="a" * 64,
                scope_sha256="b" * 64,
            )
        )

    assert runner.private_calls == []


def test_recording_is_refused_on_a_lambda_runner_without_a_verified_relay() -> None:
    # The backend can record, but only a runner that verified its control relay.
    assert browser_recording_capability("lambda-microvm").supported is True
    assert not _browser_recording_admitted(
        _LambdaWireRunner(),
        "lambda-microvm",
        "wss://cayu-control:18443/api/browser-recordings/guest",
    )


def _credentialless_broker() -> TransparentEgressBroker:
    return TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=StaticVault({}),
        policies={
            "browser": BrowserEgressPolicy(name="browser", allowed_hosts=("docs.browser.test",))
        },
        approved_destinations=[
            ApprovedEgressDestination(destination="docs.browser.test", policy_name="browser")
        ],
        require_test_mode_credentials=False,
    )


def test_credentialless_browser_egress_is_isolated_by_a_per_session_transport_token() -> None:
    _FakeProxyServer.instances = []
    adapter = _adapter(browser_workload=True)

    async def prepare() -> Any:
        return await adapter.prepare(
            session_id="session-1", grants=[], broker=_credentialless_broker()
        )

    first = asyncio.run(prepare())
    second = asyncio.run(prepare())
    try:
        tokens = [server.transport_auth_token for server in _FakeProxyServer.instances]
        assert len(tokens) == 2 and tokens[0] != tokens[1]
        assert all(token is not None and len(token) == 64 for token in tokens)
        assert first.env[PROXY_TRANSPORT_TOKEN_ENV] == tokens[0].decode("ascii")
        assert second.env[PROXY_TRANSPORT_TOKEN_ENV] == tokens[1].decode("ascii")
    finally:
        asyncio.run(first.close())
        asyncio.run(second.close())


def test_credentialless_egress_stays_refused_without_the_verified_relay() -> None:
    _FakeProxyServer.instances = []
    adapter = _adapter(metadata_isolation="unverified")

    with pytest.raises(UnsupportedEgressError, match="session-isolated"):
        asyncio.run(
            adapter.prepare(session_id="session-1", grants=[], broker=_credentialless_broker())
        )

    assert [server.transport_auth_token for server in _FakeProxyServer.instances] == [None]


def test_privilege_probe_fails_when_agents_can_see_the_transport_token(tmp_path: Path) -> None:
    # A status file for an unprivileged agent, so only the token check can fail.
    status = tmp_path / "status"
    status.write_text(
        "Uid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\n"
        "NoNewPrivs:\t1\nCapPrm:\t0\nCapEff:\t0\nCapAmb:\t0\n"
    )
    script = adapter_module._PRIVILEGE_PROBE_SCRIPT.replace("/proc/self/status", str(status))
    clean = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    clean["HOME"] = str(tmp_path)

    def probe(env: dict[str, str]) -> int:
        return subprocess.run([sys.executable, "-c", script], env=env, check=False).returncode

    assert probe(clean) == 0
    assert probe({**clean, PROXY_TRANSPORT_TOKEN_ENV: "ab" * 32}) == 15
    assert "transport token" in adapter_module._PRIVILEGE_PROBE_FAILURES[15]


def test_admission_declares_output_secrets_from_the_factory_after_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _BrowserPreflightRunner)
    monkeypatch.setattr(adapter_module, "_verify_agent_privilege_boundary", _no_guest_work)
    broker, grant = _broker_and_grant()
    adapter = _adapter(browser_workload=True)

    async def create(secret_values_present: bool | None) -> Any:
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        try:
            return await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="lambda-microvm",
                    image="image-arn",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("receiver.internal",),
                    env_overlay_secret_values_present=secret_values_present,
                )
            )
        finally:
            await binding.close()

    assert asyncio.run(create(False))._env_overlay_secret_values_present is False
    assert asyncio.run(create(True))._env_overlay_secret_values_present is True


async def _no_guest_work(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_direct_runner_output_secret_declaration_fails_closed() -> None:
    def runner(overlay: dict[str, str] | None) -> LambdaMicroVMRunner:
        return LambdaMicroVMRunner(
            object(),
            microvm_id="mvm-1",
            endpoint="mvm-1.lambda-microvm.invalid",
            env_overlay=overlay,
        )

    assert runner(None).output_secret_values_present() is False
    assert runner({"HTTPS_PROXY": "http://10.0.1.20:9443"}).output_secret_values_present() is True

"""Lambda MicroVM browser operator control and recording over the verified relay."""

from __future__ import annotations

import asyncio
import base64
import datetime
import importlib
import os
import socket
import ssl
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fcntl")

import examples.aws.lambda_microvm_sidecar.supervisor as supervisor_module
from examples.aws.lambda_microvm_sidecar.supervisor import (
    AGENT_CONTROL_HOSTNAME,
    AGENT_CONTROL_RELAY_PORT,
    CommandConflictError,
    CommandExecutionBoundary,
    CommandRequestError,
)
from tests.egress.test_aws_lambda_microvm_adapter import (
    _broker_and_grant,
    _create_adapter_runner,
    _ExecutingPreflightLambdaRunner,
    _FakeProxyServer,
    _PrivateExposure,
)
from tests.egress.test_aws_lambda_microvm_browser_admission import _LambdaWireRunner
from tests.runners.lambda_microvm_harness import (
    ConformanceLambdaClient,
    SupervisorTransport,
    _terminal_result,
)

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
import cayu.runners.aws_lambda_microvm as runner_module
import cayu.tools._browser_control_transport as transport_module
from cayu import ExecCommand, LambdaMicroVMRunner
from cayu.browser_recording import browser_recording_capability
from cayu.egress.aws_lambda_microvm_adapter import (
    LambdaMicroVMBrowserControlRelay,
    LambdaMicroVMEgressAdapter,
)
from cayu.runners import ExecResult, LambdaMicroVMOwnershipSuperseded
from cayu.runners.aws_lambda_microvm import (
    LAMBDA_MICROVM_CONTROL_HOSTNAME,
    LAMBDA_MICROVM_CONTROL_RELAY_PORT,
    LambdaMicroVMBrowserControlError,
)
from cayu.runners.base import Runner
from cayu.runners.workloads import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.tools._browser_control_transport import (
    CONTROL_CA_PATH,
    BrowserControlTransportUnavailable,
    control_tls_context,
)
from cayu.tools._runner import InvocationRunnerHandle
from cayu.tools.base import ToolContext
from cayu.tools.browser_session import (
    BrowserSessionTool,
    _browser_recording_admitted,
    _guest_control_endpoint_admitted,
    _RunnerBrowserSessionBackend,
)

SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "examples" / "aws" / "lambda_microvm_sidecar"
_RELAY_ENDPOINT = "wss://cayu-control:18443/api/browser-control/guest"


def _certificates(hostname: str = "localhost") -> tuple[bytes, bytes, bytes]:
    """Return a CA certificate and a server certificate/key issued by it."""

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cayu test control CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
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
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
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


# ---------------------------------------------------------------- sidecar relay


class _Relay:
    def __init__(self, host: str, port: int) -> None:
        self.target = (host, port)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _boundary(relays: list[_Relay], *, agent_netns: str | None = "cayu-agent") -> Any:
    def factory(host: str, port: int) -> _Relay:
        relay = _Relay(host, port)
        relays.append(relay)
        return relay

    return CommandExecutionBoundary(
        agent_uid=1000,
        agent_gid=1000,
        agent_netns=agent_netns,
        control_relay_factory=factory,
    )


def test_control_relay_is_fixed_per_owner_and_cleared_by_close() -> None:
    relays: list[_Relay] = []
    boundary = _boundary(relays)

    configured = boundary.configure_control_relay("10.0.4.7", 8443)

    assert configured == {"hostname": "cayu-control", "port": 18443, "target": "10.0.4.7:8443"}
    assert boundary.configure_control_relay("10.0.4.7", 8443) == configured
    assert [relay.target for relay in relays] == [("10.0.4.7", 8443)]
    with pytest.raises(CommandConflictError, match="changed"):
        boundary.configure_control_relay("10.0.4.8", 8443)

    boundary.close()

    assert relays[0].closed is True
    boundary.configure_control_relay("10.0.4.8", 8443)
    assert [relay.target for relay in relays] == [("10.0.4.7", 8443), ("10.0.4.8", 8443)]


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("127.0.0.1", 8443),
        ("169.254.169.254", 80),
        ("192.0.2.1", 18443),
        ("8.8.8.8", 443),
        ("cayu-control", 8443),
        (None, 8443),
        ("10.0.4.7", 0),
        ("10.0.4.7", 65536),
        ("10.0.4.7", "8443"),
        ("10.0.4.7", True),
    ],
)
def test_control_relay_accepts_only_a_private_ipv4_target(host: Any, port: Any) -> None:
    relays: list[_Relay] = []

    with pytest.raises(CommandRequestError):
        _boundary(relays).configure_control_relay(host, port)

    assert relays == []


def test_control_relay_requires_the_agent_network_namespace() -> None:
    relays: list[_Relay] = []

    with pytest.raises(CommandRequestError, match="network namespace"):
        _boundary(relays, agent_netns=None).configure_control_relay("10.0.4.7", 8443)


def test_entrypoint_admits_only_the_control_port_and_maps_the_control_name() -> None:
    entrypoint = (SIDECAR_ROOT / "entrypoint.sh").read_text()

    assert "iptables -w -I INPUT 1 -i cayu-root -j REJECT" in entrypoint
    accepted = [
        line.strip() for line in entrypoint.splitlines() if "iptables" in line and "ACCEPT" in line
    ]
    assert accepted == [
        "iptables -w -I INPUT 1 -i cayu-root -p tcp --dport 18080 -j ACCEPT",
        f"iptables -w -I INPUT 1 -i cayu-root -p tcp --dport {AGENT_CONTROL_RELAY_PORT} -j ACCEPT",
    ]
    assert f"192.0.2.1 {AGENT_CONTROL_HOSTNAME}" in entrypoint
    assert "/etc/netns/$CAYU_MICROVM_AGENT_NETNS/hosts" in entrypoint
    assert (AGENT_CONTROL_HOSTNAME, AGENT_CONTROL_RELAY_PORT) == (
        LAMBDA_MICROVM_CONTROL_HOSTNAME,
        LAMBDA_MICROVM_CONTROL_RELAY_PORT,
    )


def test_sidecar_control_relay_endpoint_is_owner_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app_module = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    relays: list[_Relay] = []

    def factory(host: str, port: int) -> _Relay:
        relays.append(_Relay(host, port))
        return relays[-1]

    monkeypatch.setattr(app_module.EXECUTION_BOUNDARY, "_control_relay_factory", factory)
    claim = "c" * 64
    try:
        asyncio.run(app_module.claim_owner({"claim_id": claim}))
        with pytest.raises(HTTPException) as unowned:
            asyncio.run(
                app_module.configure_control_relay(
                    {"owner_claim": "d" * 64, "host": "10.0.4.7", "port": 8443}
                )
            )
        assert unowned.value.status_code == app_module.OWNER_SUPERSEDED_STATUS
        assert relays == []

        configured = asyncio.run(
            app_module.configure_control_relay(
                {"owner_claim": claim, "host": "10.0.4.7", "port": 8443}
            )
        )
        assert configured["target"] == "10.0.4.7:8443"
        with pytest.raises(HTTPException) as changed:
            asyncio.run(
                app_module.configure_control_relay(
                    {"owner_claim": claim, "host": "10.0.4.8", "port": 8443}
                )
            )
        assert changed.value.status_code == 409
        with pytest.raises(HTTPException) as invalid:
            asyncio.run(
                app_module.configure_control_relay(
                    {"owner_claim": claim, "host": "8.8.8.8", "port": 8443}
                )
            )
        assert invalid.value.status_code == 400

        # A superseding owner clears the relay so it configures its own.
        asyncio.run(app_module.claim_owner({"claim_id": "e" * 64}))
        assert relays[0].closed is True

        # While the owner holds a lifecycle lease the relay is refused like a command.
        asyncio.run(app_module.acquire_lifecycle({"claim_id": "e" * 64, "action": "terminate"}))
        with pytest.raises(HTTPException) as leased:
            asyncio.run(
                app_module.configure_control_relay(
                    {"owner_claim": "e" * 64, "host": "10.0.4.7", "port": 8443}
                )
            )
        assert leased.value.status_code == app_module.OWNER_LIFECYCLE_LEASED_STATUS
        assert len(relays) == 1
    finally:
        app_module.EXECUTION_BOUNDARY.close()
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


class _TlsServer:
    """One-thread TLS listener that completes handshakes and then closes."""

    def __init__(self, certificate: bytes, key: bytes, directory: Path) -> None:
        (directory / "server.pem").write_bytes(certificate)
        (directory / "server.key").write_bytes(key)
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(directory / "server.pem", directory / "server.key")
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(0.2)
        self.port = self.listener.getsockname()[1]
        self.handshakes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                with self.context.wrap_socket(connection, server_side=True):
                    self.handshakes += 1
            except (OSError, ssl.SSLError):
                continue

    def close(self) -> None:
        self._stop.set()
        self.listener.close()
        self._thread.join(timeout=2)


def _probe(tmp_path: Path, ca: bytes, port: int) -> int:
    """Run the real relay probe with a worker shim that trusts ``ca``.

    The shim stands in for the worker's ``control_tls_context``, whose fixed
    root-owned path cannot be written by tests; its trust rule is tested below.
    """

    worker = tmp_path / "worker"
    worker.mkdir(exist_ok=True)
    (tmp_path / "control-ca.pem").write_bytes(ca)
    (worker / "_browser_control_transport.py").write_text(
        "import ssl\n"
        "def control_tls_context():\n"
        "    context = ssl.create_default_context()\n"
        f"    context.load_verify_locations(cafile={str(tmp_path / 'control-ca.pem')!r})\n"
        "    return context\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            runner_module._CONTROL_RELAY_PROBE_SCRIPT,
            str(worker),
            "localhost",
            str(port),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.returncode


def test_relay_carries_a_verified_tls_handshake_end_to_end(tmp_path: Path) -> None:
    ca, certificate, key = _certificates()
    other_ca, _certificate, _key = _certificates()
    server = _TlsServer(certificate, key, tmp_path)
    relay = supervisor_module._TcpRelay(
        "127.0.0.1", server.port, listen_host="127.0.0.1", listen_port=0
    )
    relay_port = int(relay.proxy_url.rsplit(":", 1)[1])
    try:
        assert _probe(tmp_path, ca, relay_port) == 0
        assert server.handshakes == 1
        # The relay forwards bytes only: an untrusted certificate still fails.
        assert _probe(tmp_path, other_ca, relay_port) == 22
    finally:
        relay.close()
        server.close()

    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        closed_port = unused.getsockname()[1]
    assert _probe(tmp_path, ca, closed_port) == 21


# -------------------------------------------------------------- worker TLS trust


def _root_owned(monkeypatch: pytest.MonkeyPatch, *, mode: int = 0o644) -> None:
    real_lstat = os.lstat

    def lstat(path: Any) -> os.stat_result:
        info = real_lstat(path)
        values = list(info)
        values[stat.ST_UID] = 0
        values[stat.ST_MODE] = stat.S_IFMT(info.st_mode) | (
            0o755 if stat.S_ISDIR(info.st_mode) else mode
        )
        return os.stat_result(values)

    monkeypatch.setattr(transport_module.os, "lstat", lstat)


def test_control_trust_adds_nothing_when_no_control_ca_is_installed(tmp_path: Path) -> None:
    context = control_tls_context(str(tmp_path / "absent.pem"))

    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert CONTROL_CA_PATH == "/etc/cayu/control-ca.pem"


def test_control_trust_loads_a_root_owned_read_only_control_ca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ca, _certificate, _key = _certificates()
    path = tmp_path / "control-ca.pem"
    path.write_bytes(ca)
    _root_owned(monkeypatch)

    context = control_tls_context(str(path))

    subjects = [dict(item[0] for item in cert["subject"]) for cert in context.get_ca_certs()]
    assert {"commonName": "cayu test control CA"} in subjects


def test_control_trust_refuses_a_control_ca_an_agent_could_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ca, _certificate, _key = _certificates()
    path = tmp_path / "control-ca.pem"
    path.write_bytes(ca)

    # Owned by the test user, not root.
    with pytest.raises(BrowserControlTransportUnavailable):
        control_tls_context(str(path))

    _root_owned(monkeypatch, mode=0o666)
    with pytest.raises(BrowserControlTransportUnavailable):
        control_tls_context(str(path))


def test_control_trust_refuses_an_unreadable_control_ca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "control-ca.pem"
    path.write_bytes(b"not a certificate")
    _root_owned(monkeypatch)

    with pytest.raises(BrowserControlTransportUnavailable):
        control_tls_context(str(path))


# ------------------------------------------------------------------ runner


class _ControlRelayTransport(SupervisorTransport):
    """Answer the CA install, relay configuration, and relay probe as a guest would."""

    def __init__(self, root: Path, *, probe_exit_code: int = 0) -> None:
        super().__init__(root)
        self.probe_exit_code = probe_exit_code
        self.installs: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []
        self.finalizations: list[dict[str, Any]] = []
        self.relays: list[dict[str, Any]] = []
        self.events: list[str] = []

    async def start_command(
        self, *, command_id: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        argv = payload.get("argv") or []
        if runner_module._CONTROL_CA_INSTALL_SCRIPT in argv:
            self.installs.append(self.admit_start(payload))
            self.events.append("install")
            self._scripted_results[command_id] = _terminal_result(command_id, exit_code=0)
            return {"command_id": command_id, "state": "accepted"}
        if runner_module._CONTROL_RELAY_PROBE_SCRIPT in argv:
            self.probes.append(self.admit_start(payload))
            self.events.append("probe")
            self._scripted_results[command_id] = _terminal_result(
                command_id, exit_code=self.probe_exit_code
            )
            return {"command_id": command_id, "state": "accepted"}
        if "--finalize-recordings" in argv:
            self.finalizations.append(self.admit_start(payload))
            self._scripted_results[command_id] = _terminal_result(command_id, exit_code=0)
            return {"command_id": command_id, "state": "accepted"}
        return await super().start_command(command_id=command_id, payload=payload, **kwargs)

    async def configure_control_relay(
        self, *, owner_claim: str, host: str, port: int, **_kwargs: Any
    ) -> dict[str, Any]:
        if not self.owner_fence.is_current(owner_claim):
            raise LambdaMicroVMOwnershipSuperseded("fake sidecar reports a newer owner")
        self.relays.append({"host": host, "port": port})
        self.events.append("relay")
        return {"hostname": "cayu-control", "port": 18443, "target": f"{host}:{port}"}


def _runner(transport: SupervisorTransport, root: Path) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-control",
        endpoint="conformance.lambda-microvm.invalid",
        endpoint_transport=transport,
        default_cwd=str(root),
        poll_interval_s=0,
    )


def _configure(runner: LambdaMicroVMRunner, ca: bytes, **overrides: Any) -> None:
    options: dict[str, Any] = {"host": "10.0.4.7", "port": 8443, "ca_certificate_pem": ca}
    options.update(overrides)
    asyncio.run(runner.configure_browser_control_relay(**options))


def test_relay_endpoint_is_reachable_only_after_a_verified_handshake(tmp_path: Path) -> None:
    ca, _certificate, _key = _certificates()
    transport = _ControlRelayTransport(tmp_path)
    runner = _runner(transport, tmp_path)

    assert runner.browser_control_endpoint_reachable(_RELAY_ENDPOINT) is False

    _configure(runner, ca)

    assert transport.events == ["install", "relay", "probe"]
    (install,) = transport.installs
    assert install["execution_profile"] == "trusted"
    assert install["argv"][-1] == CONTROL_CA_PATH
    assert base64.b64decode(install["stdin_base64"]) == ca
    assert transport.relays == [{"host": "10.0.4.7", "port": 8443}]
    (probe,) = transport.probes
    assert probe["execution_profile"] == "agent"
    assert probe["argv"][0] == PINNED_BROWSER_SESSION_WORKLOAD.command[0]
    assert probe["argv"][-3:] == ["/opt/cayu-browser", "cayu-control", "18443"]
    assert runner.browser_control_endpoint_reachable(_RELAY_ENDPOINT) is True
    for endpoint in (
        "wss://cayu-control:8443/api/browser-control/guest",
        "wss://control.example:18443/api/browser-control/guest",
        "https://cayu-control:18443/api/browser-control/guest",
        "wss://cayu-control/api/browser-control/guest",
        "wss://cayu-control:bad/",
        None,
    ):
        assert runner.browser_control_endpoint_reachable(endpoint) is False  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("exit_code", "reason"),
    [
        (21, "did not connect"),
        (22, "not trusted for cayu-control"),
        (23, "handshake"),
        (1, "did not complete"),
    ],
)
def test_failed_relay_probe_leaves_control_unreachable(
    tmp_path: Path, exit_code: int, reason: str
) -> None:
    ca, _certificate, _key = _certificates()
    runner = _runner(_ControlRelayTransport(tmp_path, probe_exit_code=exit_code), tmp_path)

    with pytest.raises(LambdaMicroVMBrowserControlError, match=reason):
        _configure(runner, ca)

    assert runner.browser_control_endpoint_reachable(_RELAY_ENDPOINT) is False
    assert runner.browser_recording_supported() is False


@pytest.mark.parametrize(
    "ca", [b"", b"not a certificate", "-----BEGIN CERTIFICATE-----", b"x" * (64 * 1024 + 1)]
)
def test_invalid_control_ca_is_refused_before_any_guest_command(tmp_path: Path, ca: Any) -> None:
    transport = _ControlRelayTransport(tmp_path)

    with pytest.raises(ValueError, match="control CA"):
        _configure(_runner(transport, tmp_path), ca)

    assert transport.events == []


def test_control_ca_install_carries_only_the_public_certificates(tmp_path: Path) -> None:
    ca, certificate, key = _certificates()
    transport = _ControlRelayTransport(tmp_path)

    _configure(_runner(transport, tmp_path), ca + key + certificate)

    installed = base64.b64decode(transport.installs[0]["stdin_base64"]).decode("ascii")
    assert "PRIVATE KEY" not in installed
    assert installed.count("BEGIN CERTIFICATE") == 2


def test_recording_is_declared_only_with_a_verified_worker_and_relay(tmp_path: Path) -> None:
    ca, _certificate, _key = _certificates()
    runner = _runner(_ControlRelayTransport(tmp_path), tmp_path)

    _configure(runner, ca)
    assert runner.browser_recording_supported() is False

    runner._browser_workload_verified = True
    assert runner.browser_recording_supported() is True


def test_lifecycle_transitions_clear_the_verified_relay(tmp_path: Path) -> None:
    ca, _certificate, _key = _certificates()
    runner = _runner(_ControlRelayTransport(tmp_path), tmp_path)
    runner._browser_workload_verified = True
    _configure(runner, ca)

    asyncio.run(runner.suspend())

    assert runner.browser_control_endpoint_reachable(_RELAY_ENDPOINT) is False
    assert runner.browser_recording_supported() is False


def test_recording_finalization_runs_the_worker_in_the_agent_profile(tmp_path: Path) -> None:
    ca, _certificate, _key = _certificates()
    transport = _ControlRelayTransport(tmp_path)
    runner = _runner(transport, tmp_path)

    asyncio.run(runner._finalize_browser_recordings(normal=True))
    assert transport.finalizations == []

    runner._browser_workload_verified = True
    _configure(runner, ca)
    asyncio.run(runner._finalize_browser_recordings(normal=True))
    asyncio.run(runner._finalize_browser_recordings(normal=False))

    assert [item["argv"] for item in transport.finalizations] == [
        [*PINNED_BROWSER_SESSION_WORKLOAD.command, "--finalize-recordings", "normal"],
        [*PINNED_BROWSER_SESSION_WORKLOAD.command, "--finalize-recordings", "partial"],
    ]
    assert {item["execution_profile"] for item in transport.finalizations} == {"agent"}
    assert all(item["timeout_s"] <= 6 for item in transport.finalizations)


def test_recording_finalization_failure_never_fails_disposal(tmp_path: Path) -> None:
    runner = _runner(_ControlRelayTransport(tmp_path), tmp_path)
    runner._browser_workload_verified = True
    runner._browser_control_relay_verified = True

    async def failing_exec(*_args: Any, **_kwargs: Any) -> ExecResult:
        raise RuntimeError("guest unavailable")

    runner.exec = failing_exec  # ty: ignore[invalid-assignment]

    asyncio.run(runner._finalize_browser_recordings(normal=True))


# ------------------------------------------------------------------ adapter


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


def _relay_config(ca: bytes | None = None) -> LambdaMicroVMBrowserControlRelay:
    return LambdaMicroVMBrowserControlRelay(
        host="10.0.4.7", port=8443, ca_certificate_pem=ca or _certificates()[0]
    )


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"host": "8.8.8.8"}, "private IPv4"),
        ({"host": "192.0.2.1"}, "private IPv4"),
        ({"host": "cayu-control"}, "private IPv4"),
        ({"port": 0}, "port"),
        ({"port": "8443"}, "port"),
        ({"ca_certificate_pem": b""}, "PEM CA"),
    ],
)
def test_relay_configuration_validates_its_target(options: dict[str, Any], error: str) -> None:
    values: dict[str, Any] = {"host": "10.0.4.7", "port": 8443, "ca_certificate_pem": b"pem"}
    values.update(options)

    with pytest.raises(ValueError, match=error):
        LambdaMicroVMBrowserControlRelay(**values)


def test_adapter_requires_a_verified_browser_for_the_control_relay() -> None:
    relay = _relay_config()

    with pytest.raises(ValueError, match="browser_workload=True"):
        _adapter(browser_control_relay=relay)
    with pytest.raises(ValueError, match="metadata_isolation"):
        _adapter(
            browser_workload=True,
            browser_control_relay=relay,
            metadata_isolation="unverified",
        )
    with pytest.raises(TypeError, match="LambdaMicroVMBrowserControlRelay"):
        _adapter(browser_workload=True, browser_control_relay={"host": "10.0.4.7"})
    assert _adapter(browser_workload=True, browser_control_relay=relay).browser_control_relay is (
        relay
    )


class _ControlPreflightRunner(_ExecutingPreflightLambdaRunner):
    events: list[Any] = []
    fail_relay = False

    async def verify_browser_workload(self, *, timeout_s: int = 30) -> None:
        type(self).events.append("verify")

    async def configure_browser_control_relay(self, **options: Any) -> None:
        type(self).events.append(("relay", options))
        if self.fail_relay:
            raise LambdaMicroVMBrowserControlError("the relay did not connect")

    async def _finalize_browser_recordings(self, *, normal: bool) -> None:
        type(self).events.append(("finalize", normal, self.close_action, self.closed))


def test_admission_verifies_the_relay_after_the_browser_workload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _ControlPreflightRunner)
    _ControlPreflightRunner.events = []
    relay = _relay_config()
    broker, grant = _broker_and_grant()

    asyncio.run(
        _create_adapter_runner(
            _adapter(browser_workload=True, browser_control_relay=relay),
            broker,
            grant,
            tmp_path,
        )
    )

    assert _ControlPreflightRunner.events == [
        "verify",
        (
            "relay",
            {
                "host": "10.0.4.7",
                "port": 8443,
                "ca_certificate_pem": relay.ca_certificate_pem,
                "timeout_s": 30,
            },
        ),
    ]


def test_admission_terminates_a_created_microvm_whose_relay_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _ControlPreflightRunner)
    _ControlPreflightRunner.events = []
    _ControlPreflightRunner.fail_relay = True
    broker, grant = _broker_and_grant()
    try:
        with pytest.raises(LambdaMicroVMBrowserControlError):
            asyncio.run(
                _create_adapter_runner(
                    _adapter(browser_workload=True, browser_control_relay=_relay_config()),
                    broker,
                    grant,
                    tmp_path,
                )
            )
    finally:
        _ControlPreflightRunner.fail_relay = False

    created = _ControlPreflightRunner.last_instance
    assert created is not None and created.terminated is True


@pytest.mark.parametrize(
    ("outcome", "normal"),
    [("completed", True), ("failed", False), ("interrupted", False), (None, False)],
)
def test_finalization_settles_recordings_before_the_microvm_stops(
    monkeypatch: pytest.MonkeyPatch, outcome: str | None, normal: bool
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _ControlPreflightRunner)
    _ControlPreflightRunner.events = []
    runner = _ControlPreflightRunner()

    asyncio.run(_adapter(browser_workload=True).finalize_runner(runner, outcome=outcome))

    assert _ControlPreflightRunner.events == [("finalize", normal, "none", False)]
    assert runner.closed is True


# ------------------------------------------------------------- tool admission


class _DeclaringLambdaWireRunner(_LambdaWireRunner):
    def __init__(self, *, reachable: bool | None, recording: bool = False) -> None:
        super().__init__()
        self.reachable = reachable
        self.recording = recording
        self.asked: list[str] = []

    def browser_control_endpoint_reachable(self, endpoint: str) -> bool | None:
        self.asked.append(endpoint)
        return self.reachable

    def browser_recording_supported(self) -> bool:
        return self.recording

    async def _exec_private_browser_control(self, command: Any, **kwargs: Any) -> ExecResult:
        self.private_calls.append(kwargs)
        return ExecResult(
            stdout='{"schema_version":1,"bootstrap_accepted":true}', stderr="", exit_code=0
        )


def _bootstrap(runner: Any, endpoint: str = _RELAY_ENDPOINT) -> None:
    backend = BrowserSessionTool()._backend
    assert isinstance(backend, _RunnerBrowserSessionBackend)
    asyncio.run(
        backend.bootstrap_control(
            ToolContext(session_id="parent", runner=runner),
            browser_session_id="bs_lambda",
            endpoint=endpoint,
            credential="a" * 64,
            scope_sha256="b" * 64,
        )
    )


def test_control_credential_reaches_a_lambda_guest_only_over_its_verified_relay() -> None:
    runner = _DeclaringLambdaWireRunner(reachable=True)

    _bootstrap(runner)

    assert runner.asked == [_RELAY_ENDPOINT]
    (call,) = runner.private_calls
    assert '"credential":"' + "a" * 64 + '"' in call["stdin"]


@pytest.mark.parametrize("reachable", [False, None])
def test_control_is_refused_when_a_lambda_runner_does_not_declare_the_endpoint(
    reachable: bool | None,
) -> None:
    runner = _DeclaringLambdaWireRunner(reachable=reachable)

    with pytest.raises(RuntimeError, match="no guest-to-control-plane network path"):
        _bootstrap(runner)

    assert runner.private_calls == []


def test_control_gate_keeps_undeclared_backends_unchanged() -> None:
    class _Undeclared:
        pass

    assert _guest_control_endpoint_admitted(_Undeclared(), "docker", _RELAY_ENDPOINT) is True
    assert _guest_control_endpoint_admitted(_Undeclared(), "lambda-microvm", _RELAY_ENDPOINT) is (
        False
    )
    denying = _DeclaringLambdaWireRunner(reachable=False)
    assert _guest_control_endpoint_admitted(denying, "docker", _RELAY_ENDPOINT) is False


def test_recording_admission_follows_the_runner_declaration() -> None:
    endpoint = "wss://cayu-control:18443/api/browser-recordings/guest"

    class _Undeclared:
        pass

    assert _browser_recording_admitted(_Undeclared(), "docker", endpoint) is False
    assert (
        _browser_recording_admitted(
            _DeclaringLambdaWireRunner(reachable=True, recording=False), "lambda-microvm", endpoint
        )
        is False
    )
    assert (
        _browser_recording_admitted(
            _DeclaringLambdaWireRunner(reachable=False, recording=True), "lambda-microvm", endpoint
        )
        is False
    )
    assert (
        _browser_recording_admitted(
            _DeclaringLambdaWireRunner(reachable=True, recording=True), "lambda-microvm", endpoint
        )
        is True
    )
    # Docker declares recording and makes no reachability declaration.
    assert (
        _browser_recording_admitted(
            _DeclaringLambdaWireRunner(reachable=None, recording=True), "docker", endpoint
        )
        is True
    )


def test_recording_capability_names_the_lambda_relay() -> None:
    capability = browser_recording_capability("lambda-microvm")

    assert capability.supported is True
    assert capability.reason == "lambda_microvm_control_relay"
    assert browser_recording_capability("docker").reason == "docker_sampled_active_page"
    assert browser_recording_capability("microsandbox").supported is False


def _handle(runner: Runner) -> InvocationRunnerHandle:
    return InvocationRunnerHandle(runner, redactor_snapshot_provider=lambda: None)


def test_runner_declarations_cross_the_invocation_handle() -> None:
    class _Declaring(Runner):
        def __init__(self, reachable: Any, recording: Any) -> None:
            self.reachable = reachable
            self.recording = recording

        async def exec(self, command: ExecCommand, **_kwargs: Any) -> ExecResult:
            raise AssertionError("not executed")

        def browser_control_endpoint_reachable(self, endpoint: str) -> Any:
            return self.reachable

        def browser_recording_supported(self) -> Any:
            return self.recording

    handle = _handle(_Declaring(True, True))
    assert handle.browser_control_endpoint_reachable(_RELAY_ENDPOINT) is True
    assert handle.browser_recording_supported() is True
    assert (
        _handle(_Declaring(None, False)).browser_control_endpoint_reachable(_RELAY_ENDPOINT) is None
    )
    with pytest.raises(TypeError):
        _handle(_Declaring("yes", True)).browser_control_endpoint_reachable(_RELAY_ENDPOINT)
    with pytest.raises(TypeError):
        _handle(_Declaring(True, None)).browser_recording_supported()


def test_base_runner_makes_no_control_declaration() -> None:
    class _Plain(Runner):
        async def exec(self, command: ExecCommand, **_kwargs: Any) -> ExecResult:
            raise AssertionError("not executed")

    assert _Plain().browser_control_endpoint_reachable(_RELAY_ENDPOINT) is None
    assert _Plain().browser_recording_supported() is False

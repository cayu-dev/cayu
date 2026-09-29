"""Same-MicroVM virtual-egress reconnect with fresh authority."""

from __future__ import annotations

import asyncio
import builtins
import io
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.core.test_environment_allocation_recovery import (
    _ENVIRONMENT_NAME,
    _create_session,
    _resolve,
)
from tests.egress.test_aws_lambda_microvm_recoverable_allocation import (
    _adapter,
    _Clock,
    _factory,
    _FakeProxyServer,
    _skip_guest_preflight,
    clock,
)
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, FakeLambdaClientError

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
from cayu import (
    EnvironmentFactoryOperation,
    ExecCommand,
    ExecResult,
    InMemorySessionStore,
)
from cayu.egress import (
    InvalidEgressReconnectMetadataError,
    TransparentEgressBroker,
    UnsupportedEgressCapabilityError,
    VirtualCredentialRegistry,
)
from cayu.egress.runtime import _build_reconnect_metadata
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners import LambdaMicroVMRunner
from cayu.runtime._environment_lifecycle import ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY
from cayu.vaults import StaticVault

__all__ = ["_skip_guest_preflight", "clock"]
# Captured before the autouse fixture replaces guest probes for runtime tests.
_REAL_PRIVILEGE_PROBE = adapter_module._verify_agent_privilege_boundary

_IDENTITY = {
    "microvm_id": "microvm-0001",
    "endpoint": "microvm-0001.lambda-microvm.invalid",
    "region": "us-east-1",
    "image_identifier": ClientTokenLambdaModel.image_arn,
    "image_version": "3",
    "session_id": "session-1",
    "environment_name": "sandbox",
}


class _RecordingProxyServer(_FakeProxyServer):
    instances: list[_RecordingProxyServer] = []

    def __init__(
        self,
        broker: Any,
        *,
        loop: Any,
        host: str,
        transport_auth_token: bytes | None = None,
    ) -> None:
        self.transport_auth_token = transport_auth_token
        super().__init__(broker, loop=loop, host=host)
        self.broker = broker
        self.closed = False
        type(self).instances.append(self)

    async def close(self) -> None:
        self.closed = True


def _broker() -> TransparentEgressBroker:
    return TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=StaticVault({}),
        policies={},
        require_test_mode_credentials=False,
    )


def test_identity_allowlist_rejects_authority_and_schema_drift(tmp_path: Path) -> None:
    adapter = _adapter(ClientTokenLambdaModel(), tmp_path)

    assert adapter.validate_reconnect_metadata(_IDENTITY) == _IDENTITY
    for drifted in (
        {**_IDENTITY, "endpoint_token": "jwe"},
        {**_IDENTITY, "proxy_authorization": "Bearer x"},
        {key: value for key, value in _IDENTITY.items() if key != "environment_name"},
        {**_IDENTITY, "image_identifier": "image-name"},
        {**_IDENTITY, "microvm_id": " microvm-0001"},
        {**_IDENTITY, "image_version": None},
    ):
        with pytest.raises(InvalidEgressReconnectMetadataError):
            adapter.validate_reconnect_metadata(drifted)


def test_durable_envelope_carries_no_replayable_authority(tmp_path: Path) -> None:
    adapter = _adapter(ClientTokenLambdaModel(), tmp_path)
    envelope = _build_reconnect_metadata(
        EnvironmentFactoryRequest(
            session_id="session-1", agent_name="agent", environment_name="sandbox"
        ),
        runner_kind="lambda-microvm",
        identity=adapter.validate_reconnect_metadata(_IDENTITY),
        supported=True,
        allocation_fingerprint=adapter_module._lambda_environment_fingerprint("microvm-0001"),
    )

    assert envelope["capability"] == "supported"
    assert envelope["identity"] == _IDENTITY
    serialized = repr(envelope).lower()
    for forbidden in ("token", "bearer", "private", "secret", "credential", "ca_cert"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("session_id", "session-2", "different session"),
        ("environment_name", "other", "different environment"),
        ("region", "us-west-2", "different region"),
    ],
)
def test_prepare_reconnect_refuses_foreign_identity_before_binding(
    field: str, value: str, match: str, tmp_path: Path
) -> None:
    _RecordingProxyServer.instances = []
    adapter = _adapter(ClientTokenLambdaModel(), tmp_path)
    adapter.proxy_server_factory = _RecordingProxyServer

    with pytest.raises(InvalidEgressReconnectMetadataError, match=match):
        asyncio.run(
            adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="sandbox",
                grants=(),
                broker=_broker(),
                reconnect_metadata={**_IDENTITY, field: value},
            )
        )
    assert _RecordingProxyServer.instances == []


def test_prepare_reconnect_builds_fresh_authority_each_time(tmp_path: Path) -> None:
    _RecordingProxyServer.instances = []
    adapter = _adapter(ClientTokenLambdaModel(), tmp_path)
    adapter.proxy_server_factory = _RecordingProxyServer
    brokers = [_broker(), _broker()]

    async def run() -> None:
        for broker in brokers:
            binding = await adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="sandbox",
                grants=(),
                broker=broker,
                reconnect_metadata=_IDENTITY,
            )
            await binding.close()

    asyncio.run(run())

    assert [server.broker for server in _RecordingProxyServer.instances] == brokers
    assert all(server.closed for server in _RecordingProxyServer.instances)


@pytest.mark.parametrize(
    ("state", "disposed"),
    [
        ("TERMINATED", True),
        (None, True),
        ("RUNNING", False),
        ("SUSPENDED", False),
        ("TERMINATING", False),
    ],
)
def test_disposal_proof_requires_terminal_control_plane_readback(
    state: str | None, disposed: bool, tmp_path: Path
) -> None:
    model = ClientTokenLambdaModel()
    if state is not None:
        microvm_id = model.run_microvm(imageIdentifier=model.image_arn)["microvmId"]
        model.microvms[microvm_id]["state"] = state
    else:
        microvm_id = "microvm-retired"
    adapter = _adapter(model, tmp_path)

    observed = asyncio.run(adapter.is_allocation_disposed({**_IDENTITY, "microvm_id": microvm_id}))

    assert observed is disposed
    assert model.terminate_calls == []


def test_disposal_proof_does_not_swallow_control_plane_failures(tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()

    def denied(**_kwargs: Any) -> dict[str, Any]:
        raise FakeLambdaClientError("AccessDeniedException", "denied")

    model.get_microvm = denied  # type: ignore[method-assign]

    with pytest.raises(FakeLambdaClientError):
        asyncio.run(_adapter(model, tmp_path).is_allocation_disposed(_IDENTITY))


@pytest.mark.parametrize(
    ("outcome", "state", "preserved"),
    [("interrupted", "SUSPENDED", True), ("completed", "TERMINATED", False)],
)
def test_finalization_preserves_only_interrupted_allocations(
    outcome: str, state: str, preserved: bool, tmp_path: Path
) -> None:
    from tests.runners.lambda_microvm_harness import SupervisorTransport

    async def run() -> None:
        model = ClientTokenLambdaModel()
        runner = await LambdaMicroVMRunner.create(
            model.image_arn,
            client=model,
            endpoint_transport=SupervisorTransport(tmp_path),
            poll_interval_s=0,
            close_action="none",
        )
        result = await _adapter(model, tmp_path).finalize_runner(runner, outcome=outcome)

        assert result.workspace_mutations_quiescent
        assert result.allocation_preserved is preserved
        assert model.microvms[runner.microvm_id]["state"] == state

    asyncio.run(run())


def _probe_exit_code(status: str, *, env: dict[str, str], readable: set[str]) -> int:
    real_open = builtins.open

    def fake_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if path == "/proc/self/status":
            return io.StringIO(status)
        if str(path) in readable:
            return io.StringIO("")
        if str(path).endswith("credentials"):
            raise PermissionError(path)
        return real_open(path, *args, **kwargs)

    namespace: dict[str, Any] = {}
    original_environ = dict(__import__("os").environ)
    try:
        __import__("os").environ.clear()
        __import__("os").environ.update(env)
        builtins.open = fake_open  # type: ignore[assignment]
        try:
            exec(adapter_module._PRIVILEGE_PROBE_SCRIPT, namespace)
        except SystemExit as exit_:
            return int(exit_.code or 0)
        return 0
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        __import__("os").environ.clear()
        __import__("os").environ.update(original_environ)


_UNPRIVILEGED = (
    "Uid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\n"
    "NoNewPrivs:\t1\nCapEff:\t0000000000000000\nCapPrm:\t0000000000000000\n"
    "CapAmb:\t0000000000000000\n"
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX probe semantics")
@pytest.mark.parametrize(
    ("status", "env", "readable", "expected"),
    [
        (_UNPRIVILEGED, {"HOME": "/home/agent"}, set(), 0),
        (_UNPRIVILEGED.replace("Uid:\t1000", "Uid:\t0"), {"HOME": "/home/agent"}, set(), 10),
        (_UNPRIVILEGED.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"), {}, set(), 11),
        (
            _UNPRIVILEGED.replace("CapEff:\t0000000000000000", "CapEff:\t00000000a80425fb"),
            {},
            set(),
            12,
        ),
        (_UNPRIVILEGED, {"AWS_SECRET_ACCESS_KEY": "x"}, set(), 13),
        (_UNPRIVILEGED, {"AWS_CONTAINER_CREDENTIALS_FULL_URI": "x"}, set(), 13),
        (_UNPRIVILEGED, {"AWS_REGION": "us-east-1", "HOME": "/home/agent"}, set(), 0),
        (_UNPRIVILEGED, {"HOME": "/home/agent"}, {"/home/agent/.aws/credentials"}, 14),
    ],
)
def test_privilege_probe_exit_protocol(
    status: str, env: dict[str, str], readable: set[str], expected: int
) -> None:
    assert _probe_exit_code(status, env=env, readable=readable) == expected


class _ProbeRunner:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.commands: list[ExecCommand] = []

    async def exec(self, command: ExecCommand, **_kwargs: Any) -> ExecResult:
        self.commands.append(command)
        return ExecResult(exit_code=self.exit_code)


@pytest.mark.parametrize("exit_code", [10, 11, 12, 13, 14, 99])
def test_privilege_probe_failure_is_a_typed_capability_refusal(exit_code: int) -> None:
    runner = _ProbeRunner(exit_code)

    with pytest.raises(UnsupportedEgressCapabilityError) as raised:
        asyncio.run(
            _REAL_PRIVILEGE_PROBE(runner, timeout_s=5)  # type: ignore[arg-type]
        )

    assert raised.value.capability == "guest_privilege_containment"
    assert runner.commands[0].argv is not None
    assert list(runner.commands[0].argv[:2]) == ["python3", "-c"]


def _suspend_all(model: ClientTokenLambdaModel) -> None:
    for microvm in model.microvms.values():
        if microvm["state"] == "RUNNING":
            microvm["state"] = "SUSPENDED"


def test_runtime_reconnect_resumes_the_same_microvm_with_fresh_authority(
    clock: _Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecordingProxyServer.instances = []

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = ClientTokenLambdaModel()

        def adapter():
            created = _adapter(model, tmp_path)
            created.proxy_server_factory = _RecordingProxyServer
            return created

        created = await _resolve(
            store, session, _factory(adapter()), operation=EnvironmentFactoryOperation.CREATE
        )
        assert created.error is None
        checkpoint = await store.load_checkpoint(session.id) or {}
        envelope = checkpoint[ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY][_ENVIRONMENT_NAME]
        assert envelope["capability"] == "supported"
        [microvm_id] = model.created_ids()
        assert envelope["identity"]["microvm_id"] == microvm_id
        _suspend_all(model)  # interrupted finalization, then the worker is replaced

        reconnected = await _resolve(
            store, session, _factory(adapter()), operation=EnvironmentFactoryOperation.RECONNECT
        )

        assert reconnected.error is None
        assert model.created_ids() == [microvm_id]
        assert model.microvms[microvm_id]["state"] == "RUNNING"
        assert (
            reconnected.registered_environment.live_allocation_fingerprint
            == created.registered_environment.live_allocation_fingerprint
        )
        first, second = _RecordingProxyServer.instances
        assert first.broker is not second.broker

    asyncio.run(run())


@pytest.mark.parametrize("state", ["TERMINATED", "TERMINATING", "FAILED"])
def test_runtime_reconnect_never_replaces_an_ended_microvm(
    state: str, clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = ClientTokenLambdaModel()
        created = await _resolve(
            store,
            session,
            _factory(_adapter(model, tmp_path)),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert created.error is None
        [microvm_id] = model.created_ids()
        model.microvms[microvm_id]["state"] = state
        submissions = len(model.run_calls)

        reconnected = await _resolve(
            store,
            session,
            _factory(_adapter(model, tmp_path)),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )

        assert reconnected.error is not None
        assert model.created_ids() == [microvm_id]
        assert len(model.run_calls) == submissions

    asyncio.run(run())

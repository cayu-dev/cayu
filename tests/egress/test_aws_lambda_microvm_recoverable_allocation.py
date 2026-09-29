"""Crash-window recovery for Lambda MicroVM virtual-egress allocation.

These tests drive the real durable allocation coordinator, virtual-egress
factory, and Lambda MicroVM adapter against a control-plane model of the
client-token semantics verified live. Guest preflight is replaced because it
requires a real MicroVM; the live lane covers it.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.core.test_environment_allocation_recovery import (
    _ENVIRONMENT_NAME,
    _create_session,
    _pending_record,
    _resolve,
    _SimulatedProcessDeath,
)
from tests.runners.lambda_microvm_harness import (
    ClientTokenLambdaModel,
    FakeLambdaClientError,
    SupervisorTransport,
)

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
import cayu.runners.aws_lambda_microvm as runner_module
from cayu import (
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    InMemorySessionStore,
)
from cayu.egress import (
    HttpEgressPolicy,
    VirtualCredentialSpec,
    VirtualEgressAllocationPreparation,
    VirtualEgressAllocationReap,
    VirtualEgressEnvironmentFactory,
    VirtualEgressRunnerRequest,
)
from cayu.egress.aws_lambda_microvm_adapter import (
    DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS,
    LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS,
    LambdaMicroVMAllocationRecoveryError,
    LambdaMicroVMEgressAdapter,
)
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.runtime._environment_allocation import EnvironmentAllocationCoordinator
from cayu.vaults import SecretRedactor, SecretRef, StaticVault

_NOW = 1_800_000_000


class _FakeAuthority:
    def ca_cert_pem(self) -> bytes:
        return b"session-ca"


class _FakeProxyServer:
    def __init__(
        self,
        broker: Any,
        *,
        loop: Any,
        host: str,
        transport_auth_token: bytes | None = None,
    ) -> None:
        self.transport_auth_token = transport_auth_token
        del broker, loop, host
        self.authority = _FakeAuthority()

    async def start(self) -> int:
        return 9443

    async def close(self) -> None:
        return None


class _Clock:
    def __init__(self) -> None:
        self.now = float(_NOW)

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    current = _Clock()
    monkeypatch.setattr(adapter_module, "_wall_clock", current)
    monkeypatch.setattr(runner_module, "_submission_clock", current)
    monkeypatch.setattr(runner_module, "_LAMBDA_CLIENT_TOKEN_RETRY_DELAYS_SECONDS", (0.0,) * 3)
    return current


@pytest.fixture(autouse=True)
def _skip_guest_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_guest_work(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def preflight(*_args: Any, **_kwargs: Any) -> datetime:
        return datetime.now(UTC)

    monkeypatch.setattr(adapter_module, "_install_ca", no_guest_work)
    monkeypatch.setattr(adapter_module, "run_setup_commands", no_guest_work)
    monkeypatch.setattr(adapter_module, "run_enforcement_preflight", preflight)
    monkeypatch.setattr(adapter_module, "_verify_agent_privilege_boundary", no_guest_work)


def _adapter(model: ClientTokenLambdaModel, tmp_path: Path, **options: Any):
    return LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=VpcTaskProxyExposure("10.0.1.20"),
        client=model,
        endpoint_transport_factory=lambda: SupervisorTransport(tmp_path),
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
        **options,
    )


def _factory(adapter: LambdaMicroVMEgressAdapter) -> VirtualEgressEnvironmentFactory:
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
        image=ClientTokenLambdaModel.image_arn,
    )


class _CrashingModel(ClientTokenLambdaModel):
    """Simulate worker death at an exact point around the provider submission."""

    def __init__(self, crash: str) -> None:
        super().__init__()
        self.crash = crash

    def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
        if self.crash == "before_provider":
            self.crash = ""
            raise _SimulatedProcessDeath("dispatched but the request never left the worker")
        response = super().run_microvm(**kwargs)
        if self.crash == "after_provider":
            self.crash = ""
            raise _SimulatedProcessDeath("provider created the MicroVM; acknowledgement lost")
        return response


def _live(model: ClientTokenLambdaModel) -> list[str]:
    return [
        identifier
        for identifier, microvm in model.microvms.items()
        if microvm["state"] not in {"TERMINATED"}
    ]


async def _crash_first_attempt(store: Any, session: Any, model: ClientTokenLambdaModel, factory):
    with pytest.raises(_SimulatedProcessDeath):
        await _resolve(store, session, factory, operation=EnvironmentFactoryOperation.CREATE)
    checkpoint = await store.load_checkpoint(session.id) or {}
    pending = _pending_record(checkpoint)
    assert pending is not None and pending["state"] == "dispatched"
    return pending


@pytest.mark.parametrize("crash", ["before_provider", "after_provider"])
def test_recovery_within_replay_window_adopts_exactly_one_microvm(
    crash: str, clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _CrashingModel(crash)
        pending = await _crash_first_attempt(
            store, session, model, _factory(_adapter(model, tmp_path))
        )
        created_before_recovery = model.created_ids()
        assert len(created_before_recovery) == (1 if crash == "after_provider" else 0)
        adapter_metadata = pending["intent"]["provider_metadata"]["adapter"]
        assert adapter_metadata["image_version"] == model.latest_image_version
        assert adapter_metadata["prepared_at_s"] == _NOW

        clock.now += DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS - 1
        model.latest_image_version = "4"  # a newer image must not leak into recovery
        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, tmp_path)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert resolution.error is None
        assert len(model.created_ids()) == 1
        if created_before_recovery:
            assert model.created_ids() == created_before_recovery
        tokens = {call["clientToken"] for call in model.run_calls}
        assert len(tokens) == 1
        assert {call["imageVersion"] for call in model.run_calls} == {"3"}
        published = await store.load_checkpoint(session.id)
        assert published is not None and _pending_record(published) is None

    asyncio.run(run())


def test_recovery_after_replay_window_refuses_without_provider_submission(
    clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _CrashingModel("after_provider")
        await _crash_first_attempt(store, session, model, _factory(_adapter(model, tmp_path)))
        submissions = len(model.run_calls)

        clock.now += DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS + 1
        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, tmp_path)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert resolution.error is not None
        assert len(model.run_calls) == submissions
        assert len(model.created_ids()) == 1
        checkpoint = await store.load_checkpoint(session.id) or {}
        pending = _pending_record(checkpoint)
        assert pending is not None and pending["state"] == "dispatched"

    asyncio.run(run())


async def _reap(store: Any, session: Any, factory: VirtualEgressEnvironmentFactory):
    checkpoint = await store.load_checkpoint(session.id)
    coordinator = EnvironmentAllocationCoordinator(
        session_store=store,
        checkpoint_transform=lambda candidate: lambda _session, _current: candidate,
        secret_redactor=SecretRedactor(),
    )
    record = coordinator.record_from_checkpoint(checkpoint, environment_name=_ENVIRONMENT_NAME)
    assert record is not None
    allocation = coordinator.context(
        session_id=session.id,
        inherited_owner_session_id=None,
        environment_name=_ENVIRONMENT_NAME,
        scope=record.intent.scope,
        existing=record,
    )
    request = EnvironmentFactoryRequest(
        session_id=session.id,
        agent_name="agent",
        environment_name=_ENVIRONMENT_NAME,
    )
    try:
        await factory.reap_allocation(request, allocation)
    finally:
        durable = await store.load_checkpoint(session.id) or {}
    return allocation, _pending_record(durable)


@pytest.mark.parametrize("crash", ["before_provider", "after_provider"])
def test_reap_within_replay_window_terminates_the_token_owned_microvm(
    crash: str, clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _CrashingModel(crash)
        await _crash_first_attempt(store, session, model, _factory(_adapter(model, tmp_path)))

        clock.now += 60
        allocation, pending = await _reap(store, session, _factory(_adapter(model, tmp_path)))

        assert allocation.state is EnvironmentAllocationState.REAPED
        assert pending is not None and pending["state"] == "reaped"
        assert len(model.created_ids()) == 1
        assert _live(model) == []
        assert len({call["clientToken"] for call in model.run_calls}) == 1

    asyncio.run(run())


def test_reap_between_replay_window_and_lifetime_bound_stays_pending(
    clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _CrashingModel("after_provider")
        model.clock = clock
        await _crash_first_attempt(store, session, model, _factory(_adapter(model, tmp_path)))
        submissions = len(model.run_calls)

        clock.now += DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS + 1
        with pytest.raises(LambdaMicroVMAllocationRecoveryError, match="remains pending"):
            await _reap(store, session, _factory(_adapter(model, tmp_path)))
        checkpoint = await store.load_checkpoint(session.id) or {}
        pending = _pending_record(checkpoint)
        assert pending is not None and pending["state"] == "reaping"
        assert len(model.run_calls) == submissions
        assert model.terminate_calls == []

        clock.now = (
            _NOW
            + DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS
            + runner_module.LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS
            + LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS
            + adapter_module._LIFETIME_EXPIRY_MARGIN_SECONDS
        )
        allocation, pending = await _reap(store, session, _factory(_adapter(model, tmp_path)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert pending is not None and pending["state"] == "reaped"
        assert len(model.run_calls) == submissions
        assert model.terminate_calls == []

    asyncio.run(run())


def test_reap_retries_after_unconfirmed_termination(
    clock: _Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_module, "_REAP_TERMINATION_TIMEOUT_SECONDS", 0.05)

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _CrashingModel("after_provider")
        await _crash_first_attempt(store, session, model, _factory(_adapter(model, tmp_path)))
        model.terminal_after_polls = 1_000_000

        with pytest.raises(runner_module.LambdaMicroVMError, match="did not reach TERMINATED"):
            await _reap(store, session, _factory(_adapter(model, tmp_path)))
        checkpoint = await store.load_checkpoint(session.id) or {}
        pending = _pending_record(checkpoint)
        assert pending is not None and pending["state"] == "reaping"

        model.terminal_after_polls = 0
        for microvm in model.microvms.values():
            microvm["polls_until_terminated"] = 0
        allocation, pending = await _reap(store, session, _factory(_adapter(model, tmp_path)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert _live(model) == []
        assert len(model.created_ids()) == 1

    asyncio.run(run())


def test_prepare_pins_exact_image_lifetime_and_window(clock: _Clock, tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    adapter = _adapter(model, tmp_path, client_token_replay_window_s=120)
    preparation = VirtualEgressAllocationPreparation(
        allocation_id=f"ealloc_{'a' * 32}",
        session_id="session",
        environment_name="env",
        image="arn:aws:lambda:us-east-1:123:microvm-image:cayu",
    )

    metadata = asyncio.run(adapter.prepare_allocation_metadata(preparation))

    assert metadata == {
        "version": 1,
        "image_arn": ClientTokenLambdaModel.image_arn,
        "image_version": "3",
        "maximum_duration_s": LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS,
        "prepared_at_s": _NOW,
        "replay_window_s": 120,
    }
    assert model.image_calls == [preparation.image]

    pinned = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=VpcTaskProxyExposure("10.0.1.20"),
        client=model,
        runner_options={"image_version": "9", "maximum_duration_in_seconds": 900},
    )
    pinned_metadata = asyncio.run(pinned.prepare_allocation_metadata(preparation))
    assert pinned_metadata["image_version"] == "9"
    assert pinned_metadata["maximum_duration_s"] == 900
    assert model.image_calls == [preparation.image]


@pytest.mark.parametrize("window", [0, 3_601, 1.5])
def test_replay_window_is_bounded(window: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="client_token_replay_window_s"):
        _adapter(ClientTokenLambdaModel(), tmp_path, client_token_replay_window_s=window)


def test_client_token_is_adapter_owned(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="client_token"):
        LambdaMicroVMEgressAdapter(
            region_name="us-east-1",
            egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
            exposure=VpcTaskProxyExposure("10.0.1.20"),
            runner_options={"client_token": "caller-chosen"},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"version": 2},
        {
            "version": 1,
            "image_arn": "image-name",
            "image_version": "3",
            "maximum_duration_s": 600,
            "prepared_at_s": _NOW,
            "replay_window_s": 300,
        },
        {
            "version": 1,
            "image_arn": ClientTokenLambdaModel.image_arn,
            "image_version": "3",
            "maximum_duration_s": 28_801,
            "prepared_at_s": _NOW,
            "replay_window_s": 300,
        },
    ],
)
def test_malformed_pinned_metadata_fails_before_provider_mutation(
    metadata: dict[str, Any], clock: _Clock, tmp_path: Path
) -> None:
    model = ClientTokenLambdaModel()
    adapter = _adapter(model, tmp_path)

    with pytest.raises(LambdaMicroVMAllocationRecoveryError):
        asyncio.run(
            adapter.reap_allocation(
                VirtualEgressAllocationReap(
                    allocation_id=f"ealloc_{'a' * 32}",
                    session_id="session",
                    environment_name="env",
                    image=ClientTokenLambdaModel.image_arn,
                    allocation_metadata=metadata,
                )
            )
        )
    with pytest.raises(LambdaMicroVMAllocationRecoveryError):
        asyncio.run(
            adapter.create_or_recover_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox",
                    runner_kind="lambda-microvm",
                    image=ClientTokenLambdaModel.image_arn,
                    binding=adapter_module.EgressBinding(proxy_url="http://10.0.1.20:9443"),
                    env_overlay={},
                    ca_cert_host_path=str(tmp_path / "ca.pem"),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("example.com",),
                    allocation_id=f"ealloc_{'a' * 32}",
                    allocation_metadata=metadata,
                ),
                allow_create=True,
            )
        )
    assert model.run_calls == []
    assert model.terminate_calls == []


def test_acknowledged_identity_is_terminated_directly(clock: _Clock, tmp_path: Path) -> None:
    async def run() -> None:
        model = ClientTokenLambdaModel()
        acknowledged = model.run_microvm(imageIdentifier=model.image_arn)["microvmId"]
        model.run_calls.clear()
        adapter = _adapter(model, tmp_path)
        clock.now += DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS + 1

        await adapter.reap_allocation(
            VirtualEgressAllocationReap(
                allocation_id=f"ealloc_{'a' * 32}",
                session_id="session",
                environment_name="env",
                image=model.image_arn,
                allocation_metadata={
                    "version": 1,
                    "image_arn": model.image_arn,
                    "image_version": "3",
                    "maximum_duration_s": 600,
                    "prepared_at_s": _NOW,
                    "replay_window_s": 300,
                },
                acknowledged_identity={"microvm_id": acknowledged},
            )
        )

        assert model.run_calls == []
        assert model.terminate_calls == [acknowledged]
        assert _live(model) == []

    asyncio.run(run())


class _CrashBeforeDispatchAdapter(LambdaMicroVMEgressAdapter):
    crashed = False

    async def prepare(self, **kwargs: Any):  # type: ignore[override]
        if not type(self).crashed:
            type(self).crashed = True
            raise _SimulatedProcessDeath("intent prepared; dispatch never fenced")
        return await super().prepare(**kwargs)


def _crash_before_dispatch_adapter(model: ClientTokenLambdaModel, tmp_path: Path):
    _CrashBeforeDispatchAdapter.crashed = False
    return _CrashBeforeDispatchAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=VpcTaskProxyExposure("10.0.1.20"),
        client=model,
        endpoint_transport_factory=lambda: SupervisorTransport(tmp_path),
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
    )


def test_prepared_intent_reap_precludes_dispatch_without_provider_calls(
    clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = ClientTokenLambdaModel()
        adapter = _crash_before_dispatch_adapter(model, tmp_path)
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store, session, _factory(adapter), operation=EnvironmentFactoryOperation.CREATE
            )
        checkpoint = await store.load_checkpoint(session.id) or {}
        pending = _pending_record(checkpoint)
        assert pending is not None and pending["state"] == "prepared"

        allocation, pending = await _reap(store, session, _factory(adapter))

        assert allocation.state is EnvironmentAllocationState.REAPED
        assert allocation.dispatch_precluded
        assert pending is not None and pending["state"] == "reaped"
        assert model.run_calls == [] and model.terminate_calls == []

    asyncio.run(run())


def test_prepared_intent_recovery_dispatches_the_first_submission(
    clock: _Clock, tmp_path: Path
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = ClientTokenLambdaModel()
        adapter = _crash_before_dispatch_adapter(model, tmp_path)
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store, session, _factory(adapter), operation=EnvironmentFactoryOperation.CREATE
            )

        clock.now += 30
        resolution = await _resolve(
            store, session, _factory(adapter), operation=EnvironmentFactoryOperation.CREATE
        )

        assert resolution.error is None
        assert len(model.created_ids()) == 1
        assert len(model.run_calls) == 1

    asyncio.run(run())


_ALLOCATION_ID = f"ealloc_{'b' * 32}"


def _pinned_metadata() -> adapter_module._AllocationMetadata:
    return {
        "version": 1,
        "image_arn": ClientTokenLambdaModel.image_arn,
        "image_version": "3",
        "maximum_duration_s": 600,
        "prepared_at_s": _NOW,
        "replay_window_s": 300,
    }


def _creation(
    adapter: LambdaMicroVMEgressAdapter,
    tmp_path: Path,
    metadata: adapter_module._AllocationMetadata,
):
    return adapter.create_or_recover_runner(
        VirtualEgressRunnerRequest(
            name="sandbox",
            runner_kind="lambda-microvm",
            image=ClientTokenLambdaModel.image_arn,
            binding=adapter_module.EgressBinding(proxy_url="http://10.0.1.20:9443"),
            env_overlay={},
            ca_cert_host_path=str(tmp_path / "ca.pem"),
            guest_ca_path="/etc/cayu/ca.pem",
            setup_commands=(),
            egress_destinations=("example.com",),
            allocation_id=_ALLOCATION_ID,
            allocation_metadata=metadata,
        ),
        allow_create=True,
    )


async def _reap_unacknowledged(
    adapter: LambdaMicroVMEgressAdapter, metadata: adapter_module._AllocationMetadata
) -> None:
    await adapter.reap_allocation(
        VirtualEgressAllocationReap(
            allocation_id=_ALLOCATION_ID,
            session_id="session",
            environment_name="sandbox",
            image=ClientTokenLambdaModel.image_arn,
            allocation_metadata=metadata,
        )
    )


def test_reaped_intent_excludes_a_dispatch_that_stalls_before_signing(
    clock: _Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker admitted inside the window but not yet sent can never allocate later."""

    model = ClientTokenLambdaModel()
    model.clock = clock
    metadata = _pinned_metadata()
    entered = threading.Event()
    release = threading.Event()

    def stalled_before_sending() -> float:
        entered.set()
        assert release.wait(10)
        return clock()

    monkeypatch.setattr(runner_module, "_submission_clock", stalled_before_sending)

    async def run() -> None:
        adapter = _adapter(model, tmp_path)
        creation = asyncio.create_task(_creation(adapter, tmp_path, metadata))
        assert await asyncio.to_thread(entered.wait, 5)
        clock.now = adapter_module._lifetime_deadline(metadata) + 1

        await _reap_unacknowledged(adapter, metadata)
        release.set()
        with pytest.raises(runner_module.LambdaMicroVMSubmissionClosed):
            await creation

        assert model.run_calls == []
        assert model.live_ids() == []

    asyncio.run(run())


def test_reaped_intent_excludes_a_request_signed_in_time_but_delivered_late(
    clock: _Clock, tmp_path: Path
) -> None:
    """The review's schedule: the request left the worker in time and reached AWS late."""

    metadata = _pinned_metadata()
    entered = threading.Event()
    release = threading.Event()

    class _HeldInFlight(ClientTokenLambdaModel):
        def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
            assert self.clock is not None
            signed_at = self.clock()
            entered.set()
            assert release.wait(10)
            return self.arrive(signed_at, **kwargs)

    model = _HeldInFlight()
    model.clock = clock

    async def run() -> None:
        adapter = _adapter(model, tmp_path)
        creation = asyncio.create_task(_creation(adapter, tmp_path, metadata))
        assert await asyncio.to_thread(entered.wait, 5)
        clock.now = adapter_module._lifetime_deadline(metadata) + 1

        await _reap_unacknowledged(adapter, metadata)
        release.set()
        with pytest.raises(FakeLambdaClientError, match="Signature expired"):
            await creation

        assert model.live_ids() == []

    asyncio.run(run())


def test_retry_that_crosses_the_submission_deadline_is_never_sent(
    clock: _Clock, tmp_path: Path
) -> None:
    metadata = _pinned_metadata()

    class _SlowThrottle(ClientTokenLambdaModel):
        def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
            clock.now = adapter_module._replay_deadline(metadata) + 1
            return super().run_microvm(**kwargs)

    model = _SlowThrottle()
    model.clock = clock
    model.run_failures.append(FakeLambdaClientError("ThrottlingException", "throttled"))

    async def run() -> None:
        adapter = _adapter(model, tmp_path)
        with pytest.raises(runner_module.LambdaMicroVMSubmissionClosed):
            await _creation(adapter, tmp_path, metadata)
        assert len(model.run_calls) == 1

        clock.now = adapter_module._lifetime_deadline(metadata)
        await _reap_unacknowledged(adapter, metadata)
        assert model.live_ids() == []

    asyncio.run(run())


def test_reap_by_lifetime_stays_pending_without_aws_time(clock: _Clock, tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    metadata = _pinned_metadata()
    clock.now = adapter_module._lifetime_deadline(metadata) + 3_600

    with pytest.raises(LambdaMicroVMAllocationRecoveryError, match="remains pending"):
        asyncio.run(_reap_unacknowledged(_adapter(model, tmp_path), metadata))
    assert model.run_calls == []


def test_reap_by_lifetime_uses_aws_time_not_the_local_clock(clock: _Clock, tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    metadata = _pinned_metadata()
    aws_now = [float(adapter_module._lifetime_deadline(metadata) - 60)]
    model.clock = lambda: aws_now[0]
    # This host's clock runs ahead of AWS's.
    clock.now = adapter_module._lifetime_deadline(metadata) + 60

    with pytest.raises(LambdaMicroVMAllocationRecoveryError, match="remains pending"):
        asyncio.run(_reap_unacknowledged(_adapter(model, tmp_path), metadata))

    aws_now[0] = float(adapter_module._lifetime_deadline(metadata))
    asyncio.run(_reap_unacknowledged(_adapter(model, tmp_path), metadata))

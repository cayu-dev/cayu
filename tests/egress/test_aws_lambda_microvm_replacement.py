"""Lambda MicroVM replacement restores the durable workspace into a fresh MicroVM."""

from __future__ import annotations

import asyncio
import base64
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import (
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
)
from tests.egress.test_aws_lambda_microvm_recoverable_allocation import (
    _Clock,
    _FakeProxyServer,
    _skip_guest_preflight,
    clock,
)
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, SupervisorTransport

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.artifacts import LocalArtifactStore
from cayu.egress import (
    HttpEgressPolicy,
    InvalidEgressReconnectMetadataError,
    UnsupportedEgressError,
    VirtualCredentialSpec,
    VirtualEgressAllocationPreparation,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.environments.bindings import BoundWorkspace
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY as RECONNECT,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.vaults import SecretRef, StaticVault
from cayu.workspaces import LocalWorkspace, RunnerWorkspace
from cayu.workspaces.checkpoint_lifecycle import WORKSPACE_CHECKPOINTS_KEY
from cayu.workspaces.checkpoints import WorkspaceCheckpointPolicy
from cayu.workspaces.revisions import WorkspaceWriterIsolationStatus

__all__ = ["_skip_guest_preflight", "clock"]
_ENV = "sandbox"


class _VolatileDiskTransport(SupervisorTransport):
    """All MicroVMs share one host directory; losing a MicroVM wipes it."""


def _adapter(model: ClientTokenLambdaModel, disk: Path) -> LambdaMicroVMEgressAdapter:
    return LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=VpcTaskProxyExposure("10.0.1.20"),
        client=model,
        endpoint_transport_factory=lambda: _VolatileDiskTransport(disk),
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0, "default_cwd": str(disk)},
    )


class _Scenario:
    def __init__(self, tmp_path: Path) -> None:
        self.disk = tmp_path / "microvm-disk"
        self.disk.mkdir()
        self.model = ClientTokenLambdaModel()
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        self.store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        self.artifacts = LocalArtifactStore(tmp_path / "artifacts")
        self.adapter = _adapter(self.model, self.disk)
        factory = VirtualEgressEnvironmentFactory(
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
            adapter=self.adapter,
            image=ClientTokenLambdaModel.image_arn,
            workspace_factory=lambda runner: RunnerWorkspace(runner, workspace_id="sandbox"),
            artifact_store=self.artifacts,
            execution_profile_identity=identity("lambda-replacement-factory"),
        )
        self.app = CayuApp(
            session_store=self.store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        self.app.register_provider(
            _SingleToolProvider(
                tool_name="public_workspace_write", arguments={"path": "created.txt"}
            ),
            default=True,
        )
        self.app.register_environment_factory(
            EnvironmentSpec(
                name=_ENV,
                execution_profile_identity=identity("lambda-replacement-environment"),
                workspace_checkpoint_policy=WorkspaceCheckpointPolicy(
                    allocation_replacement="restore"
                ),
            ),
            factory,
            default=True,
            artifact_store=self.artifacts,
        )
        self.app.register_agent(
            AgentSpec(name="agent", model="scripted-model"), tools=[_PublicWorkspaceWriteTool()]
        )
        self.private = runtime_checkpoint_session_store(self.store)

    async def events(self, stream) -> list[Any]:
        return [event async for event in stream]


def test_expired_microvm_is_replaced_on_its_pinned_image_with_workspace_restored(
    clock: _Clock, tmp_path: Path
) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path)
        try:
            events = await state.events(
                state.app.run(
                    RunRequest(
                        session_id="replace",
                        agent_name="agent",
                        messages=[Message.text("user", "write")],
                    )
                )
            )
            assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
            checkpoint = await state.private.load_checkpoint("replace")
            original = checkpoint[RECONNECT][_ENV]["identity"]
            receipt = checkpoint[WORKSPACE_CHECKPOINTS_KEY][_ENV]
            assert receipt["phase"] == "durable"
            assert receipt["isolation_mechanism"] == "lambda-microvm-owner-fence"
            assert receipt["isolation_generation"].startswith(original["microvm_id"] + ":")

            await state.store.update_status("replace", SessionStatus.INTERRUPTED)
            # AWS ended the MicroVM (maximum duration); its disk is gone and a
            # newer image version became active in the meantime.
            state.model.microvms[original["microvm_id"]]["state"] = "TERMINATED"
            shutil.rmtree(state.disk)
            state.disk.mkdir()
            state.model.latest_image_version = "4"
            clock.now += 3600

            events = await state.events(
                state.app.resume(
                    ResumeRequest(session_id="replace", messages=[Message.text("user", "go on")])
                )
            )

            assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
            checkpoint = await state.private.load_checkpoint("replace")
            replacement = checkpoint[RECONNECT][_ENV]["identity"]
            assert replacement["microvm_id"] != original["microvm_id"]
            assert replacement["image_version"] == original["image_version"] == "3"
            assert state.model.microvms[replacement["microvm_id"]]["imageVersion"] == "3"
            assert (state.disk / "created.txt").read_bytes() == b"public"
            receipt = checkpoint[WORKSPACE_CHECKPOINTS_KEY][_ENV]
            assert receipt["phase"] == "durable"
            assert receipt["isolation_generation"].startswith(replacement["microvm_id"] + ":")
            generation = checkpoint["environment_factory_allocation_generations"][_ENV]
            assert generation["reason"] == "replacement"
            assert generation["reconnect_metadata"]["identity"] == original
        finally:
            await state.store.close()

    asyncio.run(scenario())


def test_running_microvm_is_never_replaced(clock: _Clock, tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path)
        try:
            await state.events(
                state.app.run(
                    RunRequest(
                        session_id="replace",
                        agent_name="agent",
                        messages=[Message.text("user", "write")],
                    )
                )
            )
            original = (await state.private.load_checkpoint("replace"))[RECONNECT][_ENV]
            # Model an interrupted invocation: finalization suspended (rather
            # than terminated) the MicroVM, which therefore still exists.
            microvm = state.model.microvms[original["identity"]["microvm_id"]]
            microvm.update(state="SUSPENDED", polls_until_terminated=None)
            await state.store.update_status("replace", SessionStatus.INTERRUPTED)

            events = await state.events(
                state.app.resume(
                    ResumeRequest(session_id="replace", messages=[Message.text("user", "go on")])
                )
            )

            assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
            assert len(state.model.created_ids()) == 1
            assert (await state.private.load_checkpoint("replace"))[RECONNECT][_ENV] == original
        finally:
            await state.store.close()

    asyncio.run(scenario())


def _preparation(**overrides: Any) -> VirtualEgressAllocationPreparation:
    predecessor = {
        "microvm_id": "microvm-old",
        "endpoint": "microvm-old.lambda-microvm.invalid",
        "region": "us-east-1",
        "image_identifier": ClientTokenLambdaModel.image_arn,
        "image_version": "3",
        "session_id": "s",
        "environment_name": "e",
    }
    predecessor.update(overrides)
    return VirtualEgressAllocationPreparation(
        allocation_id=f"ealloc_{'a' * 32}",
        session_id="s",
        environment_name="e",
        image=ClientTokenLambdaModel.image_arn,
        predecessor_identity=predecessor,
    )


def test_replacement_pins_predecessor_image_version(clock: _Clock, tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    model.latest_image_version = "9"

    metadata = asyncio.run(_adapter(model, tmp_path).prepare_allocation_metadata(_preparation()))

    assert metadata["image_version"] == "3"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"image_identifier": "arn:aws:lambda:us-east-1:123:microvm-image:other"},
            UnsupportedEgressError,
        ),
        ({"region": "us-west-2"}, UnsupportedEgressError),
        ({"auth_token": "jwe"}, InvalidEgressReconnectMetadataError),
    ],
)
def test_incompatible_predecessor_fails_before_provider_mutation(
    overrides: dict[str, Any], error: type[Exception], clock: _Clock, tmp_path: Path
) -> None:
    model = ClientTokenLambdaModel()

    with pytest.raises(error):
        asyncio.run(
            _adapter(model, tmp_path).prepare_allocation_metadata(_preparation(**overrides))
        )

    assert model.run_calls == []


def test_writer_isolation_follows_the_owner_fence(tmp_path: Path) -> None:
    async def scenario() -> None:
        from cayu import LambdaMicroVMRunner

        model = ClientTokenLambdaModel()
        sidecar = SupervisorTransport(tmp_path)
        adapter = _adapter(model, tmp_path)
        unclaimed = LambdaMicroVMRunner(
            model, microvm_id="microvm-x", endpoint="microvm-x.invalid", endpoint_transport=sidecar
        )
        assert adapter.observe_writer_isolation(unclaimed).status is (
            WorkspaceWriterIsolationStatus.UNKNOWN
        )

        owner = await LambdaMicroVMRunner.create(
            model.image_arn,
            client=model,
            endpoint_transport=sidecar,
            default_cwd=str(tmp_path),
            poll_interval_s=0,
            close_action="none",
        )
        evidence = adapter.observe_writer_isolation(owner)
        assert evidence.status is WorkspaceWriterIsolationStatus.EXCLUSIVE
        assert evidence.mechanism == "lambda-microvm-owner-fence"
        assert evidence.generation == f"{owner.microvm_id}:1"

        successor = await LambdaMicroVMRunner.from_existing(
            owner.microvm_id,
            client=model,
            endpoint_transport=sidecar,
            default_cwd=str(tmp_path),
            poll_interval_s=0,
            close_action="none",
        )
        assert adapter.observe_writer_isolation(successor).generation == (f"{owner.microvm_id}:2")
        # The stale owner learns of supersession on its next guest operation.
        with pytest.raises(Exception):
            await owner.exec(adapter_module.ExecCommand.process("true"))
        assert adapter.observe_writer_isolation(owner).status is (
            WorkspaceWriterIsolationStatus.UNKNOWN
        )

    asyncio.run(scenario())


def test_non_passthrough_inner_binding_keeps_isolation_authority(tmp_path: Path) -> None:
    from cayu.egress.runtime import _EgressTeardownBinding
    from cayu.environments.bindings import NativeBinding

    class SharedMount(NativeBinding):
        """A mounted filesystem other MicroVMs can also write."""

    binding = _EgressTeardownBinding.__new__(_EgressTeardownBinding)
    binding._inner = SharedMount()
    binding._runner = object()  # type: ignore[assignment]
    bound = BoundWorkspace(workspace=LocalWorkspace(tmp_path, workspace_id="shared"))

    assert binding.observe_writer_isolation(bound).status is (
        WorkspaceWriterIsolationStatus.UNKNOWN
    )

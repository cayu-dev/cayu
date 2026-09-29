"""Opt-in allocation replacement restores the last durable workspace checkpoint."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core.test_environment_allocation_recovery import (
    _FakeRemoteFactory,
    _FakeRemoteProvider,
    _FakeResource,
)
from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import (
    _ExclusiveWriterBinding,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
)

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.artifacts import LocalArtifactStore
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY as RECONNECT,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.checkpoint_lifecycle import WORKSPACE_CHECKPOINTS_KEY
from cayu.workspaces.checkpoints import WorkspaceCheckpointPolicy

_ENV = "remote"
_GENERATIONS = "environment_factory_allocation_generations"


class _WorkspaceFactory(_FakeRemoteFactory):
    """Fake remote allocations whose workspace is a directory owned by the resource."""

    execution_profile_identity = identity("replacement-factory")

    def __init__(self, provider, root: Path, artifacts, *, disposed: Any) -> None:
        super().__init__(provider)
        self.root = root
        self.artifacts = artifacts
        self.disposed = disposed
        self.disposal_checks: list[EnvironmentFactoryRequest] = []

    async def is_allocation_disposed(self, request: EnvironmentFactoryRequest) -> bool:
        assert request.operation is EnvironmentFactoryOperation.RECONNECT
        self.disposal_checks.append(request)
        if isinstance(self.disposed, BaseException):
            raise self.disposed
        return self.disposed

    def _result(self, request, resource: _FakeResource, *, allocation=None):
        base = super()._result(request, resource, allocation=allocation)
        directory = self.root / resource.resource_name
        directory.mkdir(parents=True, exist_ok=True)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(directory, workspace_id=resource.resource_name),
                binding=_ExclusiveWriterBinding(),
                artifact_store=self.artifacts,
            ),
            reconnect_metadata=base.reconnect_metadata,
            release=base.release,
        )


def _spec(replacement: str) -> EnvironmentSpec:
    return EnvironmentSpec(
        name=_ENV,
        execution_profile_identity=identity("replacement-environment"),
        workspace_checkpoint_policy=WorkspaceCheckpointPolicy(allocation_replacement=replacement),
    )


class _Scenario:
    def __init__(self, tmp_path: Path, *, replacement: str, disposed: Any) -> None:
        self.tmp_path = tmp_path
        # Factory workspaces publish dynamic observation authority aliases.
        self.keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        self.store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(self.keyring),
        )
        self.artifacts = LocalArtifactStore(tmp_path / "artifacts")
        self.remote = _FakeRemoteProvider()
        self.factory = _WorkspaceFactory(
            self.remote, tmp_path / "allocations", self.artifacts, disposed=disposed
        )
        self.provider = _SingleToolProvider(
            tool_name="public_workspace_write", arguments={"path": "created.txt"}
        )
        self.app = CayuApp(
            session_store=self.store,
            enable_logging=False,
            public_authority_alias_keyring=self.keyring,
        )
        self.app.register_provider(self.provider, default=True)
        self.app.register_environment_factory(
            _spec(replacement), self.factory, default=True, artifact_store=self.artifacts
        )
        self.app.register_agent(
            AgentSpec(name="agent", model="scripted-model"), tools=[_PublicWorkspaceWriteTool()]
        )
        self.private = runtime_checkpoint_session_store(self.store)

    async def run(self):
        return [
            event
            async for event in self.app.run(
                RunRequest(
                    session_id="replace",
                    agent_name="agent",
                    messages=[Message.text("user", "write the file")],
                )
            )
        ]

    async def resume(self):
        return [
            event
            async for event in self.app.resume(
                ResumeRequest(session_id="replace", messages=[Message.text("user", "continue")])
            )
        ]

    async def interrupt_and_lose_allocation(self) -> dict[str, Any]:
        checkpoint = await self.private.load_checkpoint("replace")
        original = checkpoint[RECONNECT][_ENV]
        await self.store.update_status("replace", SessionStatus.INTERRUPTED)
        # The sandbox expired: the provider no longer knows the resource.
        self.remote.resources.clear()
        return original

    def workspace(self, resource_name: str) -> Path:
        return self.tmp_path / "allocations" / resource_name


def test_disposal_proof_replaces_allocation_and_restores_checkpoint(tmp_path) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path, replacement="restore", disposed=True)
        try:
            events = await state.run()
            assert events[-1].type == EventType.SESSION_COMPLETED
            before = await state.store.load("replace")
            original = await state.interrupt_and_lose_allocation()
            original_resource = original["resource_name"]
            assert (state.workspace(original_resource) / "created.txt").read_bytes() == b"public"

            events = await state.resume()

            assert events[-1].type == EventType.SESSION_COMPLETED
            assert len(state.factory.disposal_checks) == 1
            assert state.factory.disposal_checks[0].reconnect_metadata == original
            create = state.factory.requests[-1]
            assert create.operation is EnvironmentFactoryOperation.CREATE
            assert create.reconnect_metadata == {}
            assert create.replacement_predecessor == original
            checkpoint = await state.private.load_checkpoint("replace")
            replacement = checkpoint[RECONNECT][_ENV]
            assert replacement != original
            restored = state.workspace(replacement["resource_name"]) / "created.txt"
            assert restored.read_bytes() == b"public"
            generation = checkpoint[_GENERATIONS][_ENV]
            assert generation["reason"] == "replacement"
            assert generation["reconnect_metadata"] == original
            assert generation["session_instance_id"] == before.instance_id
            receipt = checkpoint[WORKSPACE_CHECKPOINTS_KEY][_ENV]
            assert receipt["phase"] == "durable"
            assert (await state.store.load("replace")).instance_id == before.instance_id
        finally:
            await state.store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "disposed", [False, RuntimeError("provider unavailable")], ids=["not_disposed", "unavailable"]
)
def test_missing_disposal_proof_never_replaces(tmp_path, disposed) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path, replacement="restore", disposed=disposed)
        try:
            await state.run()
            original = await state.interrupt_and_lose_allocation()
            creates = len(state.remote.create_calls)

            events = await state.resume()

            assert events[-1].type == EventType.SESSION_FAILED
            assert len(state.remote.create_calls) == creates
            checkpoint = await state.private.load_checkpoint("replace")
            assert checkpoint[RECONNECT][_ENV] == original
            assert _GENERATIONS not in checkpoint
        finally:
            await state.store.close()

    asyncio.run(scenario())


def test_replacement_is_opt_in(tmp_path) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path, replacement="never", disposed=True)
        try:
            await state.run()
            original = await state.interrupt_and_lose_allocation()

            events = await state.resume()

            assert events[-1].type == EventType.SESSION_FAILED
            assert state.factory.disposal_checks == []
            checkpoint = await state.private.load_checkpoint("replace")
            assert checkpoint[RECONNECT][_ENV] == original
        finally:
            await state.store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["mutating", "checkpointing"])
def test_unknown_workspace_mutation_blocks_replacement(tmp_path, phase) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path, replacement="restore", disposed=True)
        try:
            await state.run()
            original = await state.interrupt_and_lose_allocation()

            def unsettle(_session, current):
                current[WORKSPACE_CHECKPOINTS_KEY][_ENV]["phase"] = phase
                return current

            await state.private.transform_checkpoint("replace", unsettle)
            creates = len(state.remote.create_calls)

            events = await state.resume()

            assert events[-1].type == EventType.SESSION_FAILED
            assert state.factory.disposal_checks == []
            assert len(state.remote.create_calls) == creates
            checkpoint = await state.private.load_checkpoint("replace")
            assert checkpoint[RECONNECT][_ENV] == original
        finally:
            await state.store.close()

    asyncio.run(scenario())


def test_surviving_allocation_reconnects_without_replacement(tmp_path) -> None:
    async def scenario() -> None:
        state = _Scenario(tmp_path, replacement="restore", disposed=False)
        try:
            await state.run()
            checkpoint = await state.private.load_checkpoint("replace")
            original = checkpoint[RECONNECT][_ENV]
            await state.store.update_status("replace", SessionStatus.INTERRUPTED)

            events = await state.resume()

            assert events[-1].type == EventType.SESSION_COMPLETED
            assert len(state.factory.disposal_checks) == 1
            assert len(state.remote.create_calls) == 1
            assert state.factory.requests[-1].operation is EnvironmentFactoryOperation.RECONNECT
            checkpoint = await state.private.load_checkpoint("replace")
            assert checkpoint[RECONNECT][_ENV] == original
        finally:
            await state.store.close()

    asyncio.run(scenario())


def test_policy_serialization_keeps_established_identity() -> None:
    assert "allocation_replacement" not in WorkspaceCheckpointPolicy().model_dump(mode="json")
    assert (
        WorkspaceCheckpointPolicy(allocation_replacement="restore").model_dump(mode="json")[
            "allocation_replacement"
        ]
        == "restore"
    )
    assert WorkspaceCheckpointPolicy.model_validate(
        WorkspaceCheckpointPolicy(allocation_replacement="restore").model_dump(mode="json")
    ) == WorkspaceCheckpointPolicy(allocation_replacement="restore")
    with pytest.raises(ValueError):
        WorkspaceCheckpointPolicy(allocation_replacement="always")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation": EnvironmentFactoryOperation.RECONNECT},
        {"reconnect_metadata": {"resource_name": "other"}},
    ],
)
def test_replacement_predecessor_is_only_valid_on_fresh_create(overrides) -> None:
    with pytest.raises(ValueError, match="replace an allocation"):
        EnvironmentFactoryRequest(
            session_id="s",
            agent_name="a",
            environment_name="e",
            replacement_predecessor={"resource_name": "old"},
            **overrides,
        )

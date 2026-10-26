"""Workspace-mutating tools need an alias keyring with dynamic environments."""

from __future__ import annotations

import asyncio
import base64
import secrets
from pathlib import Path

import pytest
from pydantic import SecretStr

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.environments.factory import (
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
)
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.requests import ResumeRequest
from cayu.storage import SQLiteSessionStore
from cayu.tools import ExecCommandTool, ReadFileTool, StaticToolPolicy, WriteFileTool
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.vaults import StaticVault
from cayu.workspaces import LocalWorkspace


class _RootFactory(EnvironmentFactory):
    """Undeclared factory: Cayu treats it as dynamic."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def create(self, request):
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id=request.session_id),
                runner=LocalRunner(self.root),
            )
        )


class _StaticRootFactory(_RootFactory):
    @property
    def secret_resolution_scope(self):
        return "static"


def _keyring() -> PublicAuthorityAliasKeyring:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    return PublicAuthorityAliasKeyring(active_key_id="service", keys={"service": SecretStr(key)})


def _write_provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="c1",
                    name="write_file",
                    arguments={"path": "out.txt", "content": "hi\n", "mode": "create"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def _register_writer(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name="writer", model="scripted"),
        tools=[WriteFileTool()],
        tool_policy=StaticToolPolicy(allow=["write_file"]),
    )


class _RunnerOnlyVaultFactory(EnvironmentFactory):
    """Dynamic factory whose environments have a runner and a vault but no workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def create(self, request):
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=LocalRunner(self.root),
                vault=StaticVault({"token": "value"}),
            )
        )


async def _run_session(app: CayuApp, *, agent_name: str, session_id: str) -> list:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name=agent_name,
                session_id=session_id,
                environment_name="ws",
                max_steps=4,
                messages=[Message.text("user", "go")],
            )
        )
    ]


def test_dynamic_factory_workspace_fails_the_session_before_the_first_model_call(
    tmp_path: Path,
) -> None:
    """Registration can't see a factory's workspace; materialization refuses it."""

    root = tmp_path / "workspace"
    root.mkdir()
    provider = _write_provider()

    async def run():
        store = SQLiteSessionStore(tmp_path / "s.db")
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        # Registration succeeds: a factory's workspace is unknown until it is created.
        app.register_environment_factory(EnvironmentSpec(name="ws"), _RootFactory(root))
        _register_writer(app)
        try:
            return await _run_session(app, agent_name="writer", session_id="dynamic")
        finally:
            await store.close()

    events = asyncio.run(run())
    types = [event.type for event in events]
    assert provider.requests == []
    assert EventType.TOOL_CALL_STARTED not in types
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    message = failed.payload["error"]
    assert "Tool 'write_file' of agent 'writer'" in message
    assert "environment 'ws'" in message
    assert 'secret_resolution_scope = "static"' in message
    assert "public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)" in message
    assert "in-memory store makes its own key" in message
    assert not (root / "out.txt").exists()


def test_dynamic_factory_without_a_workspace_needs_no_keyring(tmp_path: Path) -> None:
    """exec_command in a vault-backed factory environment with no workspace keeps working."""

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="c1", name="exec_command", arguments={"argv": ["true"]}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )

    async def run():
        store = SQLiteSessionStore(tmp_path / "s.db")
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="ops", model="scripted"),
            tools=[ExecCommandTool()],
            tool_policy=StaticToolPolicy(allow=["exec_command"]),
        )
        app.register_environment_factory(
            EnvironmentSpec(name="ws"), _RunnerOnlyVaultFactory(tmp_path)
        )
        try:
            return await _run_session(app, agent_name="ops", session_id="no-workspace")
        finally:
            await store.close()

    events = asyncio.run(run())
    types = [event.type for event in events]
    assert EventType.SESSION_FAILED not in types
    assert EventType.TOOL_CALL_COMPLETED in types
    assert EventType.SESSION_COMPLETED in types


@pytest.mark.parametrize("agent_first", [True, False])
def test_registration_names_only_the_keyring_fix_for_an_environment_with_a_vault(
    tmp_path: Path, agent_first: bool
) -> None:
    app = CayuApp(enable_logging=False, session_store=SQLiteSessionStore(tmp_path / "s.db"))
    environment = Environment(
        EnvironmentSpec(name="vaulted"),
        workspace=LocalWorkspace(tmp_path, workspace_id="vaulted"),
        runner=LocalRunner(tmp_path),
        vault=StaticVault({"token": "value"}),
    )
    spec = AgentSpec(name="runner", model="scripted")
    if agent_first:
        app.register_agent(spec, tools=[ExecCommandTool()])
        with pytest.raises(RuntimeError) as caught:
            app.register_environment(environment)
        assert "vaulted" not in app._environment_registry.registrations
    else:
        app.register_environment(environment)
        with pytest.raises(RuntimeError) as caught:
            app.register_agent(spec, tools=[ExecCommandTool()])
        assert "runner" not in app._agent_registry.registrations
    message = str(caught.value)
    assert "Tool 'exec_command' of agent 'runner'" in message
    assert "environment 'vaulted'" in message
    assert "has a vault or credential proxy" in message
    assert 'secret_resolution_scope = "static"' not in message


@pytest.mark.parametrize("setup", ["static_factory", "static_environment", "keyring", "read_only"])
def test_registration_accepts_pairings_that_can_record_observations(
    tmp_path: Path, setup: str
) -> None:
    ring = _keyring() if setup == "keyring" else None
    store = SQLiteSessionStore(
        tmp_path / "s.db",
        public_authority_alias_codec=None if ring is None else PublicAuthorityAliasCodec(ring),
    )
    app = CayuApp(enable_logging=False, session_store=store, public_authority_alias_keyring=ring)
    spec = EnvironmentSpec(name="ws")
    if setup == "static_factory":
        app.register_environment_factory(spec, _StaticRootFactory(tmp_path))
    elif setup == "static_environment":
        app.register_environment(
            Environment(
                spec,
                workspace=LocalWorkspace(tmp_path, workspace_id="ws"),
                runner=LocalRunner(tmp_path),
            )
        )
    else:
        app.register_environment_factory(spec, _RootFactory(tmp_path))
    if setup == "read_only":
        app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[ReadFileTool()])
    else:
        _register_writer(app)


def test_unchecked_pairing_fails_only_the_tool_call_before_workspace_state(
    tmp_path: Path,
) -> None:
    """A pairing that skipped registration names the tool and environment at the call."""

    root = tmp_path / "workspace"
    root.mkdir()

    async def run():
        store = SQLiteSessionStore(tmp_path / "s.db")
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(_write_provider(), default=True)
        # Bypass the registration check to reach the runtime backstop: register
        # the agent first, then unlink the registries before the environment.
        _register_writer(app)
        app._environment_registry._agent_registrations = None
        app.register_environment(
            Environment(
                EnvironmentSpec(name="ws"),
                workspace=LocalWorkspace(root, workspace_id="ws"),
                runner=LocalRunner(root),
                vault=StaticVault({"token": "value"}),
            )
        )
        try:
            return await _run_session(app, agent_name="writer", session_id="unchecked")
        finally:
            await store.close()

    events = asyncio.run(run())
    types = [event.type for event in events]
    failed = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert failed.tool_name == "write_file"
    assert failed.environment_name == "ws"
    assert failed.payload["reason"] == "workspace_observation_keyring_unavailable"
    content = failed.payload["result"]["content"]
    assert "Tool 'write_file' cannot record workspace observations in environment 'ws'" in content
    assert failed.payload["result"]["structured"] == {
        "status": "rejected",
        "reason": "workspace_observation_keyring_unavailable",
    }
    assert EventType.SESSION_FAILED not in types
    assert types.index(EventType.TOOL_CALL_FAILED) < types.index(EventType.SESSION_COMPLETED)
    assert not (root / "out.txt").exists()


def test_tool_ceiling_without_workspace_tools_needs_no_keyring(tmp_path: Path) -> None:
    """A run whose ceiling excludes the mutating tools never records an observation."""

    root = tmp_path / "workspace"
    root.mkdir()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )

    async def run():
        store = SQLiteSessionStore(tmp_path / "s.db")
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="ws"), _RootFactory(root))
        app.register_agent(
            AgentSpec(name="writer", model="scripted"), tools=[ReadFileTool(), WriteFileTool()]
        )
        try:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id="ceiling",
                        environment_name="ws",
                        messages=[Message.text("user", "read only")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=["read_file"]),
                    )
                )
            ]
        finally:
            await store.close()

    types = [event.type for event in asyncio.run(run())]
    assert EventType.SESSION_FAILED not in types
    assert EventType.SESSION_COMPLETED in types


class _RecoverableRootFactory(EnvironmentFactory):
    """Dynamic factory whose published allocation a later process reconnects."""

    def __init__(self, root: Path, calls: list[str]) -> None:
        self.root = root
        self.calls = calls

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:keyring-reconnect-factory",
            behavior_version="1",
            implementation_version="1",
        )

    def allocation_scope(self, request):
        return EnvironmentAllocationScope(provider="fixture", adapter_generation="fixture.v1")

    async def create_recoverable(self, request, allocation):
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            await allocation.prepare({"resource": allocation.intent.allocation_id})
        if allocation.state is EnvironmentAllocationState.PREPARED:
            await allocation.mark_dispatched()
        self.calls.append("create")
        reconnect = {"resource": allocation.intent.allocation_id}
        await allocation.acknowledge(reconnect)
        return self._result(request, reconnect)

    async def create(self, request):
        assert request.operation is EnvironmentFactoryOperation.RECONNECT
        self.calls.append("reconnect")
        return self._result(request, request.reconnect_metadata)

    async def reap_allocation(self, request, allocation):
        self.calls.append("reap")
        if await allocation.mark_reaping():
            await allocation.mark_reaped()

    def _result(self, request, reconnect):
        async def release(action):
            del action

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id=request.session_id),
                runner=LocalRunner(self.root),
            ),
            reconnect_metadata=reconnect,
            release=release,
        )


def test_reconnecting_an_existing_allocation_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a fresh allocation is checked: a reconnect must stay possible.

    The first run stands in for a session started before the check existed.
    Resuming it reconnects that allocation, which must not fail on the missing
    keyring; finalization and recovery reconnect the same way to dispose of it.
    """

    import cayu.runtime._environment_lifecycle as lifecycle_module

    root = tmp_path / "workspace"
    root.mkdir()
    calls: list[str] = []

    def build(store: SQLiteSessionStore) -> CayuApp:
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(
            ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            ),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(
                name="ws",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:keyring-reconnect",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            _RecoverableRootFactory(root, calls),
            default=True,
        )
        _register_writer(app)
        return app

    async def first_run() -> None:
        store = SQLiteSessionStore(tmp_path / "s.db")
        try:
            async for _ in build(store).run(
                RunRequest(
                    agent_name="writer",
                    session_id="reconnect",
                    messages=[Message.text("user", "answer")],
                    max_steps=1,
                )
            ):
                pass
        finally:
            await store.close()

    async def resume() -> list:
        store = SQLiteSessionStore(tmp_path / "s.db")
        try:
            return [
                event
                async for event in build(store).resume(
                    ResumeRequest(
                        session_id="reconnect",
                        messages=[Message.text("user", "continue")],
                        max_steps=1,
                    )
                )
            ]
        finally:
            await store.close()

    with monkeypatch.context() as patch:
        patch.setattr(
            lifecycle_module,
            "require_materialized_workspace_observation_keyring",
            lambda **kwargs: None,
        )
        asyncio.run(first_run())
    events = asyncio.run(resume())
    types = [event.type for event in events]
    assert calls[:2] == ["create", "reconnect"]
    assert EventType.SESSION_FAILED not in types, [
        event.payload.get("error") for event in events if event.type is EventType.SESSION_FAILED
    ]
    assert EventType.SESSION_COMPLETED in types

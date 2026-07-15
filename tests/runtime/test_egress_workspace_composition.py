from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.workspaces.guard_harness import make_local_guard_exec

from cayu.egress import (
    EgressBinding,
    HttpEgressPolicy,
    SandboxEgressAdapter,
)
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners import (
    E2BRunner,
    E2BWorkspaceCapability,
    ExecCommand,
    LocalRunner,
    MicrosandboxRunner,
    MicrosandboxWorkspaceCapability,
    Runner,
)
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.vaults import SecretRef, StaticVault
from cayu.workspaces import (
    E2BWorkspace,
    MicrosandboxWorkspace,
    RunnerWorkspace,
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
)

pytest.importorskip("cryptography")


@dataclass(frozen=True)
class _MicrosandboxEntry:
    path: str
    kind: str


class _LocalMicrosandboxFilesystem:
    async def list(self, path: str) -> Sequence[_MicrosandboxEntry]:
        root = Path(path)
        return tuple(
            _MicrosandboxEntry(
                path=str(child),
                kind="dir" if child.is_dir() else "file",
            )
            for child in root.iterdir()
        )


class _LocalSftp:
    async def real_path(self, path: str) -> str:
        return str(Path(path).resolve())

    async def close(self) -> None:
        return None


class _LocalSshClient:
    async def sftp(self) -> _LocalSftp:
        return _LocalSftp()

    async def close(self) -> None:
        return None


class _LocalSsh:
    async def open_client(self, **_kwargs: Any) -> _LocalSshClient:
        return _LocalSshClient()


class _LocalMicrosandbox:
    def __init__(self) -> None:
        self.fs = _LocalMicrosandboxFilesystem()

    def ssh(self) -> _LocalSsh:
        return _LocalSsh()


@dataclass(frozen=True)
class _E2BEntry:
    path: str
    type: str
    symlink_target: str | None = None


class _LocalE2BFilesystem:
    async def get_info(self, path: str, **_kwargs: Any) -> _E2BEntry:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(path)
        return _E2BEntry(path=str(candidate), type="dir" if candidate.is_dir() else "file")

    async def list(
        self,
        path: str,
        *,
        depth: int | None,
        **_kwargs: Any,
    ) -> Sequence[_E2BEntry]:
        del depth
        root = Path(path)
        return tuple(
            _E2BEntry(path=str(child), type="dir" if child.is_dir() else "file")
            for child in root.rglob("*")
        )


class _LocalE2BSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "e2b_workspace_composition"
        self.files = _LocalE2BFilesystem()


class _WorkspaceAdapter(SandboxEgressAdapter):
    def __init__(self, runner_kind: str, runner: Runner, order: list[str]) -> None:
        self.runner_kind = runner_kind
        self.runner = runner
        self.order = order
        self.teardown_calls = 0

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        del session_id, grants, broker

        async def teardown() -> None:
            self.teardown_calls += 1
            self.order.append("binding")

        return EgressBinding(
            env={"HTTPS_PROXY": "http://proxy.invalid:8080"},
            ca_cert_pem=b"certificate",
            runner_kind=self.runner_kind,
            guest_ca_path="/etc/cayu/ca.pem",
            teardown=teardown,
        )

    async def create_runner(self, request):  # type: ignore[no-untyped-def]
        del request
        return self.runner

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        del outcome
        self.order.append("runner")
        await runner.close()


class _MinimalRunner(Runner):
    isolation = "test"

    @property
    def closed(self) -> bool:
        return self._closed

    async def exec(self, command: ExecCommand, **kwargs: Any):  # type: ignore[no-untyped-def]
        del command, kwargs
        raise NotImplementedError

    async def close(self) -> None:
        self._closed = True


class _IdentifiedRunner(_MinimalRunner):
    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("test", "managed")


class _PlainSpoofWorkspace(Workspace):
    id = "plain-spoof"

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> WorkspaceReadResult:
        del path, max_bytes
        raise NotImplementedError

    def bounded_read_limit(self, max_bytes: int) -> int:
        return max_bytes

    async def write_bytes(self, path: str, content: bytes) -> None:
        del path, content
        raise NotImplementedError

    async def delete(self, path: str) -> None:
        del path
        raise NotImplementedError

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        del pattern, limit
        raise NotImplementedError


class _MismatchedIdentityWorkspace(RunnerWorkspace):
    @property
    def bound_runner_resource_key(self) -> tuple[object, ...]:
        return ("test", "different")


def _factory(
    adapter: SandboxEgressAdapter,
    workspace_factory: Any,
) -> VirtualEgressEnvironmentFactory:
    return VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"provider_key": "sk_test_workspace_composition"}),
        policies={
            "provider": HttpEgressPolicy(
                name="provider",
                allowed_hosts=["api.example.com"],
                allowed_endpoints=[("GET", "/v1/data")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="PROVIDER_KEY",
                secret=SecretRef(name="provider_key"),
                destination="api.example.com",
                policy_name="provider",
            )
        ],
        adapter=adapter,
        workspace_factory=workspace_factory,
    )


def _local_exec(root: Path) -> Any:
    local_runner = LocalRunner(root)
    guard_exec = make_local_guard_exec()

    async def exec_command(command: ExecCommand, **kwargs: Any):  # type: ignore[no-untyped-def]
        argv = command.argv or []
        if argv and argv[0] == "python3" and len(argv) > 1 and argv[1] == "-c":
            return await guard_exec(command, **kwargs)
        return await local_runner.exec(command, **kwargs)

    return exec_command


@pytest.mark.parametrize("provider", ["microsandbox", "e2b"])
def test_factory_composes_provider_workspace_without_unwrapping(
    tmp_path: Path,
    provider: str,
) -> None:
    order: list[str] = []
    if provider == "microsandbox":
        inner = MicrosandboxRunner(
            _LocalMicrosandbox(),
            name="workspace-composition",
            default_cwd=str(tmp_path),
            close_action="none",
            sandbox_module=object(),
        )
        workspace_type = MicrosandboxWorkspace
        capability_type = MicrosandboxWorkspaceCapability
        private_identity_attribute = "name"
    else:
        inner = E2BRunner(
            _LocalE2BSandbox(),
            default_cwd=str(tmp_path),
            close_action="none",
            e2b_module=object(),
        )
        workspace_type = E2BWorkspace
        capability_type = E2BWorkspaceCapability
        private_identity_attribute = "sandbox_id"
    inner.exec = _local_exec(tmp_path)  # type: ignore[method-assign]
    adapter = _WorkspaceAdapter(provider, inner, order)

    async def run() -> tuple[Any, Any, Any]:
        result = await _factory(
            adapter,
            lambda runner: workspace_type(runner, root=str(tmp_path)),
        ).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_{provider}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        environment = result.environment
        managed = environment.runner
        workspace = environment.workspace
        binding = environment.binding
        assert managed is not None
        assert isinstance(workspace, workspace_type)
        assert binding is not None
        capability = managed.workspace_capability(capability_type)
        assert isinstance(capability, capability_type)
        assert not hasattr(capability, "close")
        assert not hasattr(capability, "filesystem")
        assert not hasattr(managed, private_identity_attribute)

        with pytest.raises(ValueError, match="escapes the workspace root"):
            await workspace.read_bytes("../escape.txt")
        await workspace.write_bytes("notes/a.txt", b"hello")
        assert (await workspace.read_bytes("notes/a.txt")).content == b"hello"
        assert (await workspace.list("**/*.txt")).paths == ("notes/a.txt",)
        with pytest.raises(ValueError, match="parent traversal"):
            workspace.resolve("notes/../notes/a.txt")
        assert workspace.resolve("notes/a.txt") == str(tmp_path / "notes/a.txt")
        generic = RunnerWorkspace(managed)
        assert workspace.resource_key == generic.resource_key

        command = await managed.exec(ExecCommand.process("python3", "-c", "print('same')"))
        assert command.stdout.strip() == "same"

        await workspace.delete("notes/a.txt")
        assert not (tmp_path / "notes/a.txt").exists()
        bound = await binding.bind(
            workspace,
            managed,
            session_id=f"sess_{provider}",
        )
        assert bound.workspace is workspace
        assert bound.runner is managed
        await binding.finalize(bound, outcome="completed")
        await binding.finalize(bound, outcome="completed")
        return managed, workspace, inner

    managed, workspace, raw_runner = asyncio.run(run())

    assert workspace.runner is managed
    assert workspace.runner is not raw_runner
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(managed.exec(ExecCommand.process("true")))
    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_workspace_factory_failure_cleans_up_managed_egress() -> None:
    order: list[str] = []
    adapter = _WorkspaceAdapter("test", _MinimalRunner(), order)

    def fail(_runner: Runner) -> Any:
        raise RuntimeError("workspace setup failed")

    async def run() -> None:
        with pytest.raises(RuntimeError, match="workspace setup failed"):
            await _factory(adapter, fail).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_failure",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_workspace_factory_rejects_non_workspace_and_cleans_up() -> None:
    order: list[str] = []
    adapter = _WorkspaceAdapter("test", _MinimalRunner(), order)

    async def run() -> None:
        with pytest.raises(TypeError, match="must return a Workspace"):
            await _factory(adapter, lambda _runner: object()).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_wrong_type",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_native_workspace_factory_rejects_a_different_runner_and_cleans_up() -> None:
    order: list[str] = []
    managed_inner = _MinimalRunner()
    foreign_runner = _MinimalRunner()
    adapter = _WorkspaceAdapter("test", managed_inner, order)

    async def run() -> None:
        with pytest.raises(ValueError, match="must be bound to the managed runner"):
            await _factory(adapter, lambda _runner: RunnerWorkspace(foreign_runner)).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_wrong_runner",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert order == ["runner", "binding"]
    assert managed_inner.closed is True
    assert foreign_runner.closed is False
    assert adapter.teardown_calls == 1


def test_native_workspace_rejects_spoofed_runner_attribute_and_cleans_up() -> None:
    order: list[str] = []
    inner = _IdentifiedRunner()
    adapter = _WorkspaceAdapter("test", inner, order)

    async def run() -> None:
        with pytest.raises(TypeError, match="must implement RunnerBoundWorkspace"):
            await _factory(adapter, _PlainSpoofWorkspace).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_spoofed_runner",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert inner.closed is True
    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_native_workspace_rejects_mismatched_resource_identity_and_cleans_up() -> None:
    order: list[str] = []
    inner = _IdentifiedRunner()
    adapter = _WorkspaceAdapter("test", inner, order)

    async def run() -> None:
        with pytest.raises(ValueError, match="different runner resource"):
            await _factory(
                adapter,
                lambda runner: _MismatchedIdentityWorkspace(runner),
            ).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_wrong_resource",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert inner.closed is True
    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_native_workspace_rejects_missing_resource_identity_and_cleans_up() -> None:
    order: list[str] = []
    inner = _MinimalRunner()
    adapter = _WorkspaceAdapter("test", inner, order)

    async def run() -> None:
        with pytest.raises(ValueError, match="must expose stable resource identity"):
            await _factory(adapter, RunnerWorkspace).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_missing_identity",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert inner.closed is True
    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_workspace_factory_cancellation_cleans_up_managed_egress() -> None:
    order: list[str] = []
    started = asyncio.Event()
    adapter = _WorkspaceAdapter("test", _MinimalRunner(), order)

    async def wait_forever(_runner: Runner) -> Any:
        started.set()
        await asyncio.Event().wait()

    async def run() -> None:
        task = asyncio.create_task(
            _factory(adapter, wait_forever).create(
                EnvironmentFactoryRequest(
                    session_id="sess_workspace_cancel",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert order == ["runner", "binding"]
    assert adapter.teardown_calls == 1


def test_provider_workspaces_reject_runners_without_typed_capability(tmp_path: Path) -> None:
    runner = LocalRunner(tmp_path)

    with pytest.raises(TypeError, match="MicrosandboxWorkspaceCapability"):
        MicrosandboxWorkspace(runner, root=str(tmp_path))
    with pytest.raises(TypeError, match="E2BWorkspaceCapability"):
        E2BWorkspace(runner, root=str(tmp_path))

from __future__ import annotations

import asyncio
import io
import os
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from guard_harness import make_local_guard_exec
from tests.workspaces.conformance import (
    WorkspaceCapabilities,
    WorkspaceCapabilityClaim,
    WorkspaceConformanceRegistration,
    WorkspaceHarness,
    verify_bounded_reads_and_result_isolation,
    verify_listing_contract,
    verify_relative_path_safety,
    verify_resource_identity,
    verify_resource_identity_relationships,
    verify_round_trip,
    verify_symlink_safety,
)

import cayu.workspaces as workspaces_module
from cayu.runners import E2BRunner, LocalRunner, MicrosandboxRunner
from cayu.workspaces import (
    E2BWorkspace,
    LocalWorkspace,
    MicrosandboxWorkspace,
    RunnerWorkspace,
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
)


@dataclass(frozen=True)
class _E2BEntry:
    path: str
    type: str
    symlink_target: str | None = None


class _HostE2BFilesystem:
    async def get_info(self, path: str, **_kwargs: object) -> _E2BEntry:
        target = Path(path)
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            raise
        return _E2BEntry(
            path=path,
            type="dir" if stat.S_ISDIR(mode) else "file",
            symlink_target=os.readlink(target) if stat.S_ISLNK(mode) else None,
        )

    async def list(self, path: str, **_kwargs: object) -> list[_E2BEntry]:
        root = Path(path)
        entries: list[_E2BEntry] = []
        for target in root.rglob("*"):
            mode = target.lstat().st_mode
            entries.append(
                _E2BEntry(
                    path=str(target),
                    type="dir" if stat.S_ISDIR(mode) else "file",
                    symlink_target=os.readlink(target) if stat.S_ISLNK(mode) else None,
                )
            )
        return list(reversed(entries))


class _E2BSandbox:
    sandbox_id = "workspace-conformance-e2b"

    def __init__(self) -> None:
        self.files = _HostE2BFilesystem()


@dataclass(frozen=True)
class _MicrosandboxEntry:
    path: str
    kind: str


class _HostMicrosandboxFilesystem:
    async def list(self, path: str) -> list[_MicrosandboxEntry]:
        entries = [
            _MicrosandboxEntry(
                path=str(target),
                kind="dir" if target.is_dir() else "file",
            )
            for target in Path(path).iterdir()
        ]
        return list(reversed(entries))


class _MicrosandboxSftp:
    async def real_path(self, path: str) -> str:
        if not os.path.lexists(path):
            raise FileNotFoundError(path)
        return os.path.realpath(path)

    async def close(self) -> None:
        return None


class _MicrosandboxSshClient:
    async def sftp(self) -> _MicrosandboxSftp:
        return _MicrosandboxSftp()

    async def close(self) -> None:
        return None


class _MicrosandboxSsh:
    async def open_client(self, **_kwargs: object) -> _MicrosandboxSshClient:
        return _MicrosandboxSshClient()


class _MicrosandboxSandbox:
    def __init__(self) -> None:
        self.fs = _HostMicrosandboxFilesystem()

    def ssh(self) -> _MicrosandboxSsh:
        return _MicrosandboxSsh()


async def _local_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> WorkspaceHarness:
    return WorkspaceHarness(LocalWorkspace(root, workspace_id="conformance-local"), root)


async def _runner_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> WorkspaceHarness:
    runner = LocalRunner(root, inherit_env=False)
    return WorkspaceHarness(
        RunnerWorkspace(
            runner,
            workspace_id="conformance-runner",
            python_executable=sys.executable,
        ),
        root,
        runner.close,
    )


async def _e2b_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> WorkspaceHarness:
    runner = E2BRunner(_E2BSandbox(), e2b_module=SimpleNamespace())
    cast("Any", runner).exec = make_local_guard_exec()
    return WorkspaceHarness(
        E2BWorkspace(runner, root=str(root), workspace_id="conformance-e2b"),
        root,
        runner.close,
    )


async def _microsandbox_factory(root: Path, _monkeypatch: pytest.MonkeyPatch) -> WorkspaceHarness:
    runner = MicrosandboxRunner(
        _MicrosandboxSandbox(),
        name="workspace-conformance-microsandbox",
        sandbox_module=SimpleNamespace(),
    )
    cast("Any", runner).exec = make_local_guard_exec()
    return WorkspaceHarness(
        MicrosandboxWorkspace(
            runner,
            root=str(root),
            workspace_id="conformance-microsandbox",
        ),
        root,
        runner.close,
    )


async def _runner_bulk_transfer_probe(harness: WorkspaceHarness) -> None:
    workspace = cast("RunnerWorkspace", harness.workspace)
    await workspace.write_bytes("bulk/a.txt", b"alpha")
    archive = await workspace.read_tar_bytes(("bulk/a.txt",))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        assert [member.name for member in tar.getmembers()] == ["bulk/a.txt"]


NOT_ON_WORKSPACE = WorkspaceCapabilityClaim.not_applicable(
    "Bulk tar transfer is an extension of RunnerWorkspace, not the Workspace interface."
)

REGISTRATIONS = (
    WorkspaceConformanceRegistration(
        "local",
        LocalWorkspace,
        _local_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE),
    ),
    WorkspaceConformanceRegistration(
        "runner",
        RunnerWorkspace,
        _runner_factory,
        WorkspaceCapabilities("stable", WorkspaceCapabilityClaim.supported()),
        bulk_transfer_probe=_runner_bulk_transfer_probe,
    ),
    WorkspaceConformanceRegistration(
        "e2b",
        E2BWorkspace,
        _e2b_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE),
    ),
    WorkspaceConformanceRegistration(
        "microsandbox",
        MicrosandboxWorkspace,
        _microsandbox_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE),
    ),
)


def _run_scenario(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Any,
) -> None:
    async def run() -> None:
        root = tmp_path / registration.name
        root.mkdir()
        harness = await registration.factory(root, monkeypatch)
        try:
            await scenario(harness)
        finally:
            await harness.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_workspace_round_trip_conformance(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(harness: WorkspaceHarness) -> None:
        await verify_round_trip(harness.workspace)

    _run_scenario(registration, tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_workspace_path_and_symlink_conformance(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(harness: WorkspaceHarness) -> None:
        await verify_relative_path_safety(harness.workspace)
        await verify_symlink_safety(harness.workspace, harness.root)

    _run_scenario(registration, tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_workspace_bounds_and_listing_conformance(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(harness: WorkspaceHarness) -> None:
        await verify_bounded_reads_and_result_isolation(harness.workspace)
        await verify_listing_contract(harness.workspace)

    _run_scenario(registration, tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_workspace_resource_identity_and_capabilities(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(harness: WorkspaceHarness) -> None:
        verify_resource_identity(
            harness.workspace,
            registration.capabilities.resource_identity,
        )
        if registration.capabilities.bulk_transfer.state == "supported":
            assert registration.bulk_transfer_probe is not None
            await registration.bulk_transfer_probe(harness)

    _run_scenario(registration, tmp_path, monkeypatch, scenario)


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
def test_workspace_resource_identity_distinguishes_resources(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        first_root = tmp_path / "first"
        different_root = tmp_path / "different"
        first_root.mkdir()
        different_root.mkdir()
        first = await registration.factory(first_root, monkeypatch)
        same = await registration.factory(first_root, monkeypatch)
        different = await registration.factory(different_root, monkeypatch)
        try:
            verify_resource_identity_relationships(
                first.workspace,
                same.workspace,
                different.workspace,
                registration.capabilities.resource_identity,
            )
        finally:
            await different.aclose()
            await same.aclose()
            await first.aclose()

    asyncio.run(run())


def test_every_builtin_workspace_adapter_is_registered() -> None:
    exported_types = {
        value
        for name in workspaces_module.__all__
        if isinstance((value := getattr(workspaces_module, name)), type)
        and issubclass(value, Workspace)
        and value is not Workspace
    }
    assert {registration.workspace_type for registration in REGISTRATIONS} == exported_types


_SeededWorkspaceDefect = Literal[
    "traversal",
    "symlink-bypass",
    "overread",
    "glob-mismatch",
    "truncation",
    "resource-alias",
    "mutation-leakage",
]
_SeededWorkspaceScenario = Literal[
    "path",
    "symlink",
    "bounds",
    "listing",
    "identity-relationships",
]


class _SeededBrokenWorkspace(Workspace):
    """Plausible broken adapter variants proving the suite detects each bug class."""

    def __init__(self, root: Path, defect: _SeededWorkspaceDefect) -> None:
        self.id = f"broken-{defect}"
        self.root = root
        self.defect = defect
        self.delegate = LocalWorkspace(root)

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        if self.defect == "resource-alias":
            return ("broken", "shared-resource")
        return self.delegate.resource_key

    def bounded_read_limit(self, max_bytes: int) -> int:
        return self.delegate.bounded_read_limit(max_bytes)

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        if self.defect == "traversal" and path == "nested/../accepted.txt":
            return WorkspaceReadResult(b"", 0)
        if self.defect == "symlink-bypass" and path in {
            "leaf.txt",
            "parent-link/secret.txt",
        }:
            content = (self.root / path).read_bytes()
            return WorkspaceReadResult(content, len(content))
        if self.defect == "overread":
            return await self.delegate.read_bytes(path)
        return await self.delegate.read_bytes(path, max_bytes=max_bytes)

    async def write_bytes(self, path: str, content: bytes) -> None:
        if self.defect == "traversal" and path == "nested/../accepted.txt":
            return
        await self.delegate.write_bytes(path, content)

    async def delete(self, path: str) -> None:
        if self.defect == "traversal" and path == "nested/../accepted.txt":
            return
        await self.delegate.delete(path)

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        if self.defect == "glob-mismatch":
            return await self.delegate.list("**/*", limit=limit)
        if self.defect == "truncation":
            return await self.delegate.list(pattern)
        result = await self.delegate.list(pattern, limit=limit)
        if self.defect != "mutation-leakage":
            return result
        broken = object.__new__(WorkspaceListResult)
        object.__setattr__(broken, "paths", list(result.paths))
        object.__setattr__(broken, "total_count", result.total_count)
        object.__setattr__(broken, "truncated", result.truncated)
        return broken


@pytest.mark.parametrize(
    ("defect", "scenario"),
    (
        ("traversal", "path"),
        ("symlink-bypass", "symlink"),
        ("overread", "bounds"),
        ("glob-mismatch", "listing"),
        ("truncation", "listing"),
        ("resource-alias", "identity-relationships"),
        ("mutation-leakage", "bounds"),
    ),
)
def test_seeded_broken_workspace_is_rejected(
    defect: _SeededWorkspaceDefect,
    scenario: _SeededWorkspaceScenario,
    tmp_path: Path,
) -> None:
    root = tmp_path / defect
    root.mkdir()
    workspace = _SeededBrokenWorkspace(root, defect)

    async def run() -> None:
        if scenario == "path":
            await verify_relative_path_safety(workspace)
        elif scenario == "symlink":
            await verify_symlink_safety(workspace, root)
        elif scenario == "bounds":
            await verify_bounded_reads_and_result_isolation(workspace)
        elif scenario == "listing":
            await verify_listing_contract(workspace)
        elif scenario == "identity-relationships":
            same = _SeededBrokenWorkspace(root, defect)
            different_root = tmp_path / f"{defect}-different"
            different_root.mkdir()
            different = _SeededBrokenWorkspace(different_root, defect)
            verify_resource_identity_relationships(workspace, same, different, "stable")
        else:
            raise ValueError(f"Unknown seeded workspace scenario: {scenario}")

    with pytest.raises((AssertionError, pytest.fail.Exception)):
        asyncio.run(run())

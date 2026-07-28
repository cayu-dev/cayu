from __future__ import annotations

import asyncio
import inspect
import io
import os
import stat
import sys
import tarfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from guard_harness import instrument_directory_open_barrier, make_local_guard_exec
from tests.workspaces.conformance import (
    WorkspaceCapabilities,
    WorkspaceCapabilityClaim,
    WorkspaceConformanceRegistration,
    WorkspaceHarness,
    verify_bounded_reads_and_result_isolation,
    verify_listing_contract,
    verify_paging_and_conditional_mutations,
    verify_relative_path_safety,
    verify_resource_identity,
    verify_resource_identity_relationships,
    verify_round_trip,
    verify_symlink_safety,
)

import cayu.workspaces as workspaces_module
import cayu.workspaces.runner as runner_workspace_module
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


async def _runner_descriptor_containment_probe(
    harness: WorkspaceHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = cast("RunnerWorkspace", harness.workspace)
    pivot = harness.root / "pivot"
    pivot.mkdir()
    (pivot / "file.txt").write_bytes(b"inside")
    outside = harness.root.parent / "descriptor-containment-outside"
    outside.mkdir()
    outside_file = outside / "file.txt"
    outside_file.write_bytes(b"outside")
    # Exercise the exact capability boundary: relocate the already-authorized
    # inode outside the root and replace its old name with a symlink to a
    # distinct external target. The write may continue through the descriptor,
    # but must never follow the replacement symlink.
    held = harness.root.parent / "descriptor-containment-relocated"
    ready = harness.root.parent / "descriptor-containment-ready"
    release = harness.root.parent / "descriptor-containment-release"
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_directory_open_barrier(
            runner_workspace_module._RUNNER_WORKSPACE_PROGRAM,
            ready=ready,
            release=release,
        ),
    )

    operation = asyncio.create_task(workspace.write_bytes("pivot/file.txt", b"changed"))
    try:
        for _ in range(1000):
            if ready.exists():
                break
            if operation.done():
                await operation
                raise AssertionError("Descriptor containment probe completed before its barrier.")
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("Descriptor containment probe did not reach its barrier.")
        pivot.rename(held)
        pivot.symlink_to(outside, target_is_directory=True)
    finally:
        release.write_text("release")
    try:
        await operation
        assert outside_file.read_bytes() == b"outside"
        assert (held / "file.txt").read_bytes() == b"changed"
    finally:
        if pivot.is_symlink():
            pivot.unlink()
        if held.exists():
            held.rename(pivot)

    assert (pivot / "file.txt").read_bytes() == b"changed"


NOT_ON_WORKSPACE = WorkspaceCapabilityClaim.not_applicable(
    "Bulk tar transfer is an extension of RunnerWorkspace, not the Workspace interface."
)
TRUSTED_LOCAL_PATHS = WorkspaceCapabilityClaim.not_applicable(
    "LocalWorkspace does not claim containment against a hostile co-resident host process."
)
NATIVE_LISTING_IS_ADVISORY = WorkspaceCapabilityClaim.not_applicable(
    "The native adapter's listing transport is advisory rather than descriptor-pinned."
)

REGISTRATIONS = (
    WorkspaceConformanceRegistration(
        "local",
        LocalWorkspace,
        _local_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE, TRUSTED_LOCAL_PATHS),
    ),
    WorkspaceConformanceRegistration(
        "runner",
        RunnerWorkspace,
        _runner_factory,
        WorkspaceCapabilities(
            "stable",
            WorkspaceCapabilityClaim.supported(),
            WorkspaceCapabilityClaim.supported(),
        ),
        bulk_transfer_probe=_runner_bulk_transfer_probe,
        descriptor_containment_probe=_runner_descriptor_containment_probe,
    ),
    WorkspaceConformanceRegistration(
        "e2b",
        E2BWorkspace,
        _e2b_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE, NATIVE_LISTING_IS_ADVISORY),
    ),
    WorkspaceConformanceRegistration(
        "microsandbox",
        MicrosandboxWorkspace,
        _microsandbox_factory,
        WorkspaceCapabilities("stable", NOT_ON_WORKSPACE, NATIVE_LISTING_IS_ADVISORY),
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
def test_workspace_paging_and_conditional_mutation_conformance(
    registration: WorkspaceConformanceRegistration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(harness: WorkspaceHarness) -> None:
        await verify_paging_and_conditional_mutations(harness.workspace)

    _run_scenario(registration, tmp_path, monkeypatch, scenario)


def test_local_and_local_runner_workspace_share_conditional_mutation_lock(
    tmp_path: Path,
) -> None:
    local = LocalWorkspace(tmp_path, workspace_id="local")
    runner = LocalRunner(tmp_path, inherit_env=False)
    runner_workspace = RunnerWorkspace(
        runner,
        workspace_id="runner",
        python_executable=sys.executable,
    )

    async def scenario() -> None:
        await local.write_bytes("shared.txt", b"base")
        revision = (await local.read_bytes("shared.txt")).revision
        assert revision is not None

        async def replace(workspace: Workspace, content: bytes) -> str:
            try:
                await workspace.replace_bytes(
                    "shared.txt",
                    content,
                    expected_revision=revision,
                )
            except workspaces_module.WorkspaceRevisionMismatchError:
                return "stale"
            return "replaced"

        outcomes = await asyncio.gather(
            replace(local, b"local"),
            replace(runner_workspace, b"runner"),
        )
        assert sorted(outcomes) == ["replaced", "stale"]

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runner.close())


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file modes")
def test_local_workspace_create_uses_conventional_umask_permissions(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    previous_umask = os.umask(0o022)
    try:
        asyncio.run(workspace.create_bytes("created.txt", b"content"))
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((tmp_path / "created.txt").stat().st_mode) == 0o644


def test_conditional_mutations_serialize_filesystem_case_aliases(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path)
    asyncio.run(workspace.write_bytes("Case.txt", b"base"))
    lower_alias = tmp_path / "case.txt"
    if not lower_alias.exists() or not os.path.samefile(tmp_path / "Case.txt", lower_alias):
        pytest.skip("filesystem is case-sensitive")
    revision = asyncio.run(workspace.read_bytes("Case.txt")).revision
    assert revision is not None

    async def replace(path: str, content: bytes) -> str:
        try:
            await workspace.replace_bytes(path, content, expected_revision=revision)
        except workspaces_module.WorkspaceRevisionMismatchError:
            return "stale"
        return "replaced"

    async def scenario() -> list[str]:
        return await asyncio.gather(
            replace("Case.txt", b"first"),
            replace("case.txt", b"second"),
        )

    outcomes = asyncio.run(scenario())
    assert sorted(outcomes) == ["replaced", "stale"]


def test_stale_local_conditional_mutation_streams_current_identity(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    with path.open("wb") as file:
        file.truncate(32 * 1024 * 1024)
    workspace = LocalWorkspace(tmp_path)

    tracemalloc.start()
    with pytest.raises(workspaces_module.WorkspaceRevisionMismatchError):
        asyncio.run(
            workspace.replace_bytes(
                "large.bin",
                b"replacement",
                expected_revision="sha256:stale",
            )
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 5 * 1024 * 1024


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
        if registration.capabilities.descriptor_relative_containment.state == "supported":
            assert registration.descriptor_containment_probe is not None
            await registration.descriptor_containment_probe(harness, monkeypatch)

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
        and not inspect.isabstract(value)
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
        offset: int = 0,
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
        return await self.delegate.read_bytes(path, offset=offset, max_bytes=max_bytes)

    async def write_bytes(self, path: str, content: bytes) -> None:
        if self.defect == "traversal" and path == "nested/../accepted.txt":
            return
        await self.delegate.write_bytes(path, content)

    async def delete(self, path: str) -> None:
        if self.defect == "traversal" and path == "nested/../accepted.txt":
            return
        await self.delegate.delete(path)

    async def create_bytes(self, path: str, content: bytes):
        return await self.delegate.create_bytes(path, content)

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        return await self.delegate.replace_bytes(
            path,
            content,
            expected_revision=expected_revision,
        )

    async def delete_if_revision(self, path: str, *, expected_revision: str):
        return await self.delegate.delete_if_revision(
            path,
            expected_revision=expected_revision,
        )

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

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from examples._workspace_conformance import (
    verify_portable_workspace_path_safety,
    verify_portable_workspace_round_trip,
)

from cayu.workspaces import Workspace

CapabilityState = Literal["supported", "not_applicable"]
ResourceIdentityState = Literal["stable", "indeterminate"]


@dataclass(frozen=True)
class WorkspaceCapabilityClaim:
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state == "supported" and self.reason is not None:
            raise ValueError("Supported workspace capabilities cannot define a reason.")
        if self.state == "not_applicable" and not (self.reason and self.reason.strip()):
            raise ValueError("Not-applicable workspace capabilities require a reason.")

    @classmethod
    def supported(cls) -> WorkspaceCapabilityClaim:
        return cls("supported")

    @classmethod
    def not_applicable(cls, reason: str) -> WorkspaceCapabilityClaim:
        return cls("not_applicable", reason)


@dataclass(frozen=True)
class WorkspaceCapabilities:
    resource_identity: ResourceIdentityState
    bulk_transfer: WorkspaceCapabilityClaim


@dataclass
class WorkspaceHarness:
    workspace: Workspace
    root: Path
    finalize: Callable[[], Awaitable[None]] | None = None

    async def aclose(self) -> None:
        if self.finalize is not None:
            await self.finalize()


WorkspaceFactory = Callable[[Path, pytest.MonkeyPatch], Awaitable[WorkspaceHarness]]
BulkTransferProbe = Callable[[WorkspaceHarness], Awaitable[None]]


@dataclass(frozen=True)
class WorkspaceConformanceRegistration:
    name: str
    workspace_type: type[Workspace]
    factory: WorkspaceFactory
    capabilities: WorkspaceCapabilities
    bulk_transfer_probe: BulkTransferProbe | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Workspace conformance registration name must be nonblank.")
        if not issubclass(self.workspace_type, Workspace):
            raise TypeError("Workspace conformance registration type must implement Workspace.")
        if (
            self.capabilities.bulk_transfer.state == "supported"
            and self.bulk_transfer_probe is None
        ):
            raise ValueError(
                "Registrations claiming bulk-transfer support require a scenario probe."
            )


async def verify_round_trip(workspace: Workspace) -> None:
    await verify_portable_workspace_round_trip(workspace, adapter=type(workspace).__name__)


async def verify_relative_path_safety(workspace: Workspace) -> None:
    await verify_portable_workspace_path_safety(workspace, adapter=type(workspace).__name__)


async def verify_symlink_safety(workspace: Workspace, root: Path) -> None:
    outside = root.parent / f"{root.name}-outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_bytes(b"secret")
    (root / "leaf.txt").symlink_to(outside_file)
    (root / "parent-link").symlink_to(outside, target_is_directory=True)

    for path in ("leaf.txt", "parent-link/secret.txt"):
        with pytest.raises(ValueError):
            await workspace.read_bytes(path)
        with pytest.raises(ValueError):
            await workspace.write_bytes(path, b"overwrite")
        with pytest.raises(ValueError):
            await workspace.delete(path)

    try:
        listed = await workspace.list("**/*")
    except ValueError:
        pass
    else:
        assert "leaf.txt" not in listed.paths
        assert "parent-link/secret.txt" not in listed.paths
        assert all("secret" not in path for path in listed.paths)
    assert outside_file.read_bytes() == b"secret"


async def verify_bounded_reads_and_result_isolation(workspace: Workspace) -> None:
    await workspace.write_bytes("bounded.bin", b"abcdef")
    result = await workspace.read_bytes("bounded.bin", max_bytes=3)
    assert result.content == b"abc"
    assert type(result.content) is bytes
    assert result.total_bytes == 6
    assert result.truncated is True

    listing = await workspace.list("**/*")
    assert type(listing.paths) is tuple
    snapshot = listing.paths
    await workspace.write_bytes("later.txt", b"later")
    assert listing.paths is snapshot
    assert "later.txt" not in listing.paths
    await workspace.delete("bounded.bin")
    await workspace.delete("later.txt")


async def verify_listing_contract(workspace: Workspace) -> None:
    for path in ("c.txt", "a.txt", "b.txt", "nested/d.txt", "nested/a.md"):
        await workspace.write_bytes(path, path.encode())

    top_level = await workspace.list("*.txt")
    assert top_level.paths == ("a.txt", "b.txt", "c.txt")
    assert top_level.total_count == 3
    assert top_level.truncated is False

    recursive = await workspace.list("**/*.txt")
    assert recursive.paths == ("a.txt", "b.txt", "c.txt", "nested/d.txt")
    assert recursive.total_count == 4
    assert recursive.truncated is False

    limited = await workspace.list("**/*.txt", limit=2)
    assert limited.paths == ("a.txt", "b.txt")
    assert limited.truncated is True
    assert limited.total_count is None or limited.total_count == 4


def verify_resource_identity(workspace: Workspace, state: ResourceIdentityState) -> None:
    first = workspace.resource_key
    second = workspace.resource_key
    if state == "indeterminate":
        assert first is None
        assert second is None
        return
    assert first is not None
    assert first == second
    hash(first)


def verify_resource_identity_relationships(
    first: Workspace,
    same_resource: Workspace,
    different_resource: Workspace,
    state: ResourceIdentityState,
) -> None:
    if state == "indeterminate":
        assert first.resource_key is None
        assert same_resource.resource_key is None
        assert different_resource.resource_key is None
        return
    assert first.resource_key is not None
    assert first.resource_key == same_resource.resource_key
    assert first.resource_key != different_resource.resource_key

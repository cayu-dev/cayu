from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from examples._workspace_conformance import (
    verify_portable_workspace_path_safety,
    verify_portable_workspace_round_trip,
)

from cayu.workspaces import Workspace, WorkspaceRevisionMismatchError

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
    descriptor_relative_containment: WorkspaceCapabilityClaim


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
DescriptorContainmentProbe = Callable[[WorkspaceHarness, pytest.MonkeyPatch], Awaitable[None]]


@dataclass(frozen=True)
class WorkspaceConformanceRegistration:
    name: str
    workspace_type: type[Workspace]
    factory: WorkspaceFactory
    capabilities: WorkspaceCapabilities
    bulk_transfer_probe: BulkTransferProbe | None = None
    descriptor_containment_probe: DescriptorContainmentProbe | None = None

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
        if (
            self.capabilities.descriptor_relative_containment.state == "supported"
            and self.descriptor_containment_probe is None
        ):
            raise ValueError(
                "Registrations claiming descriptor-relative containment require a scenario probe."
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


async def verify_paging_and_conditional_mutations(workspace: Workspace) -> None:
    original = b"abcdef"
    await workspace.write_bytes("conditional.txt", original)

    first = await workspace.read_bytes("conditional.txt", offset=0, max_bytes=2)
    assert first.content == b"ab"
    assert first.offset == 0
    assert first.total_bytes == len(original)
    assert first.truncated is True
    assert first.next_offset == 2
    assert first.revision is None
    assert first.sha256 is None

    suffix = await workspace.read_bytes("conditional.txt", offset=2, max_bytes=10)
    assert suffix.content == b"cdef"
    assert suffix.offset == 2
    assert suffix.total_bytes == len(original)
    assert suffix.truncated is False
    assert suffix.next_offset is None
    assert suffix.revision is None
    assert suffix.sha256 is None

    full = await workspace.read_bytes("conditional.txt", max_bytes=10)
    digest = hashlib.sha256(original).hexdigest()
    assert full.sha256 == digest
    assert full.revision == f"sha256:{digest}"

    await workspace.write_bytes("conditional.txt", b"newer")
    with pytest.raises(WorkspaceRevisionMismatchError):
        await workspace.replace_bytes(
            "conditional.txt",
            b"stale-write",
            expected_revision=full.revision,
        )
    assert (await workspace.read_bytes("conditional.txt")).content == b"newer"

    current = await workspace.read_bytes("conditional.txt")
    replaced = await workspace.replace_bytes(
        "conditional.txt",
        b"replacement",
        expected_revision=current.revision,
    )
    assert replaced.operation == "replace"
    assert replaced.before_revision == current.revision
    assert replaced.after_revision is not None
    assert replaced.before_bytes == len(b"newer")
    assert replaced.after_bytes == len(b"replacement")

    with pytest.raises(FileExistsError):
        await workspace.create_bytes("conditional.txt", b"must-not-overwrite")
    assert (await workspace.read_bytes("conditional.txt")).content == b"replacement"

    created = await workspace.create_bytes("created.txt", b"created")
    assert created.operation == "create"
    assert created.before_revision is None
    assert created.after_revision is not None

    stale_delete = await workspace.read_bytes("created.txt")
    await workspace.write_bytes("created.txt", b"changed")
    with pytest.raises(WorkspaceRevisionMismatchError):
        await workspace.delete_if_revision(
            "created.txt",
            expected_revision=stale_delete.revision,
        )
    current_delete = await workspace.read_bytes("created.txt")
    deleted = await workspace.delete_if_revision(
        "created.txt",
        expected_revision=current_delete.revision,
    )
    assert deleted.operation == "delete"
    assert deleted.before_revision == current_delete.revision
    assert deleted.after_revision is None
    with pytest.raises(FileNotFoundError):
        await workspace.read_bytes("created.txt")

    await workspace.write_bytes("race.txt", b"base")
    race_revision = (await workspace.read_bytes("race.txt")).revision

    async def contender(content: bytes) -> str:
        try:
            await workspace.replace_bytes(
                "race.txt",
                content,
                expected_revision=race_revision,
            )
        except WorkspaceRevisionMismatchError:
            return "stale"
        return "replaced"

    outcomes = await asyncio.gather(contender(b"first"), contender(b"second"))
    assert sorted(outcomes) == ["replaced", "stale"]
    assert (await workspace.read_bytes("race.txt")).content in {b"first", b"second"}


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

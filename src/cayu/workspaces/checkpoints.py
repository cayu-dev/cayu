"""Bounded, content-addressed workspace revisions retained by ArtifactStore pins."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts import ArtifactMetadata, ArtifactReadResult, ArtifactScope, ArtifactStore
from cayu.workspaces.base import (
    TarWriter,
    Workspace,
    WorkspaceDirectoryPruner,
    WorkspaceGitEntryListResult,
    WorkspaceGitModeMutator,
    WorkspaceReadResult,
    _validate_workspace_relative_path,
)
from cayu.workspaces.revisions import (
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
)


class WorkspaceCheckpointError(RuntimeError):
    """No subsequent execution is safe until this checkpoint failure is resolved."""


class WorkspaceCheckpointPolicy(BaseModel):
    """Enable durable revisions for an exclusively owned workspace.

    Limits cover the complete workspace projection exposed by its adapter.
    Symlinks and special files fail closed; excluded paths are not checkpointed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    timeout_seconds: StrictInt = Field(default=120, ge=1, le=3600)
    max_paths: StrictInt = Field(default=10000, ge=1, le=100000)
    max_file_bytes: StrictInt = Field(default=64 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)
    max_total_bytes: StrictInt = Field(default=256 * 1024 * 1024, ge=1, le=4 * 1024**3)
    max_manifest_bytes: StrictInt = Field(default=4 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024)


class WorkspaceCheckpointFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(ge=0)
    git_mode: Literal["100644", "100755"]
    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{32}$")

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _validate_workspace_relative_path(value)


class WorkspaceCheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[WorkspaceCheckpointFile, ...]

    def validate_limits(self, policy: WorkspaceCheckpointPolicy) -> None:
        paths = tuple(entry.path for entry in self.files)
        if paths != tuple(sorted(set(paths))) or len(paths) > policy.max_paths:
            raise WorkspaceCheckpointError("Invalid workspace checkpoint path set.")
        if any(entry.size_bytes > policy.max_file_bytes for entry in self.files):
            raise WorkspaceCheckpointError("Workspace checkpoint file exceeds policy.")
        if sum(entry.size_bytes for entry in self.files) > policy.max_total_bytes:
            raise WorkspaceCheckpointError("Workspace checkpoint exceeds aggregate byte policy.")
        if revision_digest(self.files) != self.revision:
            raise WorkspaceCheckpointError("Workspace checkpoint revision identity mismatch.")


def revision_digest(
    files: tuple[WorkspaceCheckpointFile, ...] | list[WorkspaceCheckpointFile],
) -> str:
    return hashlib.sha256(
        canonical_durable_json_bytes(
            [entry.model_dump(exclude={"artifact_id"}) for entry in files], "workspace_revision"
        )
    ).hexdigest()


def require_exclusive(isolation: WorkspaceWriterIsolationEvidence) -> None:
    if isolation.status is not WorkspaceWriterIsolationStatus.EXCLUSIVE:
        raise WorkspaceCheckpointError(
            "Workspace checkpoint requires proven exclusive writer isolation."
        )


def require_checkpoint_store(store: ArtifactStore | None) -> ArtifactStore:
    if store is None or store.supports_pins is not True:
        raise WorkspaceCheckpointError(
            "Workspace checkpoint requires an ArtifactStore with durable pins."
        )
    return store


def _artifact_id(content: bytes, environment_name: str, kind: str) -> str:
    return (
        "art_"
        + hashlib.sha256(
            environment_name.encode("utf-8") + b"\0" + kind.encode("ascii") + b"\0" + content
        ).hexdigest()[:32]
    )


async def _put_pinned(
    store: ArtifactStore, content: bytes, *, environment_name: str, kind: str, owner: str
) -> str:
    artifact_id = _artifact_id(content, environment_name, kind)
    result = await store.put_bytes(
        content,
        artifact_id=artifact_id,
        filename=f"{kind}.bin",
        content_type="application/octet-stream",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name=environment_name,
        metadata={"cayu_workspace_checkpoint": 1, "sha256": hashlib.sha256(content).hexdigest()},
    )
    if (
        type(result) is not ArtifactMetadata
        or result.id != artifact_id
        or result.size_bytes != len(content)
        or result.scope is not ArtifactScope.ENVIRONMENT
        or result.environment_name != environment_name
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint artifact acknowledgement mismatch.")
    # A concurrent collection before pin is permitted to win. In that case pin
    # fails and no receipt is published. After pin, delete must fail atomically.
    await store.pin(artifact_id, owner=owner)
    read = await store.read_bytes(artifact_id, max_bytes=max(1, len(content)))
    if (
        type(read) is not ArtifactReadResult
        or read.truncated
        or read.total_bytes != len(content)
        or read.content != content
        or read.metadata.id != artifact_id
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint artifact readback mismatch.")
    return artifact_id


async def capture_workspace_checkpoint(
    workspace: Workspace,
    store: ArtifactStore,
    *,
    policy: WorkspaceCheckpointPolicy,
    environment_name: str,
    owner: str,
    isolation: Callable[[], WorkspaceWriterIsolationEvidence],
) -> tuple[str, WorkspaceCheckpointManifest]:
    """Capture, pin, and independently verify a complete regular-file revision."""
    require_checkpoint_store(store)
    if not isinstance(workspace, WorkspaceDirectoryPruner):
        raise WorkspaceCheckpointError(
            "Workspace checkpoints require directory restoration support."
        )
    require_durable_clean_nonblank(owner, "checkpoint.owner")
    before = isolation()
    require_exclusive(before)
    files = await _capture_files(workspace, store, policy, environment_name, owner)
    manifest = WorkspaceCheckpointManifest(revision=revision_digest(files), files=tuple(files))
    manifest.validate_limits(policy)
    encoded = canonical_durable_json_bytes(manifest.model_dump(mode="json"), "workspace_manifest")
    if len(encoded) > policy.max_manifest_bytes:
        raise WorkspaceCheckpointError("Workspace checkpoint manifest exceeds policy.")
    # Re-read the projection after publication. Never bless a mixed-time snapshot.
    if await workspace_checkpoint_revision(workspace, policy=policy) != manifest.revision:
        raise WorkspaceCheckpointError("Workspace changed during checkpoint capture.")
    if isolation() != before:
        raise WorkspaceCheckpointError("Workspace isolation changed during checkpoint capture.")
    artifact_id = await _put_pinned(
        store, encoded, environment_name=environment_name, kind="workspace-manifest", owner=owner
    )
    return artifact_id, manifest


async def _capture_files(
    workspace: Workspace,
    store: ArtifactStore | None,
    policy: WorkspaceCheckpointPolicy,
    environment_name: str = "",
    owner: str = "",
) -> list[WorkspaceCheckpointFile]:
    listed = await workspace.list_git_entries(limit=policy.max_paths + 1)
    if (
        type(listed) is not WorkspaceGitEntryListResult
        or listed.truncated
        or listed.total_count > policy.max_paths
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint exceeds path policy.")
    files: list[WorkspaceCheckpointFile] = []
    total = 0
    for entry in listed.entries:
        if entry.git_mode == "120000":
            raise WorkspaceCheckpointError("Workspace checkpoints cannot represent symbolic links.")
        read = await workspace.read_bytes(
            entry.path, max_bytes=min(policy.max_file_bytes, max(1, policy.max_total_bytes - total))
        )
        if type(read) is not WorkspaceReadResult:
            raise WorkspaceCheckpointError("Invalid workspace checkpoint read result.")
        total += read.total_bytes
        if (
            read.truncated
            or read.redaction_truncated
            or read.offset != 0
            or len(read.content) != read.total_bytes
            or total > policy.max_total_bytes
            or read.git_mode != entry.git_mode
        ):
            raise WorkspaceCheckpointError(
                "Workspace checkpoint requires a complete bounded file snapshot."
            )
        artifact_id = "art_" + "0" * 32
        if store is not None:
            artifact_id = await _put_pinned(
                store,
                read.content,
                environment_name=environment_name,
                kind="workspace-file",
                owner=owner,
            )
        files.append(
            WorkspaceCheckpointFile(
                path=entry.path,
                sha256=hashlib.sha256(read.content).hexdigest(),
                size_bytes=read.total_bytes,
                git_mode=entry.git_mode,
                artifact_id=artifact_id,
            )
        )
    return files


async def workspace_checkpoint_revision(
    workspace: Workspace,
    *,
    policy: WorkspaceCheckpointPolicy,
) -> str:
    return revision_digest(await _capture_files(workspace, None, policy))


async def load_workspace_checkpoint(
    store: ArtifactStore,
    artifact_id: str,
    *,
    policy: WorkspaceCheckpointPolicy,
    expected_manifest_sha256: str,
) -> WorkspaceCheckpointManifest:
    read = await store.read_bytes(artifact_id, max_bytes=policy.max_manifest_bytes)
    if (
        type(read) is not ArtifactReadResult
        or read.truncated
        or read.total_bytes != len(read.content)
        or len(read.content) > policy.max_manifest_bytes
        or read.metadata.id != artifact_id
        or hashlib.sha256(read.content).hexdigest() != expected_manifest_sha256
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint manifest digest mismatch.")
    manifest = WorkspaceCheckpointManifest.model_validate(json.loads(read.content))
    manifest.validate_limits(policy)
    return manifest


async def restore_workspace_checkpoint(
    workspace: Workspace,
    store: ArtifactStore,
    manifest: WorkspaceCheckpointManifest,
    *,
    policy: WorkspaceCheckpointPolicy,
    isolation: Callable[[], WorkspaceWriterIsolationEvidence],
) -> None:
    """Reconstruct an owned projection without rerunning any tool effects.

    Partial restoration is idempotent. The caller must retain its recovery fence
    and cannot expose the environment until verification succeeds.
    """
    manifest.validate_limits(policy)
    before = isolation()
    require_exclusive(before)
    if not isinstance(workspace, WorkspaceDirectoryPruner):
        raise WorkspaceCheckpointError("Workspace restore requires directory restoration support.")
    if not isinstance(workspace, WorkspaceGitModeMutator | TarWriter):
        raise WorkspaceCheckpointError("Workspace restore requires exact file mode mutations.")
    listed = await workspace.list_git_entries(limit=policy.max_paths + 1)
    if (
        type(listed) is not WorkspaceGitEntryListResult
        or listed.truncated
        or listed.total_count > policy.max_paths
    ):
        raise WorkspaceCheckpointError("Workspace restore destination exceeds policy.")
    if any(entry.git_mode == "120000" for entry in listed.entries):
        raise WorkspaceCheckpointError("Workspace restore refuses destination symbolic links.")
    desired = {entry.path for entry in manifest.files}
    for entry in listed.entries:
        if entry.path not in desired:
            old = await workspace.read_bytes(entry.path, max_bytes=policy.max_file_bytes)
            if old.truncated or old.revision is None:
                raise WorkspaceCheckpointError("Workspace restore cannot authenticate deletion.")
            await workspace.delete_if_revision(entry.path, expected_revision=old.revision)
    for entry in manifest.files:
        read = await store.read_bytes(entry.artifact_id, max_bytes=max(1, entry.size_bytes))
        if (
            type(read) is not ArtifactReadResult
            or read.metadata.id != entry.artifact_id
            or read.truncated
            or read.total_bytes != entry.size_bytes
            or hashlib.sha256(read.content).hexdigest() != entry.sha256
        ):
            raise WorkspaceCheckpointError("Workspace checkpoint file digest mismatch.")
        await workspace.prune_empty_directories(entry.path, max_directories=policy.max_paths)
        if not isinstance(workspace, WorkspaceGitModeMutator):
            # RunnerWorkspace's guarded tar importer preserves exact regular-file
            # modes. Names and contents come only from the authenticated manifest;
            # writer isolation owns the whole restore until final verification.
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                member = tarfile.TarInfo(entry.path)
                member.uname = "cayu.git-mode.v1"
                member.size = entry.size_bytes
                member.mode = 0o755 if entry.git_mode == "100755" else 0o644
                tar.addfile(member, io.BytesIO(read.content))
            await workspace.write_tar_bytes(archive.getvalue())
            continue
        try:
            old = await workspace.read_bytes(entry.path, max_bytes=policy.max_file_bytes)
        except FileNotFoundError:
            await workspace.create_bytes_with_git_mode(
                entry.path, read.content, git_mode=entry.git_mode
            )
        else:
            if old.truncated or old.revision is None or old.git_mode is None:
                raise WorkspaceCheckpointError("Workspace restore cannot authenticate replacement.")
            await workspace.replace_bytes_with_git_mode(
                entry.path,
                read.content,
                expected_revision=old.revision,
                expected_git_mode=old.git_mode,
                git_mode=entry.git_mode,
            )
    if await workspace_checkpoint_revision(workspace, policy=policy) != manifest.revision:
        raise WorkspaceCheckpointError("Restored workspace revision mismatch.")
    if isolation() != before:
        raise WorkspaceCheckpointError("Workspace isolation changed during restoration.")


async def pin_workspace_checkpoint(
    store: ArtifactStore,
    artifact_id: str,
    *,
    owner: str,
    policy: WorkspaceCheckpointPolicy,
    expected_manifest_sha256: str,
) -> None:
    """Retain a revision for a session, task, snapshot, or descendant owner."""
    require_checkpoint_store(store)
    await store.pin(artifact_id, owner=owner)
    manifest = await load_workspace_checkpoint(
        store, artifact_id, policy=policy, expected_manifest_sha256=expected_manifest_sha256
    )
    for entry in manifest.files:
        await store.pin(entry.artifact_id, owner=owner)


async def release_workspace_checkpoint(
    store: ArtifactStore,
    artifact_id: str,
    *,
    owner: str,
    policy: WorkspaceCheckpointPolicy,
    expected_manifest_sha256: str,
) -> None:
    """Release a retired owner's revision. Other owners continue to block GC."""
    manifest = await load_workspace_checkpoint(
        store, artifact_id, policy=policy, expected_manifest_sha256=expected_manifest_sha256
    )
    for entry in manifest.files:
        await store.release_pin(entry.artifact_id, owner=owner)
    await store.release_pin(artifact_id, owner=owner)

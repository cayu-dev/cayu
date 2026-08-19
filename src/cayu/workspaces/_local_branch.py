from __future__ import annotations

import asyncio
import builtins
import errno
import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from cayu.workspaces._local_guard import (
    _LocalGuardStagingCleanupError,
    create_regular,
    delete_empty_directory,
    delete_regular,
    delete_regular_if_revision,
    open_regular_for_read,
    replace_regular_if_revision,
    restore_regular,
)
from cayu.workspaces._mutations import (
    fence_local_workspace_source,
    local_workspace_source_is_fenced,
    mutation_result,
    workspace_path_lock,
    workspace_source_lock,
)
from cayu.workspaces.base import (
    WorkspaceListResult,
    WorkspaceMutationResult,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
    WorkspaceRevisionMismatchError,
    _validate_workspace_relative_path,
    _WorkspaceListCollector,
    matches_list_pattern,
    validate_list_pattern,
)
from cayu.workspaces.branches import (
    WorkspaceBranch,
    WorkspaceBranchChange,
    WorkspaceBranchChangeSet,
    WorkspaceBranchClosedError,
    WorkspaceBranchConflict,
    WorkspaceBranchContentIdentity,
    WorkspaceBranchCreationResult,
    WorkspaceBranchEvidence,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchLimits,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationError,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchPublicationResult,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceBranchRollbackResult,
    _bounded_workspace_branch_evidence,
    _copy_workspace_branch_request_envelope,
    _json_text_size,
    _workspace_branch_empty_change_set_json_size,
    _workspace_branch_evidence_json_size,
    copy_workspace_branch_request,
    workspace_branch_change_set_digest,
    workspace_branch_evidence,
)
from cayu.workspaces.revisions import (
    WorkspaceIdentity,
    _deterministic_workspace_manifest_bytes,
    _deterministic_workspace_manifest_revision,
)

if TYPE_CHECKING:
    from cayu.workspaces.local import LocalWorkspace


_ResultT = TypeVar("_ResultT")
_CleanupKeyT = TypeVar("_CleanupKeyT", bound=Hashable)
_CleanupPayloadT = TypeVar("_CleanupPayloadT")
_PathKind = Literal["missing", "file", "directory", "symlink", "special"]
_SOURCE_MANAGER_LOCK = threading.Lock()
_ACTIVE_BRANCHES: dict[tuple[object, ...], int] = {}


@dataclass(slots=True)
class _RetainedCleanup(Generic[_CleanupPayloadT]):
    source_key: tuple[object, ...]
    payload: _CleanupPayloadT
    claimed: bool = False


class _RetainedCleanupRegistry(Generic[_CleanupKeyT, _CleanupPayloadT]):
    """Own retained cleanup identity, claims, and source-scoped assistance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[_CleanupKeyT, _RetainedCleanup[_CleanupPayloadT]] = {}

    def retain(
        self,
        key: _CleanupKeyT,
        *,
        source_key: tuple[object, ...],
        payload: _CleanupPayloadT,
    ) -> None:
        with self._lock:
            retained = self._records.get(key)
            if retained is None:
                self._records[key] = _RetainedCleanup(
                    source_key=source_key,
                    payload=payload,
                )
                return
            if retained.source_key != source_key or retained.payload != payload:
                raise RuntimeError("Retained cleanup identity changed while owned.")

    def claim(
        self,
        key: _CleanupKeyT,
        *,
        eligible: Callable[[_CleanupPayloadT], bool] | None = None,
    ) -> bool:
        with self._lock:
            retained = self._records.get(key)
            if (
                retained is None
                or retained.claimed
                or (eligible is not None and not eligible(retained.payload))
            ):
                return False
            retained.claimed = True
            return True

    def payload(self, key: _CleanupKeyT) -> _CleanupPayloadT | None:
        with self._lock:
            retained = self._records.get(key)
            return None if retained is None else retained.payload

    def release_claim(self, key: _CleanupKeyT) -> None:
        with self._lock:
            retained = self._records.get(key)
            if retained is not None:
                retained.claimed = False

    def forget(self, key: _CleanupKeyT) -> None:
        with self._lock:
            self._records.pop(key, None)

    def claim_pending(
        self,
        source_key: tuple[object, ...],
        *,
        eligible: Callable[[_CleanupPayloadT], bool] | None = None,
    ) -> tuple[tuple[_CleanupKeyT, _CleanupPayloadT], ...]:
        with self._lock:
            pending: list[tuple[_CleanupKeyT, _CleanupPayloadT]] = []
            for key, retained in self._records.items():
                if (
                    retained.source_key != source_key
                    or retained.claimed
                    or (eligible is not None and not eligible(retained.payload))
                ):
                    continue
                retained.claimed = True
                pending.append((key, retained.payload))
            return tuple(pending)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._records))

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._records

    def items(
        self,
    ) -> tuple[tuple[_CleanupKeyT, _RetainedCleanup[_CleanupPayloadT]], ...]:
        with self._lock:
            return tuple(self._records.items())


_SOURCE_STAGING_CLEANUPS: _RetainedCleanupRegistry[
    _LocalGuardStagingCleanupError,
    _LocalGuardStagingCleanupError,
] = _RetainedCleanupRegistry()
_RESOURCE_ERRNOS = frozenset(
    error
    for error in (
        getattr(errno, "EDQUOT", None),
        errno.ENFILE,
        errno.EMFILE,
        errno.ENOSPC,
    )
    if error is not None
)
_PRIVATE_FILE_MODE = 0o600


def write_regular(root: Path, relative_path: str, content: bytes) -> None:
    """Write one owner-private branch file through the guarded local primitive."""

    restore_regular(
        root,
        relative_path,
        content,
        mode=_PRIVATE_FILE_MODE,
    )


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    sha256: str
    bytes: int
    mode: int

    @property
    def revision(self) -> str:
        return f"sha256:{self.sha256}"

    def public(self) -> WorkspaceBranchContentIdentity:
        return WorkspaceBranchContentIdentity(sha256=self.sha256, bytes=self.bytes)


class _UnsupportedBranch(RuntimeError):
    def __init__(self, detail_code: str) -> None:
        self.detail_code = detail_code
        super().__init__(detail_code)


class _CreationConflict(RuntimeError):
    def __init__(self, conflicts: tuple[WorkspaceBranchConflict, ...]) -> None:
        self.conflicts = conflicts
        super().__init__("workspace_branch_baseline_conflicted")


@dataclass(frozen=True, slots=True)
class _CapturedBaseline:
    private_root: Path
    baseline_root: Path
    overlay_root: Path
    files: dict[str, _FileIdentity]
    directories: frozenset[str]
    root_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _PublicationReceipt:
    change_set_digest: str
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AppliedPublicationChange:
    change: WorkspaceBranchChange
    source_mode: int | None


def _required_source_mode(applied: _AppliedPublicationChange) -> int:
    if applied.source_mode is None:
        raise AssertionError("Applied source mutation has no rollback mode.")
    return applied.source_mode


@dataclass(frozen=True, slots=True)
class _RollbackReceipt:
    paths: tuple[str, ...]
    detail_code: str


class _PublicationConflictAccumulator:
    """Deduplicate and bound complete conflict evidence before retaining it."""

    def __init__(
        self,
        *,
        limits: WorkspaceBranchLimits,
        evidence_bytes_for_count: Callable[[int], int],
    ) -> None:
        self._limits = limits
        self._evidence_bytes_for_count = evidence_bytes_for_count
        self._conflicts: dict[str, tuple[WorkspaceBranchConflict, int]] = {}
        self._serialized_item_bytes = 0

    def add(self, conflict: WorkspaceBranchConflict) -> None:
        item_bytes = len(conflict.model_dump_json().encode("utf-8"))
        previous = self._conflicts.get(conflict.path)
        count = len(self._conflicts) + (previous is None)
        serialized_item_bytes = self._serialized_item_bytes + item_bytes
        if previous is not None:
            serialized_item_bytes -= previous[1]
        serialized_conflict_bytes = 2 + serialized_item_bytes + max(0, count - 1)
        if (
            count > self._limits.max_changed_paths
            or self._evidence_bytes_for_count(count) + serialized_conflict_bytes
            > self._limits.max_evidence_bytes
        ):
            raise WorkspaceBranchResourceExhaustedError("conflict_evidence_limit_exceeded")
        self._conflicts[conflict.path] = (conflict, item_bytes)
        self._serialized_item_bytes = serialized_item_bytes

    def result(self) -> tuple[WorkspaceBranchConflict, ...]:
        return tuple(self._conflicts[path][0] for path in sorted(self._conflicts))


class _BranchCapacityLease:
    """Release one admitted source slot only after its private tree is gone."""

    def __init__(self, source_key: tuple[object, ...]) -> None:
        self._source_key = source_key
        self._lock = threading.Lock()
        self._cleanup_settlement_lock = threading.Lock()
        self._retained_for_cleanup = False
        self._released = False

    def retain_for_cleanup(self) -> None:
        with self._lock:
            if not self._released:
                self._retained_for_cleanup = True

    def release_owner(self) -> None:
        with self._lock:
            if self._released or self._retained_for_cleanup:
                return
            self._released = True
        _release_branch(self._source_key)

    def release_after_cleanup(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        _release_branch(self._source_key)


_PRIVATE_TREE_CLEANUPS: _RetainedCleanupRegistry[
    tuple[Path, int],
    tuple[Path, _BranchCapacityLease],
] = _RetainedCleanupRegistry()


async def create_local_workspace_branch(
    source: LocalWorkspace,
    request: WorkspaceBranchRequest,
) -> WorkspaceBranchCreationResult:
    from cayu.workspaces.local import LocalWorkspace

    envelope = _copy_workspace_branch_request_envelope(request)
    if type(source) is not LocalWorkspace:
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            branch=None,
            evidence=_bounded_workspace_branch_evidence(
                source=envelope.source,
                baseline_revision=envelope.baseline_revision,
                outcome=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
                max_bytes=envelope.limits.max_evidence_bytes,
                detail_code="local_workspace_subclass_branching_unproven",
                hash_fixed_identity_on_overflow=True,
            ),
        )
    if envelope.source.workspace_id != source.id:
        raise ValueError("Workspace branch baseline belongs to a different workspace.")
    envelope_evidence_limit_violation = _fixed_authority_evidence_limit_violation(
        source=envelope.source,
        baseline_revision=envelope.baseline_revision,
        limits=envelope.limits,
    )
    if envelope_evidence_limit_violation is not None:
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch=None,
            evidence=_bounded_workspace_branch_evidence(
                source=envelope.source,
                baseline_revision=envelope.baseline_revision,
                outcome=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                max_bytes=envelope.limits.max_evidence_bytes,
                detail_code=envelope_evidence_limit_violation,
                hash_fixed_identity_on_overflow=True,
            ),
        )
    try:
        copied = copy_workspace_branch_request(request)
    except WorkspaceBranchResourceExhaustedError as exhausted:
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch=None,
            evidence=_bounded_workspace_branch_evidence(
                source=envelope.source,
                baseline_revision=envelope.baseline_revision,
                outcome=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                max_bytes=envelope.limits.max_evidence_bytes,
                detail_code=exhausted.detail_code,
                hash_fixed_identity_on_overflow=True,
            ),
        )
    request_limit_violation = _request_limit_violation(copied)
    if request_limit_violation is not None:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            detail_code=request_limit_violation,
        )
    evidence_limit_violation = _request_evidence_limit_violation(copied)
    if evidence_limit_violation is not None:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            detail_code=evidence_limit_violation,
        )
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            detail_code="source_identity_unavailable",
        )
    await _await_owned_thread(_retry_pending_branch_cleanups, source_key)
    if local_workspace_source_is_fenced(source.root):
        raise WorkspaceBranchFencedError(
            "Local workspace is fenced after an incomplete branch publication rollback."
        )
    if not _admit_branch(source_key, copied.limits.max_active_branches):
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            detail_code="active_branch_limit_exceeded",
        )
    capacity_lease = _BranchCapacityLease(source_key)

    branch_id = f"wsb_{uuid.uuid4().hex}"
    captured: _CapturedBaseline | None = None
    ownership_transferred = False
    try:
        captured_result = await _await_owned_thread(
            _capture_baseline,
            source.root,
            copied,
            branch_id,
            capacity_lease,
            abandon_result=lambda result: _discard_captured_baseline(
                result,
                capacity_lease,
            ),
        )
        captured = captured_result
        created_evidence = _bounded_branch_evidence(
            copied,
            source=copied.baseline.identity,
            baseline_revision=copied.baseline.revision,
            branch_id=branch_id,
            outcome=WorkspaceBranchOutcomeStatus.CREATED,
            paths=(),
            detail_code="workspace_branch_created",
        )
        branch = LocalWorkspaceBranch(
            source=source,
            branch_id=branch_id,
            request=copied,
            captured=captured,
            source_key=source_key,
            capacity_lease=capacity_lease,
        )
        ownership_transferred = True
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.CREATED,
            branch=branch,
            evidence=created_evidence,
        )
    except _CreationConflict as conflict:
        evidence = workspace_branch_evidence(
            source=copied.baseline.identity,
            baseline_revision=copied.baseline.revision,
            branch_id=branch_id,
            outcome=WorkspaceBranchOutcomeStatus.CONFLICTED,
            paths=tuple(item.path for item in conflict.conflicts),
            detail_code="workspace_branch_baseline_conflicted",
        )
        try:
            _validate_conflict_evidence_limit(
                conflict.conflicts,
                copied.limits.max_evidence_bytes,
                evidence=evidence,
            )
        except WorkspaceBranchResourceExhaustedError as exhausted:
            return _creation_result_without_branch(
                copied,
                WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                branch_id=branch_id,
                detail_code=exhausted.detail_code,
            )
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.CONFLICTED,
            branch=None,
            evidence=evidence,
            conflicts=conflict.conflicts,
        )
    except _UnsupportedBranch as unsupported:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            branch_id=branch_id,
            detail_code=unsupported.detail_code,
        )
    except WorkspaceBranchResourceExhaustedError as exhausted:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch_id=branch_id,
            detail_code=exhausted.detail_code,
        )
    except OSError as error:
        if error.errno not in _RESOURCE_ERRNOS:
            raise
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch_id=branch_id,
            detail_code="branch_storage_exhausted",
        )
    finally:
        if not ownership_transferred:
            try:
                if captured is not None:
                    await _await_owned_thread(
                        _discard_private_tree,
                        captured.private_root,
                        capacity_lease,
                    )
            finally:
                capacity_lease.release_owner()


def _creation_result_without_branch(
    request: WorkspaceBranchRequest,
    status: WorkspaceBranchOutcomeStatus,
    *,
    branch_id: str | None = None,
    detail_code: str,
) -> WorkspaceBranchCreationResult:
    return WorkspaceBranchCreationResult(
        status=status,
        branch=None,
        evidence=_bounded_branch_evidence(
            request,
            source=request.baseline.identity,
            baseline_revision=request.baseline.revision,
            branch_id=branch_id,
            outcome=status,
            detail_code=detail_code,
            hash_fixed_identity_on_overflow=True,
        ),
    )


def _request_limit_violation(request: WorkspaceBranchRequest) -> str | None:
    if len(request.baseline.paths) > request.limits.max_files:
        return "file_count_limit_exceeded"
    if len(request.baseline.paths) > request.limits.max_paths:
        return "path_count_limit_exceeded"
    if any(
        len(entry.path.encode("utf-8")) > request.limits.max_path_bytes
        for entry in request.baseline.paths
    ):
        return "path_byte_limit_exceeded"
    return None


def _request_evidence_limit_violation(request: WorkspaceBranchRequest) -> str | None:
    baseline_revision = request.baseline.revision
    if baseline_revision is None:  # pragma: no cover - request invariant
        raise AssertionError("Workspace branch baseline revision disappeared.")
    return _fixed_authority_evidence_limit_violation(
        source=request.baseline.identity,
        baseline_revision=baseline_revision,
        limits=request.limits,
    )


def _fixed_authority_evidence_limit_violation(
    *,
    source: WorkspaceIdentity,
    baseline_revision: str,
    limits: WorkspaceBranchLimits,
) -> str | None:
    branch_id = "wsb_" + "0" * 32
    digest = "sha256:" + "0" * 64
    if (
        _workspace_branch_empty_change_set_json_size(
            branch_id=branch_id,
            source=source,
            baseline_revision=baseline_revision,
        )
        > limits.max_evidence_bytes
    ):
        return "change_evidence_limit_exceeded"
    for outcome, change_set_digest, detail_code in (
        (
            WorkspaceBranchOutcomeStatus.CREATED,
            None,
            "workspace_branch_created",
        ),
        (
            WorkspaceBranchOutcomeStatus.COMMITTED,
            digest,
            "workspace_branch_committed",
        ),
        (
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            digest,
            "result_evidence_limit_exceeded",
        ),
    ):
        if (
            _workspace_branch_evidence_json_size(
                source=source,
                outcome=outcome,
                baseline_revision=baseline_revision,
                branch_id=branch_id,
                change_set_digest=change_set_digest,
                affected_path_count=0,
                detail_code=detail_code,
            )
            > limits.max_evidence_bytes
        ):
            return "result_evidence_limit_exceeded"
    return None


def _bounded_branch_evidence(
    request: WorkspaceBranchRequest,
    *,
    source: WorkspaceIdentity,
    outcome: WorkspaceBranchOutcomeStatus,
    baseline_revision: str | None,
    branch_id: str | None = None,
    change_set_digest: str | None = None,
    paths: Sequence[str] = (),
    detail_code: str | None = None,
    hash_fixed_identity_on_overflow: bool = False,
) -> WorkspaceBranchEvidence:
    return _bounded_workspace_branch_evidence(
        source=source,
        outcome=outcome,
        baseline_revision=baseline_revision,
        max_bytes=request.limits.max_evidence_bytes,
        branch_id=branch_id,
        change_set_digest=change_set_digest,
        paths=paths,
        detail_code=detail_code,
        hash_fixed_identity_on_overflow=hash_fixed_identity_on_overflow,
    )


def _admit_branch(source_key: tuple[object, ...], limit: int) -> bool:
    with _SOURCE_MANAGER_LOCK:
        active = _ACTIVE_BRANCHES.get(source_key, 0)
        if active >= limit:
            return False
        _ACTIVE_BRANCHES[source_key] = active + 1
        return True


def _release_branch(source_key: tuple[object, ...]) -> None:
    with _SOURCE_MANAGER_LOCK:
        active = _ACTIVE_BRANCHES.get(source_key, 0)
        if active <= 1:
            _ACTIVE_BRANCHES.pop(source_key, None)
        else:
            _ACTIVE_BRANCHES[source_key] = active - 1


def _capture_baseline(
    root: Path,
    request: WorkspaceBranchRequest,
    branch_id: str,
    capacity_lease: _BranchCapacityLease,
) -> _CapturedBaseline:
    private_root: Path | None = None
    captured: _CapturedBaseline | None = None
    primary_error: BaseException | None = None
    try:
        with workspace_source_lock(root, exclusive=True):
            before = root.stat()
            if not stat.S_ISDIR(before.st_mode):
                raise _UnsupportedBranch("source_root_is_not_directory")
            private_parent = root.parent
            if private_parent == root:
                raise _UnsupportedBranch("source_root_has_no_private_sibling")
            private_root = Path(
                tempfile.mkdtemp(
                    prefix=f".cayu-{branch_id}-",
                    dir=private_parent,
                )
            )
            os.chmod(private_root, 0o700)
            private_info = private_root.stat()
            if private_info.st_dev != before.st_dev:
                raise _UnsupportedBranch("private_storage_filesystem_mismatch")
            baseline_root = private_root / "baseline"
            overlay_root = private_root / "overlay"
            baseline_root.mkdir(mode=0o700)
            overlay_root.mkdir(mode=0o700)
            files, directories = _copy_regular_tree(root, baseline_root, request.limits)
            after = root.stat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise _CreationConflict(
                    (
                        WorkspaceBranchConflict(
                            path="__source_root__",
                            actual_kind="directory",
                        ),
                    )
                )
            actual_revision = _revision_for_files(files)
            conflicts = _baseline_conflicts(request, files, branch_id=branch_id)
            if actual_revision != request.baseline.revision or conflicts:
                if not conflicts:
                    conflicts = (
                        WorkspaceBranchConflict(
                            path="__baseline_revision__",
                            actual_kind="special",
                        ),
                    )
                raise _CreationConflict(conflicts)
            captured = _CapturedBaseline(
                private_root=private_root,
                baseline_root=baseline_root,
                overlay_root=overlay_root,
                files=files,
                directories=frozenset(directories),
                root_identity=(before.st_dev, before.st_ino),
            )
    except BaseException as error:
        primary_error = error
    if primary_error is not None:
        if private_root is not None:
            try:
                _discard_private_tree(private_root, capacity_lease)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
        raise primary_error
    if captured is None:  # pragma: no cover - successful capture invariant
        raise AssertionError("Workspace branch capture produced no baseline.")
    return captured


def _copy_regular_tree(
    source_root: Path,
    baseline_root: Path,
    limits: WorkspaceBranchLimits,
) -> tuple[dict[str, _FileIdentity], set[str]]:
    root_fd = os.open(
        source_root,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    files: dict[str, _FileIdentity] = {}
    directories: set[str] = set()
    scanned_paths = 0
    total_bytes = 0
    pending_directories = [""]
    try:
        while pending_directories:
            prefix = pending_directories.pop()
            directory_fd = _open_captured_directory(root_fd, prefix)
            try:
                # Revision encoding sorts the finished bounded manifest, so
                # traversal can stream entries in filesystem order.
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        scanned_paths += 1
                        if scanned_paths > limits.max_paths:
                            raise WorkspaceBranchResourceExhaustedError("path_count_limit_exceeded")
                        path = f"{prefix}/{entry.name}" if prefix else entry.name
                        _validate_captured_path(path, limits)
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode):
                            raise _UnsupportedBranch("source_contains_symlink")
                        if stat.S_ISDIR(info.st_mode):
                            directories.add(path)
                            (baseline_root / path).mkdir(mode=0o700, parents=False)
                            pending_directories.append(path)
                            continue
                        if not stat.S_ISREG(info.st_mode):
                            raise _UnsupportedBranch("source_contains_special_file")
                        if len(files) >= limits.max_files:
                            raise WorkspaceBranchResourceExhaustedError("file_count_limit_exceeded")
                        if info.st_size > limits.max_file_bytes:
                            raise WorkspaceBranchResourceExhaustedError("file_byte_limit_exceeded")
                        if total_bytes + info.st_size > limits.max_baseline_bytes:
                            raise WorkspaceBranchResourceExhaustedError(
                                "baseline_byte_limit_exceeded"
                            )
                        remaining_baseline_bytes = limits.max_baseline_bytes - total_bytes
                        copy_limit = min(
                            limits.max_file_bytes,
                            remaining_baseline_bytes,
                        )
                        target = baseline_root / path
                        identity = _copy_one_regular(
                            directory_fd,
                            entry.name,
                            target,
                            relative_path=path,
                            expected=info,
                            max_bytes=copy_limit,
                            limit_detail_code=(
                                "file_byte_limit_exceeded"
                                if copy_limit == limits.max_file_bytes
                                else "baseline_byte_limit_exceeded"
                            ),
                        )
                        total_bytes += identity.bytes
                        if total_bytes > limits.max_baseline_bytes:
                            raise WorkspaceBranchResourceExhaustedError(
                                "baseline_byte_limit_exceeded"
                            )
                        files[path] = identity
            finally:
                os.close(directory_fd)
    finally:
        os.close(root_fd)
    return files, directories


def _open_captured_directory(root_fd: int, path: str) -> int:
    current_fd = os.dup(root_fd)
    try:
        for part in path.split("/") if path else ():
            child_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _validate_captured_path(
    path: str,
    limits: WorkspaceBranchLimits,
) -> None:
    if _validate_workspace_relative_path(path) != path:
        raise _UnsupportedBranch("source_contains_noncanonical_path")
    if len(path.encode("utf-8")) > limits.max_path_bytes:
        raise WorkspaceBranchResourceExhaustedError("path_byte_limit_exceeded")


def _copy_one_regular(
    directory_fd: int,
    name: str,
    target: Path,
    *,
    relative_path: str,
    expected: os.stat_result,
    max_bytes: int,
    limit_detail_code: str,
) -> _FileIdentity:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    copied = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise _UnsupportedBranch("source_file_kind_changed_during_capture")
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise _CreationConflict(
                (WorkspaceBranchConflict(path=relative_path, actual_kind="file"),)
            )
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            _PRIVATE_FILE_MODE,
        )
        try:
            while True:
                chunk = os.read(descriptor, min(1 << 16, max_bytes + 1 - copied))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > max_bytes:
                    raise WorkspaceBranchResourceExhaustedError(limit_detail_code)
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            os.fchmod(target_fd, _PRIVATE_FILE_MODE)
        finally:
            os.close(target_fd)
        finished = os.fstat(descriptor)
        if (finished.st_dev, finished.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ) or finished.st_size != copied:
            raise _CreationConflict(
                (WorkspaceBranchConflict(path=relative_path, actual_kind="file"),)
            )
        return _FileIdentity(
            sha256=digest.hexdigest(),
            bytes=copied,
            mode=stat.S_IMODE(opened.st_mode),
        )
    finally:
        os.close(descriptor)


def _revision_for_files(files: dict[str, _FileIdentity]) -> str:
    manifest: list[dict[str, object]] = [
        {"path": path, "sha256": identity.sha256, "bytes": identity.bytes}
        for path, identity in sorted(files.items())
    ]
    return _deterministic_workspace_manifest_revision(
        _deterministic_workspace_manifest_bytes(manifest)
    )


def _baseline_conflicts(
    request: WorkspaceBranchRequest,
    actual: dict[str, _FileIdentity],
    *,
    branch_id: str,
) -> tuple[WorkspaceBranchConflict, ...]:
    expected = {entry.path: entry for entry in request.baseline.paths}
    conflicts: list[WorkspaceBranchConflict] = []
    serialized_conflicts_bytes = 2
    for path in sorted(expected.keys() | actual.keys()):
        expected_entry = expected.get(path)
        actual_entry = actual.get(path)
        if (
            expected_entry is not None
            and actual_entry is not None
            and expected_entry.kind == "file"
            and expected_entry.present is not False
            and expected_entry.content_sha256 == actual_entry.sha256
        ):
            continue
        item_bytes = _baseline_conflict_json_size(path, actual_entry)
        if conflicts:
            item_bytes += 1
        projected_evidence_bytes = _workspace_branch_evidence_json_size(
            source=request.baseline.identity,
            outcome=WorkspaceBranchOutcomeStatus.CONFLICTED,
            baseline_revision=request.baseline.revision,
            branch_id=branch_id,
            change_set_digest=None,
            affected_path_count=len(conflicts) + 1,
            detail_code="workspace_branch_baseline_conflicted",
        )
        if (
            projected_evidence_bytes + serialized_conflicts_bytes + item_bytes
            > request.limits.max_evidence_bytes
        ):
            raise WorkspaceBranchResourceExhaustedError("conflict_evidence_limit_exceeded")
        conflict = WorkspaceBranchConflict(
            path=path,
            actual=None if actual_entry is None else actual_entry.public(),
            actual_kind="missing" if actual_entry is None else "file",
        )
        serialized_conflicts_bytes += item_bytes
        conflicts.append(conflict)
    return tuple(conflicts)


def _baseline_conflict_json_size(path: str, actual: _FileIdentity | None) -> int:
    actual_size = len(b"null")
    actual_kind = "missing"
    if actual is not None:
        actual_kind = "file"
        actual_size = sum(
            (
                len(b'{"sha256":'),
                _json_text_size(actual.sha256),
                len(b',"bytes":'),
                len(str(actual.bytes)),
                1,
            )
        )
    return sum(
        (
            len(b'{"path":'),
            _json_text_size(path),
            len(b',"expected":null,"actual":'),
            actual_size,
            len(b',"actual_kind":'),
            _json_text_size(actual_kind),
            1,
        )
    )


def _validate_conflict_evidence_limit(
    conflicts: Sequence[WorkspaceBranchConflict],
    max_bytes: int,
    *,
    evidence: WorkspaceBranchEvidence | None = None,
) -> None:
    # Account incrementally so enforcing the evidence limit never requires a
    # second complete serialized copy of hostile path evidence.
    serialized_bytes = 2 + (
        0 if evidence is None else len(evidence.model_dump_json().encode("utf-8"))
    )
    for index, conflict in enumerate(conflicts):
        if type(conflict) is not WorkspaceBranchConflict:
            raise TypeError("Workspace branch conflict evidence is invalid.")
        if index:
            serialized_bytes += 1
        serialized_bytes += len(conflict.model_dump_json().encode("utf-8"))
        if serialized_bytes > max_bytes:
            raise WorkspaceBranchResourceExhaustedError("conflict_evidence_limit_exceeded")


class LocalWorkspaceBranch(WorkspaceBranch):
    def __init__(
        self,
        *,
        source: LocalWorkspace,
        branch_id: str,
        request: WorkspaceBranchRequest,
        captured: _CapturedBaseline,
        source_key: tuple[object, ...],
        capacity_lease: _BranchCapacityLease,
    ) -> None:
        self._source = source
        self._branch_id = branch_id
        self._limits = request.limits
        self._source_workspace_id = request.baseline.identity.workspace_id
        self._source_observer = request.baseline.identity.observer
        baseline_revision = request.baseline.revision
        if baseline_revision is None:  # pragma: no cover - request invariant
            raise AssertionError("Branch baseline revision disappeared.")
        self._baseline_revision: str = baseline_revision
        self._private_root = captured.private_root
        self._baseline_root = captured.baseline_root
        self._overlay_root = captured.overlay_root
        self._baseline = dict(captured.files)
        self._baseline_directories = captured.directories
        self._root_identity = captured.root_identity
        self._source_key = source_key
        self._capacity_lease = capacity_lease
        self._overlay: dict[str, _FileIdentity] = {}
        self._tombstones: set[str] = set()
        self._state_lock = threading.RLock()
        self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
        self._publication_receipt: _PublicationReceipt | None = None
        self._rollback_receipt: _RollbackReceipt | None = None
        self._expires_at = time.monotonic() + self._limits.lifetime_ms / 1000
        self._timer = threading.Timer(self._limits.lifetime_ms / 1000, self._expire)
        self._timer.daemon = True
        self._timer.start()
        self.id = f"{source.id}#branch:{branch_id}"

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def lifecycle_status(self) -> WorkspaceBranchLifecycleStatus:
        with self._state_lock:
            return self._lifecycle

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("local-workspace-branch", self._source_key, self._branch_id)

    def bounded_read_limit(self, max_bytes: int) -> int:
        if type(max_bytes) is not int:
            raise TypeError("Workspace max_bytes must be an integer.")
        if max_bytes <= 0:
            raise ValueError("Workspace max_bytes must be greater than zero.")
        return min(max_bytes, self._limits.max_file_bytes)

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        return await _await_owned_thread(self._read, path, offset, max_bytes)

    def _read(self, path: str, offset: int, max_bytes: int | None) -> WorkspaceReadResult:
        relative = self._validated_path(path)
        if type(offset) is not int:
            raise TypeError("Workspace offset must be an integer.")
        if offset < 0:
            raise ValueError("Workspace offset must be non-negative.")
        if max_bytes is not None:
            if type(max_bytes) is not int:
                raise TypeError("Workspace max_bytes must be an integer.")
            if max_bytes <= 0:
                raise ValueError("Workspace max_bytes must be greater than zero.")
            max_bytes = min(max_bytes, self._limits.max_file_bytes)
        with self._state_lock:
            self._require_active()
            self._validate_filesystem_request_path(relative)
            location, identity = self._visible_location(relative)
            if location is None or identity is None:
                raise FileNotFoundError(f"Workspace file not found: {relative}")
            snapshot = _read_regular_bytes(location, relative, expected=identity)
            total_bytes = len(snapshot)
            if offset > total_bytes:
                raise WorkspaceReadOffsetError(offset, total_bytes)
            content = (
                snapshot[offset:] if max_bytes is None else snapshot[offset : offset + max_bytes]
            )
            complete = offset == 0 and len(content) == total_bytes
            return WorkspaceReadResult(
                content=content,
                total_bytes=total_bytes,
                truncated=offset + len(content) < total_bytes,
                offset=offset,
                revision=identity.revision if complete else None,
                sha256=identity.sha256 if complete else None,
            )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        return await _await_owned_thread(self._list, pattern, limit)

    def _list(self, pattern: str, limit: int | None) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        if limit is not None and (type(limit) is not int or limit <= 0):
            if type(limit) is not int:
                raise TypeError("Workspace limit must be an integer.")
            raise ValueError("Workspace limit must be greater than zero.")
        with self._state_lock:
            self._require_active()
            collector = _WorkspaceListCollector(limit)
            for path in self._visible_paths():
                if matches_list_pattern(path, pattern):
                    collector.add(path)
            return collector.result(exact_total_when_truncated=False)

    async def write_bytes(self, path: str, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        await _await_owned_thread(self._write, path, content, False, None)

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        result = await _await_owned_thread(self._write, path, content, True, None)
        if result is None:  # pragma: no cover - create-only invariant
            raise AssertionError("Branch create returned no mutation result.")
        return result

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace replace content must be bytes.")
        if type(expected_revision) is not str or not expected_revision.strip():
            raise ValueError("Workspace expected_revision must be a nonblank string.")
        result = await _await_owned_thread(
            self._write,
            path,
            content,
            False,
            expected_revision,
        )
        if result is None:  # pragma: no cover - conditional mutation invariant
            raise AssertionError("Branch replace returned no mutation result.")
        return result

    def _write(
        self,
        path: str,
        content: bytes,
        create_only: bool,
        expected_revision: str | None,
    ) -> WorkspaceMutationResult | None:
        relative = self._validated_path(path)
        if len(content) > self._limits.max_file_bytes:
            raise WorkspaceBranchResourceExhaustedError("file_byte_limit_exceeded")
        after = _identity_for_bytes(content)
        with self._state_lock:
            self._require_active()
            self._validate_filesystem_request_path(relative)
            self._validate_file_location(relative)
            _location, before = self._visible_location(relative)
            if create_only and before is not None:
                raise FileExistsError(f"Workspace file already exists: {relative}")
            if expected_revision is not None:
                if before is None:
                    raise FileNotFoundError(f"Workspace file not found: {relative}")
                if before.revision != expected_revision:
                    raise WorkspaceRevisionMismatchError(expected_revision, before.revision)
            before_content = (
                _read_identity_bytes(self, relative, before)
                if before is not None and (create_only or expected_revision is not None)
                else None
            )
            proposed_overlay = dict(self._overlay)
            proposed_tombstones = set(self._tombstones)
            baseline = self._baseline.get(relative)
            if (
                baseline is not None
                and baseline.sha256 == after.sha256
                and baseline.bytes == after.bytes
            ):
                proposed_overlay.pop(relative, None)
                proposed_tombstones.discard(relative)
            else:
                proposed_overlay[relative] = after
                proposed_tombstones.discard(relative)
            self._validate_proposed(relative, proposed_overlay, proposed_tombstones)
            try:
                if relative in proposed_overlay:
                    write_regular(self._overlay_root, relative, content)
                else:
                    delete_regular(self._overlay_root, relative)
                    _prune_empty_overlay_ancestors(self._overlay_root, relative)
            except BaseException as primary:
                if isinstance(primary, _LocalGuardStagingCleanupError):
                    fencing_error = self._fence_private_staging_failure_unlocked(primary)
                    raise primary from fencing_error
                reconciliation_error = self._reconcile_overlay_mutation_failure_unlocked(
                    relative,
                    proposed_overlay,
                    proposed_tombstones,
                )
                if reconciliation_error is not None:
                    raise primary from reconciliation_error
                raise
            self._overlay = proposed_overlay
            self._tombstones = proposed_tombstones
            if expected_revision is None and not create_only:
                return None
            operation = "create" if before is None else "replace"
            return mutation_result(
                operation,
                before=before_content,
                after=content,
            )

    async def delete(self, path: str) -> None:
        await _await_owned_thread(self._delete, path, None)

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(expected_revision) is not str or not expected_revision.strip():
            raise ValueError("Workspace expected_revision must be a nonblank string.")
        result = await _await_owned_thread(self._delete, path, expected_revision)
        if result is None:  # pragma: no cover - expected revision guarantees a result
            raise AssertionError("Conditional branch delete returned no result.")
        return result

    def _delete(
        self,
        path: str,
        expected_revision: str | None,
    ) -> WorkspaceMutationResult | None:
        relative = self._validated_path(path)
        with self._state_lock:
            self._require_active()
            self._validate_filesystem_request_path(relative)
            if relative in self._logical_directories():
                raise IsADirectoryError(f"Workspace path is not a file: {relative}")
            _location, before = self._visible_location(relative)
            if before is None:
                if expected_revision is not None:
                    raise FileNotFoundError(f"Workspace file not found: {relative}")
                return None
            if expected_revision is not None and before.revision != expected_revision:
                raise WorkspaceRevisionMismatchError(expected_revision, before.revision)
            before_content = (
                _read_identity_bytes(self, relative, before)
                if expected_revision is not None
                else None
            )
            proposed_overlay = dict(self._overlay)
            proposed_tombstones = set(self._tombstones)
            proposed_overlay.pop(relative, None)
            if relative in self._baseline:
                proposed_tombstones.add(relative)
            else:
                proposed_tombstones.discard(relative)
            self._validate_proposed(relative, proposed_overlay, proposed_tombstones)
            try:
                delete_regular(self._overlay_root, relative)
                _prune_empty_overlay_ancestors(self._overlay_root, relative)
            except BaseException as primary:
                if isinstance(primary, _LocalGuardStagingCleanupError):
                    fencing_error = self._fence_private_staging_failure_unlocked(primary)
                    raise primary from fencing_error
                reconciliation_error = self._reconcile_overlay_mutation_failure_unlocked(
                    relative,
                    proposed_overlay,
                    proposed_tombstones,
                )
                if reconciliation_error is not None:
                    raise primary from reconciliation_error
                raise
            self._overlay = proposed_overlay
            self._tombstones = proposed_tombstones
            if expected_revision is None:
                return None
            return mutation_result("delete", before=before_content, after=None)

    async def changes(self) -> WorkspaceBranchChangeSet:
        return await _await_owned_thread(self._changes)

    def _changes(self) -> WorkspaceBranchChangeSet:
        with self._state_lock:
            self._require_active(allow_publishing=True)
            return self._change_set_unlocked()

    async def publish(
        self,
        request: WorkspaceBranchPublicationRequest,
    ) -> WorkspaceBranchPublicationResult:
        if type(request) is not WorkspaceBranchPublicationRequest:
            raise TypeError("Workspace branch publication request is invalid.")
        copied = WorkspaceBranchPublicationRequest(
            branch_id=request.branch_id,
            baseline_revision=request.baseline_revision,
            change_set_digest=request.change_set_digest,
        )
        return await _await_owned_thread(self._publish, copied)

    def _publish(
        self,
        request: WorkspaceBranchPublicationRequest,
    ) -> WorkspaceBranchPublicationResult:
        with self._state_lock:
            if self._lifecycle is WorkspaceBranchLifecycleStatus.COMMITTED:
                receipt = self._publication_receipt
                if receipt is None:  # pragma: no cover - state invariant
                    raise AssertionError("Committed branch has no receipt.")
                if (
                    request.branch_id != self._branch_id
                    or request.baseline_revision != self._baseline_revision
                    or request.change_set_digest != receipt.change_set_digest
                ):
                    raise WorkspaceBranchClosedError(
                        "Workspace branch already committed under different publication authority."
                    )
                _retry_retained_private_tree_cleanup(
                    self._private_root,
                    self._capacity_lease,
                )
                return self._publication_result_from_receipt(receipt)
            self._require_active()
            if (
                request.branch_id != self._branch_id
                or request.baseline_revision != self._baseline_revision
            ):
                raise ValueError(
                    "Workspace branch publication authority does not match the branch."
                )
            change_set = self._change_set_unlocked()
            if request.change_set_digest != change_set.digest:
                raise ValueError(
                    "Workspace branch changed after the requested change set was observed."
                )
            self._lifecycle = WorkspaceBranchLifecycleStatus.PUBLISHING
            try:
                result = self._publish_change_set(change_set)
            except BaseException as error:
                if self._lifecycle is WorkspaceBranchLifecycleStatus.COMMITTED:
                    self._finish_terminal_cleanup_unlocked()
                    raise
                if self._lifecycle is WorkspaceBranchLifecycleStatus.FENCED:
                    self._timer.cancel()
                    self._finish_terminal_cleanup_unlocked()
                    raise
                if isinstance(error, WorkspaceBranchResourceExhaustedError):
                    self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
                    return WorkspaceBranchPublicationResult(
                        status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                        evidence=self._evidence(
                            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                            change_set,
                            paths=(),
                            detail_code=error.detail_code,
                        ),
                    )
                if self._lifecycle is not WorkspaceBranchLifecycleStatus.FENCED:
                    self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
                raise
            if result.status is WorkspaceBranchOutcomeStatus.CONFLICTED:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
                return result
            if result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
                return result
            if self._lifecycle is not WorkspaceBranchLifecycleStatus.COMMITTED:
                raise AssertionError("Committed publication was not recorded before lock release.")
            self._finish_terminal_cleanup_unlocked()
            return result

    def _publish_change_set(
        self,
        change_set: WorkspaceBranchChangeSet,
    ) -> WorkspaceBranchPublicationResult:
        applied: list[_AppliedPublicationChange] = []
        with workspace_source_lock(
            self._source.root,
            exclusive=True,
            fence_on_cleanup_failure=(
                lambda: self._lifecycle is WorkspaceBranchLifecycleStatus.COMMITTED
            ),
        ):
            self._validate_source_root()
            conflicts, source_modes = self._publication_conflicts(change_set)
            if conflicts:
                evidence = self._evidence(
                    WorkspaceBranchOutcomeStatus.CONFLICTED,
                    change_set,
                    paths=tuple(conflict.path for conflict in conflicts),
                    detail_code="affected_path_conflicted",
                )
                _validate_conflict_evidence_limit(
                    conflicts,
                    self._limits.max_evidence_bytes,
                    evidence=evidence,
                )
                return WorkspaceBranchPublicationResult(
                    status=WorkspaceBranchOutcomeStatus.CONFLICTED,
                    evidence=evidence,
                    conflicts=conflicts,
                )
            created_directories = _missing_ancestor_paths(
                self._source.root,
                tuple(
                    change.path for change in change_set.changes if change.operation == "created"
                ),
                replaced_files=frozenset(
                    change.path for change in change_set.changes if change.operation == "deleted"
                ),
            )
            # Publication evidence is part of the atomic public outcome. Build
            # and bound it before the first irreversible source mutation.
            publication_paths = tuple(change.path for change in change_set.changes)
            committed_evidence = self._evidence(
                WorkspaceBranchOutcomeStatus.COMMITTED,
                change_set,
                paths=publication_paths,
                detail_code="workspace_branch_committed",
            )
            storage_exhausted_evidence = self._evidence(
                WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                change_set,
                paths=publication_paths,
                detail_code="publication_storage_exhausted",
            )
            committed_result = WorkspaceBranchPublicationResult(
                status=WorkspaceBranchOutcomeStatus.COMMITTED,
                evidence=committed_evidence,
            )
            committed_receipt = _PublicationReceipt(
                change_set_digest=change_set.digest,
                paths=publication_paths,
            )
            try:
                for change in change_set.changes:
                    # Treat every admitted attempt as acknowledgement-ambiguous:
                    # a guarded primitive can mutate successfully and then
                    # raise, so reverse recovery must include it either way.
                    source_mode = source_modes.get(change.path)
                    if change.operation != "created" and source_mode is None:
                        raise AssertionError(
                            "Publication preflight did not retain the source mode."
                        )
                    applied.append(
                        _AppliedPublicationChange(
                            change=change,
                            source_mode=source_mode,
                        )
                    )
                    with workspace_path_lock(self._source.root, change.path):
                        if change.operation == "created":
                            create_regular(
                                self._source.root,
                                change.path,
                                self._overlay_bytes(change.path),
                            )
                        elif change.operation == "modified":
                            before = self._baseline[change.path]
                            replace_regular_if_revision(
                                self._source.root,
                                change.path,
                                self._overlay_bytes(change.path),
                                before.revision,
                            )
                        else:
                            before = self._baseline[change.path]
                            delete_regular_if_revision(
                                self._source.root,
                                change.path,
                                before.revision,
                            )
                verification = self._published_identity_conflicts(change_set)
                if verification:
                    raise WorkspaceBranchPublicationError(
                        "Published workspace paths did not match the requested change set."
                    )
            except BaseException as primary:
                rollback_errors = self._restore_applied(
                    tuple(reversed(applied)),
                    created_directories=created_directories,
                )
                staging_cleanup_errors = tuple(
                    error
                    for error in (primary, *rollback_errors)
                    if isinstance(error, _LocalGuardStagingCleanupError)
                )
                if rollback_errors or staging_cleanup_errors:
                    self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
                    fence_local_workspace_source(self._source.root)
                    retention_errors: list[BaseException] = []
                    for cleanup_error in staging_cleanup_errors:
                        try:
                            _retain_source_staging_cleanup(
                                cleanup_error,
                                source_key=self._source_key,
                            )
                        except BaseException as retention_error:
                            retention_errors.append(retention_error)
                    failures = (primary, *rollback_errors, *retention_errors)
                    cause: BaseException = (
                        failures[0]
                        if len(failures) == 1
                        else BaseExceptionGroup(
                            "Workspace branch publication and rollback failures.",
                            list(failures),
                        )
                    )
                    raise WorkspaceBranchPublicationError(
                        "Workspace branch publication failed and source rollback was incomplete."
                    ) from cause
                if isinstance(primary, OSError) and primary.errno in _RESOURCE_ERRNOS:
                    return WorkspaceBranchPublicationResult(
                        status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                        evidence=storage_exhausted_evidence,
                    )
                raise WorkspaceBranchPublicationError(
                    "Workspace branch publication failed; the source was restored."
                ) from primary
            # Record the verified source outcome before fallible source-lock
            # teardown. A teardown failure may hide the first acknowledgement,
            # but an exact retry must still recover this committed result.
            self._publication_receipt = committed_receipt
            self._lifecycle = WorkspaceBranchLifecycleStatus.COMMITTED
            self._timer.cancel()
            return committed_result

    def _publication_conflicts(
        self,
        change_set: WorkspaceBranchChangeSet,
    ) -> tuple[tuple[WorkspaceBranchConflict, ...], dict[str, int]]:
        source = self._public_source_identity()
        conflicts = _PublicationConflictAccumulator(
            limits=self._limits,
            evidence_bytes_for_count=lambda count: _workspace_branch_evidence_json_size(
                source=source,
                outcome=WorkspaceBranchOutcomeStatus.CONFLICTED,
                baseline_revision=self._baseline_revision,
                branch_id=self._branch_id,
                change_set_digest=change_set.digest,
                affected_path_count=count,
                detail_code="affected_path_conflicted",
            ),
        )
        _filesystem_publication_alias_conflicts(
            self._source.root,
            change_set,
            limits=self._limits,
            conflicts=conflicts,
        )
        source_modes: dict[str, int] = {}
        deleted_paths = frozenset(
            change.path for change in change_set.changes if change.operation == "deleted"
        )
        for change in change_set.changes:
            expected = self._baseline.get(change.path)
            actual_path, actual_kind, actual = _inspect_source_path(
                self._source.root,
                change.path,
                max_bytes=self._limits.max_file_bytes,
                require_identity=change.operation != "created",
                identity_bytes=None if expected is None else expected.bytes,
            )
            if actual_path != change.path:
                if (
                    change.operation == "created"
                    and actual_path in deleted_paths
                    and actual_kind == "file"
                ):
                    actual_path, actual_kind, actual = change.path, "missing", None
                else:
                    conflicts.add(
                        WorkspaceBranchConflict(
                            path=actual_path,
                            actual_kind=actual_kind,
                        )
                    )
                    continue
            if change.operation == "created":
                matches = actual_kind == "missing"
            else:
                matches = (
                    expected is not None
                    and actual_kind == "file"
                    and actual is not None
                    and actual.sha256 == expected.sha256
                    and actual.bytes == expected.bytes
                )
            if not matches:
                conflicts.add(
                    WorkspaceBranchConflict(
                        path=change.path,
                        expected=None if expected is None else expected.public(),
                        actual=None if actual is None else actual.public(),
                        actual_kind=actual_kind,
                    )
                )
            elif actual is not None:
                source_modes[change.path] = actual.mode
        return conflicts.result(), source_modes

    def _published_identity_conflicts(
        self,
        change_set: WorkspaceBranchChangeSet,
    ) -> tuple[WorkspaceBranchConflict, ...]:
        conflicts: list[WorkspaceBranchConflict] = []
        created_ancestors = {
            "/".join(parts[: index + 1])
            for change in change_set.changes
            if change.operation == "created"
            for parts in (change.path.split("/")[:-1],)
            for index in range(len(parts))
        }
        for change in change_set.changes:
            actual_path, kind, actual = _inspect_source_path(
                self._source.root,
                change.path,
                max_bytes=self._limits.max_file_bytes,
            )
            expected = change.after
            if actual_path != change.path:
                matches = False
            elif expected is None:
                matches = kind == "missing" or (
                    kind == "directory" and change.path in created_ancestors
                )
            else:
                matches = kind == "file" and actual is not None and actual.public() == expected
            if not matches:
                conflicts.append(
                    WorkspaceBranchConflict(
                        path=actual_path,
                        expected=expected,
                        actual=None if actual is None else actual.public(),
                        actual_kind=kind,
                    )
                )
        return tuple(conflicts)

    def _restore_applied(
        self,
        applied_changes: Sequence[_AppliedPublicationChange],
        *,
        created_directories: tuple[str, ...],
    ) -> builtins.list[BaseException]:
        errors: builtins.list[BaseException] = []
        for applied in applied_changes:
            change = applied.change
            try:
                with workspace_path_lock(self._source.root, change.path):
                    actual_path, actual_kind, actual = _inspect_source_path(
                        self._source.root,
                        change.path,
                        max_bytes=self._limits.max_file_bytes,
                    )
                    if actual_path != change.path:
                        raise WorkspaceBranchFencedError(
                            "A source ancestor changed during publication rollback."
                        )
                    before = self._baseline.get(change.path)
                    after = self._overlay.get(change.path)
                    if before is not None and _same_content(actual, before):
                        continue
                    if (
                        change.operation == "deleted"
                        and actual_kind == "directory"
                        and change.path in created_directories
                    ):
                        _remove_created_directory_subtree(
                            self._source.root,
                            change.path,
                            created_directories,
                        )
                        actual_path, actual_kind, actual = _inspect_source_path(
                            self._source.root,
                            change.path,
                            max_bytes=self._limits.max_file_bytes,
                        )
                    if change.operation == "created":
                        if actual_kind == "missing":
                            continue
                        if after is None or not _same_content(actual, after):
                            raise WorkspaceBranchFencedError(
                                "A created source path changed during publication rollback."
                            )
                        delete_regular(self._source.root, change.path)
                    elif change.operation == "modified":
                        if after is None or not _same_content(actual, after):
                            raise WorkspaceBranchFencedError(
                                "A modified source path changed during publication rollback."
                            )
                        baseline = self._baseline[change.path]
                        content = _read_regular_bytes(
                            self._baseline_root,
                            change.path,
                            expected=baseline,
                        )
                        restore_regular(
                            self._source.root,
                            change.path,
                            content,
                            mode=_required_source_mode(applied),
                        )
                    else:
                        if actual_kind != "missing":
                            raise WorkspaceBranchFencedError(
                                "A deleted source path changed during publication rollback."
                            )
                        baseline = self._baseline[change.path]
                        content = _read_regular_bytes(
                            self._baseline_root,
                            change.path,
                            expected=baseline,
                        )
                        restore_regular(
                            self._source.root,
                            change.path,
                            content,
                            mode=_required_source_mode(applied),
                        )
            except BaseException as error:
                errors.append(error)
        for directory in reversed(created_directories):
            try:
                delete_empty_directory(self._source.root, directory)
            except FileNotFoundError:
                continue
            except OSError as error:
                if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    errors.append(error)
        return errors

    async def rollback(self) -> WorkspaceBranchRollbackResult:
        return await _await_owned_thread(self._rollback, "explicit")

    def _rollback(self, reason: str) -> WorkspaceBranchRollbackResult:
        with self._state_lock:
            if self._lifecycle is WorkspaceBranchLifecycleStatus.ROLLED_BACK:
                receipt = self._rollback_receipt
                if receipt is None:  # pragma: no cover - state invariant
                    raise AssertionError("Rolled-back branch has no receipt.")
                return self._rollback_result_from_receipt(receipt)
            if self._lifecycle is WorkspaceBranchLifecycleStatus.COMMITTED:
                raise WorkspaceBranchClosedError(
                    "A committed workspace branch cannot be rolled back."
                )
            if self._lifecycle is WorkspaceBranchLifecycleStatus.FENCED:
                raise WorkspaceBranchFencedError("Workspace branch is fenced.")
            self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLING_BACK
            receipt = _RollbackReceipt(
                paths=tuple(sorted(self._overlay.keys() | self._tombstones)),
                detail_code=(
                    "workspace_branch_expired"
                    if reason == "expired"
                    else "workspace_branch_rolled_back"
                ),
            )
            result = self._rollback_result_from_receipt(receipt)
            try:
                self._cleanup_and_release()
            except BaseException:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLING_BACK
                _retain_private_tree_cleanup(
                    self._private_root,
                    self._capacity_lease,
                )
                raise
            self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLED_BACK
            self._rollback_receipt = receipt
            self._timer.cancel()
            return result

    def _expire(self) -> None:
        try:
            self._rollback("expired")
        except (WorkspaceBranchClosedError, WorkspaceBranchFencedError):
            return
        except BaseException:
            retry = threading.Timer(1.0, self._expire)
            retry.daemon = True
            try:
                retry.start()
            except Exception:
                # ``_rollback`` retained the private tree before returning its
                # cleanup failure, so loss of this lifecycle retry cannot lose
                # resource ownership.
                return

    def _validated_path(self, path: str) -> str:
        relative = _validate_workspace_relative_path(path)
        if len(relative.encode("utf-8")) > self._limits.max_path_bytes:
            raise WorkspaceBranchResourceExhaustedError("path_byte_limit_exceeded")
        return relative

    def _validate_filesystem_request_path(self, path: str) -> None:
        """Reject only aliases the backing filesystems actually resolve as one entry."""

        logical_paths = set(self._visible_paths()) | self._logical_directories()
        for existing in logical_paths:
            if existing != path and any(
                _filesystem_paths_alias(root, existing, path)
                for root in (
                    self._source.root,
                    self._baseline_root,
                    self._overlay_root,
                )
            ):
                raise ValueError(
                    "Workspace path conflicts with an existing filesystem-equivalent path."
                )

    def _require_active(self, *, allow_publishing: bool = False) -> None:
        if self._lifecycle is WorkspaceBranchLifecycleStatus.FENCED:
            raise WorkspaceBranchFencedError("Workspace branch is fenced.")
        if self._lifecycle is WorkspaceBranchLifecycleStatus.PUBLISHING and allow_publishing:
            return
        if self._lifecycle is not WorkspaceBranchLifecycleStatus.ACTIVE:
            raise WorkspaceBranchClosedError(
                f"Workspace branch is not active: {self._lifecycle.value}."
            )
        if time.monotonic() >= self._expires_at:
            self._rollback("expired")
            raise WorkspaceBranchClosedError("Workspace branch expired.")

    def _visible_location(self, path: str) -> tuple[Path | None, _FileIdentity | None]:
        if path in self._tombstones:
            return None, None
        overlay = self._overlay.get(path)
        if overlay is not None:
            return self._overlay_root, overlay
        baseline = self._baseline.get(path)
        if baseline is not None:
            return self._baseline_root, baseline
        return None, None

    def _visible_paths(self) -> tuple[str, ...]:
        return tuple(sorted((self._baseline.keys() | self._overlay.keys()) - self._tombstones))

    def _logical_directories(self) -> set[str]:
        directories = set(self._baseline_directories)
        for visible_path in self._visible_paths():
            parts = visible_path.split("/")[:-1]
            directories.update("/".join(parts[: index + 1]) for index in range(len(parts)))
        return directories

    def _validate_file_location(self, path: str) -> None:
        visible = set(self._visible_paths())
        if path in self._logical_directories():
            raise IsADirectoryError(f"Workspace path is not a file: {path}")
        parts = path.split("/")[:-1]
        if any("/".join(parts[: index + 1]) in visible for index in range(len(parts))):
            raise NotADirectoryError(f"Workspace path parent is not a directory: {path}")

    def _validate_proposed(
        self,
        path: str,
        overlay: dict[str, _FileIdentity],
        tombstones: set[str],
    ) -> None:
        visible = (self._baseline.keys() | overlay.keys()) - tombstones
        if len(visible) > self._limits.max_files:
            raise WorkspaceBranchResourceExhaustedError("file_count_limit_exceeded")
        changed = overlay.keys() | tombstones
        if len(changed) > self._limits.max_changed_paths:
            raise WorkspaceBranchResourceExhaustedError("changed_path_limit_exceeded")
        if sum(identity.bytes for identity in overlay.values()) > self._limits.max_overlay_bytes:
            raise WorkspaceBranchResourceExhaustedError("overlay_byte_limit_exceeded")
        directories = set(self._baseline_directories)
        for visible_path in visible:
            parts = visible_path.split("/")[:-1]
            directories.update("/".join(parts[: index + 1]) for index in range(len(parts)))
        if len(visible) + len(directories) > self._limits.max_paths:
            raise WorkspaceBranchResourceExhaustedError("path_count_limit_exceeded")
        changes = self._changes_from(overlay, tombstones)
        digest = workspace_branch_change_set_digest(
            branch_id=self._branch_id,
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            changes=changes,
        )
        candidate = WorkspaceBranchChangeSet(
            branch_id=self._branch_id,
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            changes=changes,
            digest=digest,
        )
        if len(candidate.model_dump_json().encode("utf-8")) > self._limits.max_evidence_bytes:
            raise WorkspaceBranchResourceExhaustedError("change_evidence_limit_exceeded")
        del path

    def _changes_from(
        self,
        overlay: dict[str, _FileIdentity],
        tombstones: set[str],
    ) -> tuple[WorkspaceBranchChange, ...]:
        changes: list[WorkspaceBranchChange] = []
        for path in sorted(overlay.keys() | tombstones):
            before = self._baseline.get(path)
            after = overlay.get(path)
            if path in tombstones:
                if before is None:  # pragma: no cover - normalization invariant
                    continue
                operation = "deleted"
            elif before is None:
                operation = "created"
            else:
                operation = "modified"
            changes.append(
                WorkspaceBranchChange(
                    path=path,
                    operation=operation,
                    before=None if before is None else before.public(),
                    after=None if after is None else after.public(),
                )
            )
        return tuple(changes)

    def _change_set_unlocked(self) -> WorkspaceBranchChangeSet:
        changes = self._changes_from(self._overlay, self._tombstones)
        digest = workspace_branch_change_set_digest(
            branch_id=self._branch_id,
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            changes=changes,
        )
        result = WorkspaceBranchChangeSet(
            branch_id=self._branch_id,
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            changes=changes,
            digest=digest,
        )
        if len(result.model_dump_json().encode("utf-8")) > self._limits.max_evidence_bytes:
            raise WorkspaceBranchResourceExhaustedError("change_evidence_limit_exceeded")
        return result

    def _overlay_bytes(self, path: str) -> bytes:
        identity = self._overlay.get(path)
        if identity is None:  # pragma: no cover - change-set invariant
            raise WorkspaceBranchFencedError("Branch overlay identity is unavailable.")
        return _read_regular_bytes(self._overlay_root, path, expected=identity)

    def _reconcile_overlay_mutation_failure_unlocked(
        self,
        path: str,
        proposed_overlay: dict[str, _FileIdentity],
        proposed_tombstones: set[str],
    ) -> BaseException | None:
        """Reconcile a private mutation whose filesystem acknowledgement was lost."""

        prior = self._overlay.get(path)
        proposed = proposed_overlay.get(path)
        try:
            actual_path, actual_kind, actual = _inspect_source_path(
                self._overlay_root,
                path,
                max_bytes=self._limits.max_file_bytes,
            )
        except BaseException as inspection_error:
            return self._fence_private_branch_unlocked(inspection_error)
        if actual_path == path and (
            (prior is None and actual_kind == "missing") or _same_content(actual, prior)
        ):
            if prior is None:
                try:
                    _prune_empty_overlay_ancestors(self._overlay_root, path)
                except BaseException as cleanup_error:
                    return self._fence_private_branch_unlocked(cleanup_error)
            return None
        if actual_path == path and (
            (proposed is None and actual_kind == "missing") or _same_content(actual, proposed)
        ):
            if proposed is None:
                try:
                    _prune_empty_overlay_ancestors(self._overlay_root, path)
                except BaseException as cleanup_error:
                    return self._fence_private_branch_unlocked(cleanup_error)
            self._overlay = proposed_overlay
            self._tombstones = proposed_tombstones
            return None
        return self._fence_private_branch_unlocked(
            WorkspaceBranchFencedError(
                "Private workspace branch state changed during mutation reconciliation."
            )
        )

    def _fence_private_branch_unlocked(self, primary: BaseException) -> BaseException:
        self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
        self._timer.cancel()
        try:
            self._cleanup_and_release()
        except BaseException as cleanup_error:
            _retain_private_tree_cleanup(
                self._private_root,
                self._capacity_lease,
            )
            return BaseExceptionGroup(
                "Private workspace branch fencing and cleanup failures.",
                [primary, cleanup_error],
            )
        return primary

    def _fence_private_staging_failure_unlocked(
        self,
        staging_error: _LocalGuardStagingCleanupError,
    ) -> BaseException:
        fenced = WorkspaceBranchFencedError(
            "Private workspace branch staging cleanup was incomplete."
        )
        try:
            return self._fence_private_branch_unlocked(fenced)
        except BaseException as ownership_error:
            return BaseExceptionGroup(
                "Private workspace branch staging ownership failures.",
                [fenced, ownership_error],
            )
        finally:
            staging_error.release_cleanup_owner()

    def _public_source_identity(self) -> WorkspaceIdentity:
        return WorkspaceIdentity(
            workspace_id=self._source_workspace_id,
            observer=self._source_observer,
        )

    def _publication_result_from_receipt(
        self,
        receipt: _PublicationReceipt,
    ) -> WorkspaceBranchPublicationResult:
        evidence = _bounded_workspace_branch_evidence(
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            branch_id=self._branch_id,
            outcome=WorkspaceBranchOutcomeStatus.COMMITTED,
            change_set_digest=receipt.change_set_digest,
            paths=receipt.paths,
            detail_code="workspace_branch_committed",
            max_bytes=self._limits.max_evidence_bytes,
        )
        return WorkspaceBranchPublicationResult(
            status=WorkspaceBranchOutcomeStatus.COMMITTED,
            evidence=evidence,
        )

    def _rollback_result_from_receipt(
        self,
        receipt: _RollbackReceipt,
    ) -> WorkspaceBranchRollbackResult:
        evidence = _bounded_workspace_branch_evidence(
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            branch_id=self._branch_id,
            outcome=WorkspaceBranchOutcomeStatus.ROLLED_BACK,
            paths=receipt.paths,
            detail_code=receipt.detail_code,
            max_bytes=self._limits.max_evidence_bytes,
        )
        return WorkspaceBranchRollbackResult(
            status=WorkspaceBranchOutcomeStatus.ROLLED_BACK,
            evidence=evidence,
        )

    def _evidence(
        self,
        outcome: WorkspaceBranchOutcomeStatus,
        change_set: WorkspaceBranchChangeSet,
        *,
        paths: Sequence[str],
        detail_code: str,
    ) -> WorkspaceBranchEvidence:
        evidence = workspace_branch_evidence(
            source=self._public_source_identity(),
            baseline_revision=self._baseline_revision,
            branch_id=self._branch_id,
            outcome=outcome,
            change_set_digest=change_set.digest,
            paths=paths,
            detail_code=detail_code,
        )
        if len(evidence.model_dump_json().encode("utf-8")) > self._limits.max_evidence_bytes:
            raise WorkspaceBranchResourceExhaustedError("result_evidence_limit_exceeded")
        return evidence

    def _validate_source_root(self) -> None:
        info = self._source.root.stat()
        if (info.st_dev, info.st_ino) != self._root_identity:
            raise WorkspaceBranchFencedError("Local workspace root identity changed.")
        if local_workspace_source_is_fenced(self._source.root):
            raise WorkspaceBranchFencedError("Local workspace source is fenced.")

    def _cleanup_and_release(self) -> None:
        _settle_private_tree_cleanup(self._private_root, self._capacity_lease)

    def _finish_terminal_cleanup_unlocked(self) -> None:
        try:
            self._cleanup_and_release()
        except BaseException as cleanup_error:
            _retain_private_tree_cleanup(
                self._private_root,
                self._capacity_lease,
            )
            if not isinstance(cleanup_error, Exception):
                raise

    def _cleanup_committed(self) -> None:
        try:
            with self._state_lock:
                self._cleanup_and_release()
        except BaseException as cleanup_error:
            _retain_private_tree_cleanup(
                self._private_root,
                self._capacity_lease,
            )
            if not isinstance(cleanup_error, Exception):
                raise


def _identity_for_bytes(content: bytes, *, mode: int = 0o666) -> _FileIdentity:
    digest = hashlib.sha256(content).hexdigest()
    return _FileIdentity(sha256=digest, bytes=len(content), mode=mode)


def _prune_empty_overlay_ancestors(root: Path, path: str) -> None:
    """Remove private directories made unnecessary by a normalized net deletion."""

    parts = path.split("/")[:-1]
    for length in range(len(parts), 0, -1):
        directory = "/".join(parts[:length])
        try:
            delete_empty_directory(root, directory)
        except FileNotFoundError:
            continue
        except OSError as error:
            if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                break
            raise


def _potential_alias_path_key(path: str) -> str:
    return unicodedata.normalize("NFC", path.replace("\\", "/")).casefold()


def _filesystem_paths_alias(root: Path, first: str, second: str) -> bool:
    """Return whether distinct logical spellings resolve to one existing entry."""

    first_parts = first.split("/")
    second_parts = second.split("/")
    for length in range(1, min(len(first_parts), len(second_parts)) + 1):
        first_prefix = "/".join(first_parts[:length])
        second_prefix = "/".join(second_parts[:length])
        if first_prefix == second_prefix:
            continue
        try:
            if os.path.samefile(root / first_prefix, root / second_prefix):
                return True
        except OSError:
            continue
    return False


def _same_content(
    first: _FileIdentity | None,
    second: _FileIdentity | None,
) -> bool:
    return (
        first is not None
        and second is not None
        and first.sha256 == second.sha256
        and first.bytes == second.bytes
    )


def _filesystem_publication_alias_conflicts(
    root: Path,
    change_set: WorkspaceBranchChangeSet,
    *,
    limits: WorkspaceBranchLimits,
    conflicts: _PublicationConflictAccumulator,
) -> None:
    """Find late aliases by scanning only parents of affected logical paths."""

    expected_paths: dict[str, set[str]] = {}
    parent_paths: set[str] = set()
    for change in change_set.changes:
        parts = change.path.split("/")
        for length in range(1, len(parts) + 1):
            expected = "/".join(parts[:length])
            expected_paths.setdefault(_potential_alias_path_key(expected), set()).add(expected)
            parent_paths.add("/".join(parts[: length - 1]))
    if not expected_paths:
        return

    scanned_paths = 0
    root_fd = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for parent in sorted(parent_paths, key=lambda value: (value.count("/"), value)):
            try:
                directory_fd = _open_captured_directory(root_fd, parent)
            except OSError:
                # The ordinary affected-path inspection below classifies a
                # missing, replaced, or unsafe canonical parent.
                continue
            try:
                try:
                    with os.scandir(directory_fd) as entries:
                        for entry in entries:
                            scanned_paths += 1
                            if scanned_paths > limits.max_paths:
                                raise WorkspaceBranchResourceExhaustedError(
                                    "source_alias_scan_path_limit_exceeded"
                                )
                            candidate = f"{parent}/{entry.name}" if parent else entry.name
                            expected = expected_paths.get(_potential_alias_path_key(candidate))
                            if expected is None or (len(expected) == 1 and candidate in expected):
                                continue
                            if not any(
                                _filesystem_paths_alias(root, candidate, expected_path)
                                for expected_path in expected
                            ):
                                continue
                            if len(candidate.encode("utf-8")) > limits.max_path_bytes:
                                raise WorkspaceBranchResourceExhaustedError(
                                    "path_byte_limit_exceeded"
                                )
                            try:
                                info = entry.stat(follow_symlinks=False)
                            except OSError:
                                actual_kind: _PathKind = "special"
                            else:
                                if stat.S_ISLNK(info.st_mode):
                                    actual_kind = "symlink"
                                elif stat.S_ISDIR(info.st_mode):
                                    actual_kind = "directory"
                                elif stat.S_ISREG(info.st_mode):
                                    actual_kind = "file"
                                else:
                                    actual_kind = "special"
                            conflict = WorkspaceBranchConflict(
                                path=candidate,
                                actual_kind=actual_kind,
                            )
                            conflicts.add(
                                conflict,
                            )
                except OSError:
                    # Failure to enumerate one canonical parent is not proof
                    # that no filesystem-equivalent alias exists beneath it.
                    conflict_path = parent or change_set.changes[0].path
                    conflicts.add(
                        WorkspaceBranchConflict(
                            path=conflict_path,
                            actual_kind="special",
                        ),
                    )
            finally:
                os.close(directory_fd)
    finally:
        os.close(root_fd)


def _read_identity_bytes(
    branch: LocalWorkspaceBranch,
    path: str,
    identity: _FileIdentity,
) -> bytes:
    location, current = branch._visible_location(path)
    if location is None or current != identity:
        raise WorkspaceBranchFencedError("Private branch identity changed during mutation.")
    return _read_regular_bytes(location, path, expected=identity)


def _read_regular_bytes(
    root: Path,
    path: str,
    *,
    expected: _FileIdentity,
) -> bytes:
    with open_regular_for_read(root, path) as (file, total_bytes):
        if total_bytes != expected.bytes:
            raise WorkspaceBranchFencedError("Private branch content identity changed.")
        # The descriptor can grow after fstat(). Read one bounded sentinel byte
        # beyond the captured identity so concurrent growth is detected without
        # allocating the complete hostile value.
        content = file.read(expected.bytes + 1)
    if len(content) != expected.bytes:
        raise WorkspaceBranchFencedError("Private branch file changed during read.")
    if hashlib.sha256(content).hexdigest() != expected.sha256:
        raise WorkspaceBranchFencedError("Private branch content identity changed.")
    return content


def _remove_created_directory_subtree(
    root: Path,
    path: str,
    created_directories: tuple[str, ...],
) -> None:
    """Remove empty publication-created descendants before restoring a file."""

    descendant_prefix = f"{path}/"
    for directory in reversed(created_directories):
        if directory != path and not directory.startswith(descendant_prefix):
            continue
        try:
            delete_empty_directory(root, directory)
        except FileNotFoundError:
            continue


def _inspect_source_path(
    root: Path,
    path: str,
    *,
    max_bytes: int,
    require_identity: bool = True,
    identity_bytes: int | None = None,
) -> tuple[str, _PathKind, _FileIdentity | None]:
    """Inspect one source path without following any component or buffering content.

    Callers that only need positive evidence that a regular file exists can
    skip its content identity. Callers comparing against a known identity can
    also reject a size mismatch before opening or hashing the file.
    """

    root_fd = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    current_fd = root_fd
    parts = path.split("/")
    try:
        for index, part in enumerate(parts):
            current_path = "/".join(parts[: index + 1])
            final = index == len(parts) - 1
            try:
                info = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                return path, "missing", None
            if stat.S_ISLNK(info.st_mode):
                return current_path, "symlink", None
            if not final:
                if not stat.S_ISDIR(info.st_mode):
                    kind: _PathKind = "file" if stat.S_ISREG(info.st_mode) else "special"
                    return current_path, kind, None
                child_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current_fd,
                )
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = child_fd
                continue
            if stat.S_ISDIR(info.st_mode):
                return path, "directory", None
            if not stat.S_ISREG(info.st_mode):
                return path, "special", None
            if not require_identity or (
                identity_bytes is not None and info.st_size != identity_bytes
            ):
                return path, "file", None
            descriptor = os.open(
                part,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=current_fd,
            )
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                    info.st_dev,
                    info.st_ino,
                ):
                    return path, "special", None
                digest = hashlib.sha256()
                total = 0
                while chunk := os.read(descriptor, 1 << 16):
                    total += len(chunk)
                    if total > max_bytes:
                        return path, "file", None
                    digest.update(chunk)
                finished = os.fstat(descriptor)
                if (finished.st_dev, finished.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ) or finished.st_size != total:
                    return path, "special", None
                return (
                    path,
                    "file",
                    _FileIdentity(
                        sha256=digest.hexdigest(),
                        bytes=total,
                        mode=stat.S_IMODE(opened.st_mode),
                    ),
                )
            finally:
                os.close(descriptor)
        raise AssertionError("Workspace path inspection received an empty path.")
    except OSError:
        # Permission, replacement, and descriptor races are not positive
        # evidence that an affected source path still matches its baseline.
        return path, "special", None
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _missing_ancestor_paths(
    root: Path,
    paths: tuple[str, ...],
    *,
    replaced_files: frozenset[str],
) -> tuple[str, ...]:
    missing: set[str] = set()
    for path in paths:
        parts = path.split("/")[:-1]
        ancestor_missing = False
        for index in range(len(parts)):
            ancestor = "/".join(parts[: index + 1])
            if ancestor_missing or ancestor in replaced_files or not (root / ancestor).exists():
                missing.add(ancestor)
                ancestor_missing = True
    return tuple(sorted(missing, key=lambda value: (value.count("/"), value)))


def _remove_private_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _discard_captured_baseline(
    captured: _CapturedBaseline,
    capacity_lease: _BranchCapacityLease,
) -> None:
    _discard_private_tree(captured.private_root, capacity_lease)


def _discard_private_tree(path: Path, capacity_lease: _BranchCapacityLease) -> None:
    try:
        _settle_private_tree_cleanup(path, capacity_lease)
    except BaseException:
        _retain_private_tree_cleanup(path, capacity_lease)
        raise


def _retain_source_staging_cleanup(
    error: _LocalGuardStagingCleanupError,
    *,
    source_key: tuple[object, ...],
) -> None:
    _SOURCE_STAGING_CLEANUPS.retain(
        error,
        source_key=source_key,
        payload=error,
    )
    _schedule_source_staging_cleanup(error)


def _schedule_source_staging_cleanup(error: _LocalGuardStagingCleanupError) -> None:
    _schedule_retained_cleanup(
        _SOURCE_STAGING_CLEANUPS,
        error,
        _retry_source_staging_cleanup,
        eligible=lambda retained: retained.cleanup_owned,
    )


def _retry_source_staging_cleanup(error: _LocalGuardStagingCleanupError) -> None:
    settled = error.retry_cleanup()
    if settled:
        _SOURCE_STAGING_CLEANUPS.forget(error)
        return
    _SOURCE_STAGING_CLEANUPS.release_claim(error)
    _schedule_source_staging_cleanup(error)


def _schedule_retained_cleanup(
    registry: _RetainedCleanupRegistry[_CleanupKeyT, _CleanupPayloadT],
    key: _CleanupKeyT,
    retry_operation: Callable[[_CleanupKeyT], None],
    *,
    eligible: Callable[[_CleanupPayloadT], bool] | None = None,
) -> None:
    if not registry.claim(key, eligible=eligible):
        return
    try:
        retry = threading.Timer(1.0, retry_operation, args=(key,))
        retry.daemon = True
        retry.start()
    except Exception:
        # Thread creation can fail under process resource exhaustion. The
        # registry remains authoritative and a later branch entrance or exact
        # terminal retry assists it synchronously.
        registry.release_claim(key)
    except BaseException:
        registry.release_claim(key)
        raise


def _schedule_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> None:
    key = _private_tree_cleanup_key(path, capacity_lease)
    _schedule_retained_cleanup(
        _PRIVATE_TREE_CLEANUPS,
        key,
        _retry_private_tree_cleanup,
    )


def _retry_private_tree_cleanup(key: tuple[Path, int]) -> None:
    retained = _PRIVATE_TREE_CLEANUPS.payload(key)
    if retained is None:
        return
    path, capacity_lease = retained
    try:
        _settle_private_tree_cleanup(path, capacity_lease)
    except BaseException as cleanup_error:
        _PRIVATE_TREE_CLEANUPS.release_claim(key)
        _schedule_private_tree_cleanup(path, capacity_lease)
        if not isinstance(cleanup_error, Exception):
            raise


def _settle_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> None:
    """Join the exact lease owner before removing and releasing one private tree."""

    with capacity_lease._cleanup_settlement_lock:
        _remove_private_tree(path)
        capacity_lease.release_after_cleanup()
        _forget_private_tree_cleanup(path, capacity_lease)


def _retain_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> None:
    capacity_lease.retain_for_cleanup()
    key = _private_tree_cleanup_key(path, capacity_lease)
    _PRIVATE_TREE_CLEANUPS.retain(
        key,
        source_key=capacity_lease._source_key,
        payload=(path, capacity_lease),
    )
    _schedule_private_tree_cleanup(path, capacity_lease)


def _retry_pending_branch_cleanups(source_key: tuple[object, ...]) -> None:
    _retry_pending_private_tree_cleanups(source_key)
    _retry_pending_source_staging_cleanups(source_key)


def _retry_pending_source_staging_cleanups(source_key: tuple[object, ...]) -> None:
    pending = _SOURCE_STAGING_CLEANUPS.claim_pending(
        source_key,
        eligible=lambda retained: retained.cleanup_owned,
    )
    for error, _payload in pending:
        _retry_source_staging_cleanup(error)


def _retry_pending_private_tree_cleanups(source_key: tuple[object, ...]) -> None:
    pending = _PRIVATE_TREE_CLEANUPS.claim_pending(source_key)
    for key, _payload in pending:
        _retry_private_tree_cleanup(key)


def _retry_retained_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> None:
    key = _private_tree_cleanup_key(path, capacity_lease)
    if _PRIVATE_TREE_CLEANUPS.claim(key):
        _retry_private_tree_cleanup(key)


def _forget_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> None:
    key = _private_tree_cleanup_key(path, capacity_lease)
    _PRIVATE_TREE_CLEANUPS.forget(key)


def _private_tree_cleanup_key(
    path: Path,
    capacity_lease: _BranchCapacityLease,
) -> tuple[Path, int]:
    return path, id(capacity_lease)


async def _await_owned_thread(
    operation: Callable[..., _ResultT],
    *args: object,
    abandon_result: Callable[[_ResultT], None] | None = None,
) -> _ResultT:
    """Do not release branch ownership while a cancelled to_thread call still runs."""

    await asyncio.sleep(0)
    worker = asyncio.create_task(asyncio.to_thread(operation, *args))
    caller_signal: BaseException | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            if caller_signal is None:
                caller_signal = cancellation
            else:
                caller_signal.add_note(
                    "Additional caller cancellation arrived while draining branch work."
                )
            continue
        except BaseException as signal:
            if worker.done():
                break
            caller_signal = (
                signal
                if caller_signal is None
                else BaseExceptionGroup(
                    "Workspace branch operation received multiple control signals.",
                    [caller_signal, signal],
                )
            )
    try:
        result = worker.result()
    except BaseException as worker_error:
        if caller_signal is not None:
            raise caller_signal from worker_error
        raise
    if caller_signal is not None:
        if abandon_result is not None:
            cleanup_worker = asyncio.create_task(asyncio.to_thread(abandon_result, result))
            while not cleanup_worker.done():
                try:
                    await asyncio.shield(cleanup_worker)
                except asyncio.CancelledError:
                    caller_signal.add_note(
                        "Additional caller cancellation arrived while draining branch cleanup."
                    )
                    continue
                except BaseException:
                    if cleanup_worker.done():
                        break
            try:
                cleanup_worker.result()
            except BaseException as abandon_error:
                raise caller_signal from abandon_error
        raise caller_signal
    return result


__all__ = ["LocalWorkspaceBranch", "create_local_workspace_branch"]

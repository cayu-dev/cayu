from __future__ import annotations

import asyncio
import builtins
import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Hashable, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, NoReturn, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from cayu._exception_groups import exception_group_children
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.core.runtime_authority import SessionRunFenced
from cayu.workspaces._local_guard import (
    _LocalGuardStagingCleanupError,
    _LocalGuardStagingConflictError,
    create_regular,
    delete_empty_directory,
    delete_regular,
    delete_regular_if_revision,
    open_regular_for_read,
    replace_regular_if_revision,
    restore_regular,
    settle_regular_staging,
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
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityClaim,
    WorkspaceBranchBindingAuthorityClaimScope,
    WorkspaceBranchChange,
    WorkspaceBranchChangeSet,
    WorkspaceBranchClosedError,
    WorkspaceBranchConflict,
    WorkspaceBranchContentIdentity,
    WorkspaceBranchCreationResult,
    WorkspaceBranchDurableState,
    WorkspaceBranchEvidence,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchLimits,
    WorkspaceBranchOperationConflict,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationError,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchPublicationResult,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRecoveryResult,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceBranchRollbackRequest,
    WorkspaceBranchRollbackResult,
    WorkspaceBranchStore,
    WorkspaceBranchStoreDurability,
    _bounded_workspace_branch_evidence,
    _copy_workspace_branch_request_envelope,
    _json_text_size,
    _workspace_branch_evidence_json_size,
    _workspace_branch_fixed_authority_evidence_limit_violation,
    copy_workspace_branch_request,
    workspace_branch_change_set_digest,
    workspace_branch_evidence,
)
from cayu.workspaces.revisions import (
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionObservationStatus,
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
_ACTIVE_BRANCHES: dict[
    tuple[object, ...],
    dict[Hashable, _BranchCapacityOwner],
] = {}
_LINUX_IOCTL_READ = 2
_LINUX_FS_IOC_GETFLAGS_TYPE = ord("f")
_LINUX_FS_IOC_GETFLAGS_NUMBER = 1
_LINUX_FS_CASEFOLD_FL = 0x40000000
_DARWIN_PC_CASE_SENSITIVE = 11


class _DirectoryLookupSemantics(Enum):
    UNKNOWN = "unknown"
    CASE_SENSITIVE = "case_sensitive"
    UNICODE_CASEFOLDED = "unicode_casefolded"


def _directory_lookup_semantics(root: Path) -> _DirectoryLookupSemantics:
    """Return per-directory lookup semantics that can differ on one device."""

    if sys.platform == "darwin":
        try:
            case_sensitive = os.pathconf(root, _DARWIN_PC_CASE_SENSITIVE)
        except OSError:
            return _DirectoryLookupSemantics.UNKNOWN
        if case_sensitive == 1:
            return _DirectoryLookupSemantics.CASE_SENSITIVE
        if case_sensitive == 0:
            return _DirectoryLookupSemantics.UNICODE_CASEFOLDED
        return _DirectoryLookupSemantics.UNKNOWN
    if not sys.platform.startswith("linux"):
        return _DirectoryLookupSemantics.UNKNOWN

    import array
    import fcntl

    flags = array.array("L", [0])
    ioctl_request = (
        (_LINUX_IOCTL_READ << 30)
        | (flags.itemsize << 16)
        | (_LINUX_FS_IOC_GETFLAGS_TYPE << 8)
        | _LINUX_FS_IOC_GETFLAGS_NUMBER
    )
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        try:
            fcntl.ioctl(descriptor, ioctl_request, flags, True)
        except OSError as error:
            if error.errno in {errno.EINVAL, errno.ENOTTY, errno.EOPNOTSUPP}:
                return _DirectoryLookupSemantics.UNKNOWN
            raise
    finally:
        os.close(descriptor)
    if flags[0] & _LINUX_FS_CASEFOLD_FL:
        return _DirectoryLookupSemantics.UNICODE_CASEFOLDED
    return _DirectoryLookupSemantics.CASE_SENSITIVE


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
    ) -> bool:
        with self._lock:
            retained = self._records.get(key)
            if retained is None:
                self._records[key] = _RetainedCleanup(
                    source_key=source_key,
                    payload=payload,
                )
                return True
            if retained.source_key != source_key or retained.payload != payload:
                raise RuntimeError("Retained cleanup identity changed while owned.")
            return False

    def retain_or_existing(
        self,
        key: _CleanupKeyT,
        *,
        source_key: tuple[object, ...],
        payload: _CleanupPayloadT,
        replace_unclaimed: Callable[[_CleanupPayloadT], bool] | None = None,
    ) -> tuple[bool, _CleanupPayloadT]:
        """Retain one exact owner or return the equivalent existing owner."""

        with self._lock:
            retained = self._records.get(key)
            if retained is None:
                self._records[key] = _RetainedCleanup(
                    source_key=source_key,
                    payload=payload,
                )
                return True, payload
            if retained.source_key != source_key:
                raise RuntimeError("Retained cleanup identity changed while owned.")
            if (
                not retained.claimed
                and replace_unclaimed is not None
                and replace_unclaimed(retained.payload)
            ):
                retained.payload = payload
                return True, payload
            return False, retained.payload

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

    def pending_keys(
        self,
        source_key: tuple[object, ...],
    ) -> tuple[_CleanupKeyT, ...]:
        with self._lock:
            return tuple(
                key
                for key, retained in self._records.items()
                if retained.source_key == source_key and not retained.claimed
            )

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
    tuple[tuple[object, ...], str],
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
_PRIVATE_ROOT_OWNER_FILE = ".cayu-owner"
_PRIVATE_CLEANUP_CLAIM_PREFIX = "cayu.local-workspace-branch-cleanup.v1"
_PRIVATE_CLEANUP_CLAIM_MAX_BYTES = 1024
_PRIVATE_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_PRIVATE_OWNER_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4


def write_regular(root: Path, relative_path: str, content: bytes) -> None:
    """Write one owner-private branch file through the guarded local primitive."""

    restore_regular(
        root,
        relative_path,
        content,
        mode=_PRIVATE_FILE_MODE,
    )


def _project_overlay_write(
    relative: str,
    after: _FileIdentity,
    *,
    baseline: dict[str, _FileIdentity],
    overlay: dict[str, _FileIdentity],
    tombstones: set[str],
) -> tuple[dict[str, _FileIdentity], set[str]]:
    proposed_overlay = dict(overlay)
    proposed_tombstones = set(tombstones)
    baseline_identity = baseline.get(relative)
    if (
        baseline_identity is not None
        and baseline_identity.sha256 == after.sha256
        and baseline_identity.bytes == after.bytes
    ):
        proposed_overlay.pop(relative, None)
        proposed_tombstones.discard(relative)
    else:
        proposed_overlay[relative] = after
        proposed_tombstones.discard(relative)
    return proposed_overlay, proposed_tombstones


def _project_overlay_delete(
    relative: str,
    *,
    baseline: dict[str, _FileIdentity],
    overlay: dict[str, _FileIdentity],
    tombstones: set[str],
) -> tuple[dict[str, _FileIdentity], set[str]]:
    proposed_overlay = dict(overlay)
    proposed_tombstones = set(tombstones)
    proposed_overlay.pop(relative, None)
    if relative in baseline:
        proposed_tombstones.add(relative)
    else:
        proposed_tombstones.discard(relative)
    return proposed_overlay, proposed_tombstones


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
    private_root_owner: str | None
    baseline_root: Path
    overlay_root: Path
    files: dict[str, _FileIdentity]
    directories: frozenset[str]
    root_identity: tuple[int, int]
    path_semantics: _DirectoryLookupSemantics


@dataclass(frozen=True, slots=True)
class _BaselineCaptureAuthority:
    source: WorkspaceIdentity
    revision: str
    limits: WorkspaceBranchLimits
    expected_paths: tuple[WorkspacePathRevision, ...] | None


def _capture_authority_from_request(
    request: WorkspaceBranchRequest,
) -> _BaselineCaptureAuthority:
    revision = request.baseline.revision
    if revision is None:  # pragma: no cover - request invariant
        raise AssertionError("Workspace branch baseline revision disappeared.")
    return _BaselineCaptureAuthority(
        source=request.baseline.identity,
        revision=revision,
        limits=request.limits,
        expected_paths=request.baseline.paths,
    )


def _capture_authority_from_record(
    record: _DurableBranchRecord,
) -> _BaselineCaptureAuthority:
    return _BaselineCaptureAuthority(
        source=record.source,
        revision=record.baseline_revision,
        limits=record.limits,
        expected_paths=None,
    )


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


class _DurableFileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: StrictInt = Field(ge=0)
    mode: StrictInt = Field(ge=0)

    @classmethod
    def from_private(cls, identity: _FileIdentity) -> _DurableFileIdentity:
        return cls(sha256=identity.sha256, bytes=identity.bytes, mode=identity.mode)

    def private(self) -> _FileIdentity:
        return _FileIdentity(sha256=self.sha256, bytes=self.bytes, mode=self.mode)


class _DurablePublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    idempotency_key: str = Field(max_length=512)
    staging_owner: str = Field(pattern=r"^[0-9a-f]{32}$")
    change_set: WorkspaceBranchChangeSet

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "idempotency_key")


class _DurablePublicationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    idempotency_key: str = Field(max_length=512)
    change_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "idempotency_key")


def _durable_publication_staging_owner(
    *,
    private_root_owner: str,
    idempotency_key: str,
    change_set_digest: str,
) -> str:
    return hashlib.sha256(
        f"{private_root_owner}\0{idempotency_key}\0{change_set_digest}".encode()
    ).hexdigest()[:32]


class _DurableRollback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    idempotency_key: str = Field(max_length=512)
    reason: Literal["explicit", "expired"]
    paths: tuple[str, ...]

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "idempotency_key")


class _DurableFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    phase: Literal["creation", "publication"]
    outcome: Literal["conflicted", "unsupported", "resource_exhausted", "failed"]
    detail_code: str = Field(max_length=512)
    conflicts: tuple[WorkspaceBranchConflict, ...] = ()

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "detail_code")

    @model_validator(mode="after")
    def validate_conflicts(self) -> _DurableFailure:
        if self.phase == "publication" and self.outcome != "failed":
            raise ValueError("Durable publication failures require a failed outcome.")
        if (self.outcome == "conflicted") != bool(self.conflicts):
            raise ValueError("Durable creation conflicts require a conflicted outcome.")
        return self


def _parse_durable_branch_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Durable {field_name} must be a canonical UTC timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Durable {field_name} must be a canonical UTC timestamp.")
    if parsed.isoformat() != value:
        raise ValueError(f"Durable {field_name} must be a canonical UTC timestamp.")
    return parsed


class _DurableBranchRecord(BaseModel):
    """Complete reconstructible authority for one local branch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    branch_id: str = Field(max_length=256)
    creation_idempotency_key: str = Field(max_length=512)
    state: WorkspaceBranchDurableState
    source: WorkspaceIdentity
    source_root: str
    baseline_revision: str
    authority: WorkspaceBranchAuthority
    limits: WorkspaceBranchLimits
    private_root: str
    private_root_owner: str = Field(pattern=r"^[0-9a-f]{32}$")
    capture_attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    root_device: StrictInt
    root_inode: StrictInt
    created_at: str
    expires_at: str
    overlay: dict[str, _DurableFileIdentity] = Field(default_factory=dict)
    tombstones: tuple[str, ...] = ()
    publication_attempts: tuple[_DurablePublicationAttempt, ...] = ()
    publication: _DurablePublication | None = None
    rollback: _DurableRollback | None = None
    failure: _DurableFailure | None = None
    revision: StrictInt = Field(ge=0)

    @field_validator(
        "branch_id",
        "creation_idempotency_key",
        "source_root",
        "baseline_revision",
        "private_root",
        "created_at",
        "expires_at",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle_invariants(self) -> _DurableBranchRecord:
        created_at = _parse_durable_branch_timestamp(self.created_at, "created_at")
        expires_at = _parse_durable_branch_timestamp(self.expires_at, "expires_at")
        if expires_at - created_at != timedelta(milliseconds=self.limits.lifetime_ms):
            raise ValueError("Durable branch lifetime authority is invalid.")
        if (self.state is WorkspaceBranchDurableState.FAILED) != (self.failure is not None):
            raise ValueError("Durable failed state requires exact failure evidence.")
        rollback_states = {
            WorkspaceBranchDurableState.ROLLBACK_INTENT,
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
        }
        if (self.rollback is not None) != (self.state in rollback_states):
            raise ValueError("Durable rollback evidence does not match its lifecycle state.")
        if self.state in {
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
        }:
            expected_reason = (
                "expired" if self.state is WorkspaceBranchDurableState.EXPIRED else "explicit"
            )
            if self.rollback is None:  # pragma: no cover - rollback-state invariant above
                raise AssertionError("Durable rollback evidence disappeared.")
            if self.rollback.reason != expected_reason:
                raise ValueError("Durable rollback terminal reason does not match its state.")
        publication_required_states = {
            WorkspaceBranchDurableState.PUBLICATION_INTENT,
            WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
            WorkspaceBranchDurableState.COMMITTED,
            WorkspaceBranchDurableState.CONFLICTED,
        }
        if self.state in publication_required_states and self.publication is None:
            raise ValueError("Durable publication state requires publication evidence.")
        if (
            self.state
            in {
                WorkspaceBranchDurableState.CREATING,
                WorkspaceBranchDurableState.OPEN,
            }
            and self.publication is not None
        ):
            raise ValueError(
                "Durable publication evidence is incompatible with its lifecycle state."
            )
        if self.failure is not None:
            if self.failure.phase == "creation" and self.publication is not None:
                raise ValueError("Durable creation failure cannot carry publication evidence.")
            if self.failure.phase == "publication" and self.publication is None:
                raise ValueError("Durable publication failure requires publication evidence.")
        attempt_keys = tuple(attempt.idempotency_key for attempt in self.publication_attempts)
        if len(set(attempt_keys)) != len(attempt_keys):
            raise ValueError("Durable publication attempt identities must be unique.")
        if len(self.publication_attempts) > self.limits.max_publication_attempts:
            raise ValueError("Durable publication attempt evidence exceeds its limit.")
        if self.publication is not None and not any(
            attempt.idempotency_key == self.publication.idempotency_key
            and attempt.change_set_digest == self.publication.change_set.digest
            for attempt in self.publication_attempts
        ):
            raise ValueError("Durable publication is missing its attempt identity.")
        if self.publication is not None and self.publication.staging_owner != (
            _durable_publication_staging_owner(
                private_root_owner=self.private_root_owner,
                idempotency_key=self.publication.idempotency_key,
                change_set_digest=self.publication.change_set.digest,
            )
        ):
            raise ValueError("Durable publication staging authority is invalid.")
        retained_paths = tuple(self.overlay) + self.tombstones
        if len(set(retained_paths)) != len(retained_paths):
            raise ValueError("Durable private paths must be unique.")
        if self.tombstones != tuple(sorted(self.tombstones)):
            raise ValueError("Durable tombstones must be path ordered.")
        if len(retained_paths) > self.limits.max_paths:
            raise ValueError("Durable private path evidence exceeds its limit.")
        if len(self.overlay) > self.limits.max_files:
            raise ValueError("Durable overlay file evidence exceeds its limit.")
        if (
            sum(identity.bytes for identity in self.overlay.values())
            > self.limits.max_overlay_bytes
        ):
            raise ValueError("Durable overlay byte evidence exceeds its limit.")
        for path, identity in self.overlay.items():
            self._validate_retained_path(path)
            if identity.bytes > self.limits.max_file_bytes:
                raise ValueError("Durable overlay file evidence exceeds its byte limit.")
        for path in self.tombstones:
            self._validate_retained_path(path)
        if self.publication is not None:
            change_set = self.publication.change_set
            if (
                change_set.branch_id != self.branch_id
                or change_set.source != self.source
                or change_set.baseline_revision != self.baseline_revision
            ):
                raise ValueError("Durable publication authority does not match its branch.")
            if len(change_set.changes) > self.limits.max_changed_paths:
                raise ValueError("Durable publication change evidence exceeds its limit.")
            retained_path_set = set(retained_paths)
            if {change.path for change in change_set.changes} != retained_path_set:
                raise ValueError("Durable publication paths do not match retained branch state.")
            for change in change_set.changes:
                self._validate_retained_path(change.path)
                retained_identity = self.overlay.get(change.path)
                if change.operation == "deleted":
                    if change.path not in self.tombstones or retained_identity is not None:
                        raise ValueError("Durable deletion evidence does not match its tombstone.")
                    continue
                if (
                    retained_identity is None
                    or change.path in self.tombstones
                    or change.after is None
                    or change.after.sha256 != retained_identity.sha256
                    or change.after.bytes != retained_identity.bytes
                ):
                    raise ValueError(
                        "Durable publication content does not match retained branch state."
                    )
        if self.rollback is not None:
            if self.rollback.paths != tuple(sorted(set(self.rollback.paths))):
                raise ValueError("Durable rollback paths must be unique and path ordered.")
            if self.rollback.paths != tuple(sorted(retained_paths)):
                raise ValueError("Durable rollback paths do not match retained branch state.")
            if len(self.rollback.paths) > self.limits.max_changed_paths:
                raise ValueError("Durable rollback path evidence exceeds its limit.")
            for path in self.rollback.paths:
                self._validate_retained_path(path)
        return self

    def _validate_retained_path(self, path: str) -> None:
        if _validate_workspace_relative_path(path) != path:
            raise ValueError("Durable workspace branch paths must be canonical.")
        if len(path.encode("utf-8")) > self.limits.max_path_bytes:
            raise ValueError("Durable workspace branch path evidence exceeds its byte limit.")


def _durable_branch_creation_identity(record: _DurableBranchRecord) -> tuple[object, ...]:
    """Return every immutable field owned by one durable creation attempt."""

    authority = record.authority
    return (
        record.schema_version,
        record.branch_id,
        record.creation_idempotency_key,
        record.source.workspace_id,
        record.source.observer,
        record.source_root,
        record.baseline_revision,
        record.private_root,
        record.private_root_owner,
        record.capture_attempt_id,
        record.root_device,
        record.root_inode,
        record.created_at,
        record.expires_at,
        authority.session_id,
        authority.expected_run_epoch,
        authority.environment_name,
        authority.binding_generation,
        authority.binding_identity,
        authority.creating_authority,
        authority.resource_policy,
        tuple((name, getattr(record.limits, name)) for name in WorkspaceBranchLimits.model_fields),
    )


def _durable_branch_capacity_identity(record: _DurableBranchRecord) -> tuple[object, ...]:
    """Return the immutable authority tuple for one logical durable branch."""

    return ("durable", *_durable_branch_creation_identity(record))


def _durable_terminal_lifecycle(
    record: _DurableBranchRecord,
) -> WorkspaceBranchLifecycleStatus:
    if record.state is WorkspaceBranchDurableState.COMMITTED:
        return WorkspaceBranchLifecycleStatus.COMMITTED
    if record.state in {
        WorkspaceBranchDurableState.ROLLED_BACK,
        WorkspaceBranchDurableState.EXPIRED,
    }:
        return WorkspaceBranchLifecycleStatus.ROLLED_BACK
    if record.state in {
        WorkspaceBranchDurableState.FAILED,
        WorkspaceBranchDurableState.AMBIGUOUS,
    }:
        return WorkspaceBranchLifecycleStatus.FENCED
    raise AssertionError("Durable branch record is not terminal.")


def _mark_matching_branch_owner_terminal(
    source_key: tuple[object, ...],
    record: _DurableBranchRecord,
) -> _BranchCapacityOwner | None:
    with _SOURCE_MANAGER_LOCK:
        owner = _ACTIVE_BRANCHES.get(source_key, {}).get(_durable_branch_capacity_identity(record))
        if owner is not None:
            owner.mark_terminal(_durable_terminal_lifecycle(record))
    return owner


def _changes_for_retained_branch_state(
    baseline: dict[str, _FileIdentity],
    overlay: dict[str, _FileIdentity],
    tombstones: set[str],
) -> tuple[WorkspaceBranchChange, ...]:
    changes: list[WorkspaceBranchChange] = []
    for path in sorted(overlay.keys() | tombstones):
        before = baseline.get(path)
        after = overlay.get(path)
        if path in tombstones:
            if before is None:
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


def _validate_retained_publication_evidence(
    record: _DurableBranchRecord,
    baseline: dict[str, _FileIdentity],
    overlay: dict[str, _FileIdentity],
) -> None:
    publication = record.publication
    if publication is None:
        return
    expected_changes = _changes_for_retained_branch_state(
        baseline,
        overlay,
        set(record.tombstones),
    )
    if publication.change_set.changes != expected_changes:
        raise WorkspaceBranchFencedError(
            "Durable publication evidence conflicts with retained branch state."
        )


@dataclass(slots=True)
class _DurableBranchContext:
    store: WorkspaceBranchStore
    record: _DurableBranchRecord
    storage_key: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _DurablePublicationConflict(RuntimeError):
    def __init__(self, conflicts: tuple[WorkspaceBranchConflict, ...]) -> None:
        self.conflicts = conflicts
        super().__init__("durable workspace publication conflicted")


class _DurablePublicationAmbiguous(RuntimeError):
    def __init__(self, paths: tuple[str, ...]) -> None:
        self.paths = paths
        super().__init__("durable workspace publication is ambiguous")


class _DurablePublicationFailed(RuntimeError):
    def __init__(self, detail_code: str) -> None:
        self.detail_code = require_durable_clean_nonblank(detail_code, "detail_code")
        super().__init__(detail_code)


def _durable_branch_storage_key(branch_id: str) -> str:
    return "workspace-branch:" + hashlib.sha256(branch_id.encode("utf-8")).hexdigest()


def _durable_private_root(
    source_root: Path,
    branch_id: str,
    authority: WorkspaceBranchAuthority,
) -> Path:
    identity = hashlib.sha256(
        canonical_durable_json_bytes(
            {
                "source_root": str(source_root),
                "branch_id": branch_id,
                "authority": authority.model_dump(mode="json"),
            },
            "workspace_branch.private_root",
        )
    ).hexdigest()[:32]
    return source_root.parent / f".cayu-workspace-branch-{identity}"


def _durable_record_wrapper(record: _DurableBranchRecord) -> dict[str, object]:
    payload = record.model_dump(mode="json", warnings=False)
    digest = hashlib.sha256(
        canonical_durable_json_bytes(payload, "workspace_branch.record")
    ).hexdigest()
    return {
        "record_type": "cayu.local-workspace-branch",
        "payload": payload,
        "record_digest": digest,
    }


def _durable_record_from_wrapper(raw: object) -> _DurableBranchRecord:
    if type(raw) is not dict or set(raw) != {"record_type", "payload", "record_digest"}:
        raise WorkspaceBranchFencedError("Durable workspace branch record is corrupt.")
    record = cast("dict[str, object]", raw)
    if record.get("record_type") != "cayu.local-workspace-branch":
        raise WorkspaceBranchFencedError("Durable workspace branch record type is invalid.")
    payload = record.get("payload")
    digest = record.get("record_digest")
    if type(payload) is not dict or type(digest) is not str:
        raise WorkspaceBranchFencedError("Durable workspace branch record is corrupt.")
    expected = hashlib.sha256(
        canonical_durable_json_bytes(payload, "workspace_branch.record")
    ).hexdigest()
    if digest != expected:
        raise WorkspaceBranchFencedError("Durable workspace branch record digest is invalid.")
    try:
        return _DurableBranchRecord.model_validate(payload)
    except ValidationError as error:
        raise WorkspaceBranchFencedError(
            "Durable workspace branch record schema is invalid."
        ) from error


def _validated_durable_branch_record(record: _DurableBranchRecord) -> _DurableBranchRecord:
    """Revalidate an internally copied record before it reaches durable storage."""

    return _DurableBranchRecord.model_validate(record.model_dump(mode="python", warnings=False))


async def _load_durable_branch_record(
    store: WorkspaceBranchStore,
    *,
    session_id: str,
    storage_key: str,
) -> _DurableBranchRecord | None:
    raw = await store.load_workspace_branch_record(session_id, storage_key)
    return None if raw is None else _durable_record_from_wrapper(raw)


async def _publish_durable_branch_record(
    store: WorkspaceBranchStore,
    *,
    session_id: str,
    storage_key: str,
    expected: _DurableBranchRecord | None,
    updated: _DurableBranchRecord,
    expected_run_epoch: int,
    commit_guard: Callable[[], None] | None = None,
) -> _DurableBranchRecord:
    updated = _validated_durable_branch_record(updated)
    wrapper = _durable_record_wrapper(updated)

    def transform(current_raw: dict[str, Any] | None) -> dict[str, Any]:
        if current_raw is None:
            if expected is not None:
                raise WorkspaceBranchOperationConflict(
                    "Durable workspace branch record disappeared during transition."
                )
        else:
            current = _durable_record_from_wrapper(current_raw)
            if current == updated:
                return current_raw
            if expected is None or current != expected:
                raise WorkspaceBranchOperationConflict(
                    "Durable workspace branch changed under another operation."
                )
        return wrapper

    await store.publish_workspace_branch_record(
        session_id,
        storage_key,
        record_transform=transform,
        commit_guard=commit_guard,
        expected_run_epoch=expected_run_epoch,
    )
    return updated


def _raise_primary_with_reconciliation_failure(
    signal: BaseException,
    reconciliation_error: BaseException,
) -> NoReturn:
    """Keep the original operation signal authoritative during reconciliation."""

    if not isinstance(reconciliation_error, Exception):
        raise reconciliation_error from signal
    if reconciliation_error.__cause__ is None and reconciliation_error.__context__ is signal:
        reconciliation_error.__context__ = None
    existing_evidence = (
        signal.__cause__
        if signal.__cause__ is not None
        else (
            signal.__context__
            if signal.__context__ is not None and not signal.__suppress_context__
            else None
        )
    )
    secondary_evidence: BaseException = reconciliation_error
    if (
        existing_evidence is not None
        and existing_evidence is not reconciliation_error
        and existing_evidence is not signal
    ):
        secondary_evidence = BaseExceptionGroup(
            "Durable branch publication and reconciliation failures.",
            [existing_evidence, reconciliation_error],
        )
    raise signal from secondary_evidence


def _publication_staging_cleanup_error(
    signal: BaseException,
) -> _LocalGuardStagingCleanupError | None:
    """Find cleanup evidence delivered by this guarded publication attempt."""

    pending = [signal]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, _LocalGuardStagingCleanupError):
            return candidate
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(reversed(children))
        if candidate.__cause__ is not None:
            pending.append(candidate.__cause__)
        elif candidate.__context__ is not None and not candidate.__suppress_context__:
            pending.append(candidate.__context__)
    return None


async def _publish_or_reconcile_exact_record(
    store: WorkspaceBranchStore,
    *,
    session_id: str,
    storage_key: str,
    expected: _DurableBranchRecord | None,
    updated: _DurableBranchRecord,
    expected_run_epoch: int,
    commit_guard: Callable[[], None] | None = None,
    on_exact_control_signal: Callable[[_DurableBranchRecord], Awaitable[None]] | None = None,
    retain_uncertain_owner: Callable[[], None] | None = None,
) -> _DurableBranchRecord:
    """Publish one transition or prove its exact commit after acknowledgement loss."""

    updated = _validated_durable_branch_record(updated)
    guard_completed = commit_guard is None

    def guarded_commit() -> None:
        nonlocal guard_completed
        if commit_guard is None:  # pragma: no cover - wrapper construction invariant
            raise AssertionError("Durable branch commit guard disappeared.")
        commit_guard()
        guard_completed = True

    try:
        await _publish_durable_branch_record(
            store,
            session_id=session_id,
            storage_key=storage_key,
            expected=expected,
            updated=updated,
            expected_run_epoch=expected_run_epoch,
            commit_guard=None if commit_guard is None else guarded_commit,
        )
    except BaseException as signal:

        def retain_owner() -> BaseException | None:
            if retain_uncertain_owner is None:
                return None
            try:
                retain_uncertain_owner()
            except BaseException as ownership_error:
                return ownership_error
            return None

        try:
            loaded = await _load_durable_branch_record(
                store,
                session_id=session_id,
                storage_key=storage_key,
            )
        except BaseException as reconciliation_error:
            ownership_error = retain_owner()
            if ownership_error is not None:
                reconciliation_error = BaseExceptionGroup(
                    "Durable branch reconciliation and ownership-transfer failures.",
                    [reconciliation_error, ownership_error],
                )
            _raise_primary_with_reconciliation_failure(signal, reconciliation_error)
        if loaded is None or loaded != updated or not guard_completed:
            ownership_error = retain_owner()
            if ownership_error is not None:
                _raise_primary_with_reconciliation_failure(signal, ownership_error)
            raise
        if not isinstance(signal, Exception):
            if on_exact_control_signal is not None:
                try:
                    await on_exact_control_signal(loaded)
                except BaseException as settlement_error:
                    _raise_primary_with_reconciliation_failure(signal, settlement_error)
            raise
        return loaded
    return updated


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


class _BranchCapacityOwner:
    """One process-local capacity slot shared by an exact durable branch."""

    def __init__(self, source_key: tuple[object, ...], identity: Hashable) -> None:
        self.source_key = source_key
        self.identity = identity
        self.attachments = 0
        self.released = False
        self.cleanup_settlement_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._terminal_lifecycle: WorkspaceBranchLifecycleStatus | None = None
        self._active_generation = 0
        self._generation_uncertain = False

    def attachment_generation(self) -> int:
        with self._lifecycle_lock:
            return self._active_generation

    def mark_terminal(self, lifecycle: WorkspaceBranchLifecycleStatus) -> None:
        if lifecycle not in {
            WorkspaceBranchLifecycleStatus.COMMITTED,
            WorkspaceBranchLifecycleStatus.ROLLED_BACK,
            WorkspaceBranchLifecycleStatus.FENCED,
        }:
            raise AssertionError("Shared branch lifecycle must be terminal or fenced.")
        with self._lifecycle_lock:
            if self._terminal_lifecycle is None:
                self._terminal_lifecycle = lifecycle
            elif self._terminal_lifecycle is not lifecycle:
                self._terminal_lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            self._generation_uncertain = False

    def mark_uncertain(self) -> None:
        with self._lifecycle_lock:
            if self._terminal_lifecycle is None:
                self._generation_uncertain = True

    def lifecycle_for_generation(
        self,
        generation: int,
    ) -> WorkspaceBranchLifecycleStatus | None:
        with self._lifecycle_lock:
            if self._terminal_lifecycle is not None:
                return self._terminal_lifecycle
            if self._generation_uncertain or generation != self._active_generation:
                return WorkspaceBranchLifecycleStatus.FENCED
            return None

    def recover_generation(self, generation: int) -> int:
        """Activate one reconstructed generation without reviving stale attachments."""

        with self._lifecycle_lock:
            terminal = self._terminal_lifecycle
            if terminal is WorkspaceBranchLifecycleStatus.FENCED:
                raise WorkspaceBranchFencedError("Workspace branch is fenced.")
            if terminal is not None:
                raise WorkspaceBranchClosedError(
                    f"Workspace branch is not active: {terminal.value}."
                )
            if self._generation_uncertain:
                if generation != self._active_generation:
                    raise WorkspaceBranchFencedError(
                        "Workspace branch recovery attachment is stale."
                    )
                self._active_generation += 1
                self._generation_uncertain = False
            return self._active_generation


class _BranchCapacityLease:
    """Attach one handle or construction attempt to one logical branch slot."""

    def __init__(self, owner: _BranchCapacityOwner, generation: int) -> None:
        self._owner = owner
        self._generation = generation
        self._source_key = owner.source_key
        self._cleanup_settlement_lock = owner.cleanup_settlement_lock
        self._lock = threading.Lock()
        self._active = False
        self._retained_for_cleanup = False
        self._settled = False

    def activate(self) -> None:
        with self._lock:
            if self._settled:
                raise RuntimeError("Workspace branch capacity attachment was already settled.")
            self._active = True

    def mark_terminal(self, lifecycle: WorkspaceBranchLifecycleStatus) -> None:
        self._owner.mark_terminal(lifecycle)

    def mark_uncertain(self) -> None:
        self._owner.mark_uncertain()

    def terminal_lifecycle(self) -> WorkspaceBranchLifecycleStatus | None:
        return self._owner.lifecycle_for_generation(self._generation)

    def adopt_recovered_generation(self) -> None:
        with self._lock:
            if self._settled or self._retained_for_cleanup or self._active:
                raise RuntimeError(
                    "Workspace branch recovery generation cannot replace an active attachment."
                )
            self._generation = self._owner.recover_generation(self._generation)

    def retain_for_cleanup(self) -> None:
        with self._lock:
            if not self._settled:
                self._retained_for_cleanup = True

    def release_owner(self) -> None:
        with self._lock:
            if self._settled or self._retained_for_cleanup:
                return
            self._settled = True
            terminal = self._active
        _release_branch_attachment(self._owner, terminal=terminal)

    def release_after_cleanup(self) -> None:
        with self._lock:
            if self._settled:
                return
            self._settled = True
            terminal = self._active
        _release_branch_attachment(self._owner, terminal=terminal)

    def released(self) -> bool:
        with self._lock:
            return self._settled


class _BindingAuthorityClaimLease:
    """Transfer one exact binding claim to retained terminal settlement."""

    def __init__(
        self,
        claim: WorkspaceBranchBindingAuthorityClaim,
        expected: WorkspaceBranchBindingAuthority,
        source_key: tuple[object, ...],
    ) -> None:
        if not isinstance(claim, WorkspaceBranchBindingAuthorityClaim):
            raise TypeError(
                "Workspace branch binding authority claims must support transferable release."
            )
        self._claim = claim
        self._expected = expected
        self._source_key = source_key
        self._lock = threading.Lock()
        self._retained_for_settlement = False
        self._released = False

    def validate(self, authority: WorkspaceBranchAuthority) -> None:
        expected = self._expected
        with self._lock:
            if self._released:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch binding authority claim is no longer active."
                )
        if (
            expected.environment_name != authority.environment_name
            or expected.binding_generation != authority.binding_generation
            or expected.binding_identity != authority.binding_identity
        ):
            raise WorkspaceBranchOperationConflict(
                "Workspace branch binding authority claim does not match the operation."
            )

    def retain_for_settlement(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError("Binding authority claim was already released.")
            self._retained_for_settlement = True

    def release_owner(self) -> None:
        with self._lock:
            if self._released or self._retained_for_settlement:
                return
            self._claim.release()
            self._released = True

    def release_after_settlement(self) -> None:
        with self._lock:
            if self._released:
                return
            self._claim.release()
            self._released = True

    @property
    def source_key(self) -> tuple[object, ...]:
        return self._source_key


@dataclass(slots=True)
class _RetainedDurableTerminalSettlement:
    """Process-local owner for one exact terminal transition and cleanup."""

    branch: LocalWorkspaceBranch
    terminal_record: _DurableBranchRecord
    publication_expected: _DurableBranchRecord | None = field(compare=False)
    binding_claim_lease: _BindingAuthorityClaimLease = field(compare=False)


@dataclass(slots=True)
class _RetainedDurableRecoveryCleanup:
    """Own one fresh-process terminal cleanup and its exact binding claim."""

    source: LocalWorkspace = field(compare=False)
    record: _DurableBranchRecord
    binding_claim_lease: _BindingAuthorityClaimLease = field(compare=False)


_PRIVATE_TREE_CLEANUPS: _RetainedCleanupRegistry[
    tuple[Path, int],
    tuple[Path, _BranchCapacityLease, str | None],
] = _RetainedCleanupRegistry()

_DURABLE_TERMINAL_SETTLEMENTS: _RetainedCleanupRegistry[
    tuple[Path, int],
    _RetainedDurableTerminalSettlement,
] = _RetainedCleanupRegistry()
_DURABLE_RECOVERY_CLEANUPS: _RetainedCleanupRegistry[
    tuple[Path, str],
    _RetainedDurableRecoveryCleanup,
] = _RetainedCleanupRegistry()
_BINDING_CLAIM_RELEASES: _RetainedCleanupRegistry[
    tuple[tuple[object, ...], int],
    _BindingAuthorityClaimLease,
] = _RetainedCleanupRegistry()


def _scan_private_tree(
    root: Path,
    limits: WorkspaceBranchLimits,
    *,
    max_total_bytes: int,
    total_limit_detail_code: str,
) -> tuple[dict[str, _FileIdentity], frozenset[str]]:
    try:
        root_fd = os.open(root, _PRIVATE_DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise WorkspaceBranchFencedError("Durable workspace branch tree is unavailable.") from error
    try:
        root_identity = os.fstat(root_fd)
    except OSError as error:
        os.close(root_fd)
        raise WorkspaceBranchFencedError("Durable workspace branch tree is unavailable.") from error
    files: dict[str, _FileIdentity] = {}
    directories: set[str] = set()
    total_bytes = 0
    scanned_paths = 0
    pending_directories: list[tuple[str, tuple[int, int]]] = [
        ("", (root_identity.st_dev, root_identity.st_ino))
    ]
    try:
        while pending_directories:
            prefix, expected_directory_identity = pending_directories.pop()
            try:
                directory_fd = (
                    os.dup(root_fd) if not prefix else _open_captured_directory(root_fd, prefix)
                )
            except OSError as error:
                raise WorkspaceBranchFencedError(
                    "Durable workspace branch directory changed during reconstruction."
                ) from error
            try:
                opened_directory = os.fstat(directory_fd)
                if (
                    opened_directory.st_dev,
                    opened_directory.st_ino,
                ) != expected_directory_identity:
                    raise WorkspaceBranchFencedError(
                        "Durable workspace branch directory changed during reconstruction."
                    )
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        relative = f"{prefix}/{entry.name}" if prefix else entry.name
                        _validate_private_scanned_path(relative, limits)
                        scanned_paths += 1
                        if scanned_paths > limits.max_paths:
                            raise WorkspaceBranchResourceExhaustedError("path_count_limit_exceeded")
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except OSError as error:
                            raise WorkspaceBranchFencedError(
                                "Durable workspace branch path changed during reconstruction."
                            ) from error
                        if stat.S_ISDIR(info.st_mode):
                            child_identity = _open_private_scanned_directory(
                                directory_fd,
                                entry.name,
                                info,
                            )
                            directories.add(relative)
                            pending_directories.append((relative, child_identity))
                            continue
                        if not stat.S_ISREG(info.st_mode):
                            raise WorkspaceBranchFencedError(
                                "Durable workspace branch contains an unsafe private file."
                            )
                        if len(files) >= limits.max_files:
                            raise WorkspaceBranchResourceExhaustedError("file_count_limit_exceeded")
                        remaining_total_bytes = max_total_bytes - total_bytes
                        if info.st_size > limits.max_file_bytes:
                            raise WorkspaceBranchResourceExhaustedError("file_byte_limit_exceeded")
                        if info.st_size > remaining_total_bytes:
                            raise WorkspaceBranchResourceExhaustedError(total_limit_detail_code)
                        identity = _scan_private_regular(
                            directory_fd,
                            entry.name,
                            expected=info,
                            max_bytes=min(limits.max_file_bytes, remaining_total_bytes),
                            limit_detail_code=(
                                "file_byte_limit_exceeded"
                                if limits.max_file_bytes <= remaining_total_bytes
                                else total_limit_detail_code
                            ),
                        )
                        total_bytes += identity.bytes
                        files[relative] = identity
            finally:
                os.close(directory_fd)
        try:
            current_root = os.stat(root, follow_symlinks=False)
        except OSError as error:
            raise WorkspaceBranchFencedError(
                "Durable workspace branch tree changed during reconstruction."
            ) from error
        if not os.path.samestat(root_identity, current_root):
            raise WorkspaceBranchFencedError(
                "Durable workspace branch tree changed during reconstruction."
            )
        return files, frozenset(directories)
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable workspace branch tree changed during reconstruction."
        ) from error
    finally:
        os.close(root_fd)


def _validate_private_scanned_path(path: str, limits: WorkspaceBranchLimits) -> None:
    try:
        _validate_captured_path(path, limits)
    except (TypeError, ValueError, _UnsupportedBranch) as error:
        raise WorkspaceBranchFencedError(
            "Durable workspace branch contains an unsafe private path."
        ) from error


def _open_private_scanned_directory(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> tuple[int, int]:
    child_fd = -1
    try:
        child_fd = os.open(name, _PRIVATE_DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
        opened = os.fstat(child_fd)
        if not stat.S_ISDIR(opened.st_mode) or not os.path.samestat(expected, opened):
            raise WorkspaceBranchFencedError(
                "Durable workspace branch directory changed during reconstruction."
            )
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not os.path.samestat(opened, current):
            raise WorkspaceBranchFencedError(
                "Durable workspace branch directory changed during reconstruction."
            )
        return opened.st_dev, opened.st_ino
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable workspace branch directory changed during reconstruction."
        ) from error
    finally:
        if child_fd >= 0:
            os.close(child_fd)


def _scan_private_regular(
    directory_fd: int,
    name: str,
    *,
    expected: os.stat_result,
    max_bytes: int,
    limit_detail_code: str,
) -> _FileIdentity:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(expected, opened):
            raise WorkspaceBranchFencedError(
                "Durable workspace branch file changed during reconstruction."
            )
        if opened.st_size > max_bytes:
            raise WorkspaceBranchResourceExhaustedError(limit_detail_code)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 16, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise WorkspaceBranchResourceExhaustedError(limit_detail_code)
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if not os.path.samestat(opened, finished) or finished.st_size != total:
            raise WorkspaceBranchFencedError(
                "Durable workspace branch file changed during reconstruction."
            )
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not os.path.samestat(opened, current):
            raise WorkspaceBranchFencedError(
                "Durable workspace branch file changed during reconstruction."
            )
        return _FileIdentity(
            sha256=digest.hexdigest(),
            bytes=total,
            mode=stat.S_IMODE(opened.st_mode),
        )
    except WorkspaceBranchResourceExhaustedError:
        raise
    except WorkspaceBranchFencedError:
        raise
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable workspace branch file changed during reconstruction."
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validated_durable_private_root(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
) -> Path:
    _validate_durable_source_workspace_identity(source, record)
    private_root = Path(record.private_root)
    expected_private_root = _durable_private_root(
        source.root,
        record.branch_id,
        record.authority,
    )
    if private_root != expected_private_root or private_root.parent != source.root.parent:
        raise WorkspaceBranchFencedError("Durable private branch location is invalid.")
    return private_root


def _durable_private_root_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private branch ownership is unavailable."
        ) from error
    return True


def _remove_durable_private_tree(source: LocalWorkspace, record: _DurableBranchRecord) -> None:
    _require_durable_terminal_expiry_eligible(record)
    _validate_durable_source_root_identity(source, record)
    private_root = _validated_durable_private_root(source, record)
    if record.publication is not None and _durable_private_root_exists(private_root):
        try:
            _captured_from_private_root(source, record, private_root)
        except WorkspaceBranchFencedError:
            # An exact concurrent terminal recovery may have authenticated and
            # quarantined this tree after our existence check. Preserve every
            # other mismatch as corruption rather than deleting its evidence.
            if _durable_private_root_exists(private_root):
                raise
    _discard_owned_capture_staging(
        _capture_staging_root(private_root, record.capture_attempt_id),
        record.private_root_owner,
    )
    _remove_private_tree(
        private_root,
        private_root_owner=record.private_root_owner,
    )


def _settle_durable_private_tree_removal(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    capacity_lease: _BranchCapacityLease,
) -> None:
    """Serialize authenticated terminal removal with every cleanup owner."""

    with capacity_lease._cleanup_settlement_lock:
        _remove_durable_private_tree(source, record)


def _private_root_owner_bytes(owner: str) -> bytes:
    return f"cayu.local-workspace-branch-owner.v1\n{owner}\n".encode()


def _write_private_root_owner(path: Path, owner: str) -> None:
    root_fd = os.open(path, _PRIVATE_DIRECTORY_OPEN_FLAGS)
    try:
        _write_private_root_owner_to_fd(root_fd, owner)
    finally:
        os.close(root_fd)


def _write_private_root_owner_to_fd(root_fd: int, owner: str) -> None:
    marker_fd = -1
    published = False
    try:
        marker_fd = os.open(
            _PRIVATE_ROOT_OWNER_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            _PRIVATE_FILE_MODE,
            dir_fd=root_fd,
        )
        with os.fdopen(marker_fd, "wb") as marker:
            marker_fd = -1
            marker.write(_private_root_owner_bytes(owner))
            marker.flush()
            os.fsync(marker.fileno())
        os.fsync(root_fd)
        published = True
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        if not published:
            with suppress(OSError):
                os.unlink(_PRIVATE_ROOT_OWNER_FILE, dir_fd=root_fd)


def _require_private_root_owner(directory_fd: int, owner: str) -> None:
    expected = _private_root_owner_bytes(owner)
    marker_fd = -1
    try:
        marker_fd = os.open(
            _PRIVATE_ROOT_OWNER_FILE,
            _PRIVATE_OWNER_OPEN_FLAGS,
            dir_fd=directory_fd,
        )
        marker_info = os.fstat(marker_fd)
        if not stat.S_ISREG(marker_info.st_mode) or marker_info.st_size != len(expected):
            raise WorkspaceBranchFencedError(
                "Durable private branch ownership evidence is invalid."
            )
        content = bytearray()
        while len(content) <= len(expected):
            chunk = os.read(marker_fd, len(expected) + 1 - len(content))
            if not chunk:
                break
            content.extend(chunk)
        current = os.stat(
            _PRIVATE_ROOT_OWNER_FILE,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not os.path.samestat(marker_info, current) or bytes(content) != expected:
            raise WorkspaceBranchFencedError(
                "Durable private branch ownership evidence is invalid."
            )
    except FileNotFoundError as error:
        raise WorkspaceBranchFencedError(
            "Durable private branch ownership evidence is missing."
        ) from error
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private branch ownership evidence is unavailable."
        ) from error
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)


def _validate_private_root_owner(path: Path, owner: str) -> None:
    try:
        directory_fd = os.open(path, _PRIVATE_DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private branch ownership is unavailable."
        ) from error
    try:
        _require_private_root_owner(directory_fd, owner)
    finally:
        os.close(directory_fd)


def _call_private_root_rename(function_name: str, arguments: tuple[object, ...]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(libc, function_name)
    except AttributeError as error:
        raise _UnsupportedBranch("atomic_private_root_publication_unavailable") from error
    function.restype = ctypes.c_int
    if function(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))


def _rename_private_root_no_replace(
    source: Path,
    target: Path,
    *,
    parent_fd: int,
) -> None:
    if sys.platform == "darwin":
        _call_private_root_rename(
            "renameatx_np",
            (
                parent_fd,
                os.fsencode(source.name),
                parent_fd,
                os.fsencode(target.name),
                _RENAME_EXCL,
            ),
        )
        return
    if sys.platform.startswith("linux"):
        _call_private_root_rename(
            "renameat2",
            (
                parent_fd,
                os.fsencode(source.name),
                parent_fd,
                os.fsencode(target.name),
                _RENAME_NOREPLACE,
            ),
        )
        return
    raise _UnsupportedBranch("atomic_private_root_publication_unavailable")


def _capture_staging_root(private_root: Path, capture_attempt_id: str) -> Path:
    return private_root.with_name(f"{private_root.name}.capture-{capture_attempt_id}")


def _remove_incomplete_capture_staging(path: Path, owner: str) -> None:
    """Remove only the exact pre-owner shape of one durable capture attempt."""

    parent_fd = os.open(path.parent, _PRIVATE_DIRECTORY_OPEN_FLAGS)
    directory_fd = -1
    marker_fd = -1
    try:
        try:
            directory_fd = os.open(
                path.name,
                _PRIVATE_DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        owned_identity = os.fstat(directory_fd)
        with os.scandir(directory_fd) as entries:
            names = [entry.name for entry in entries]
        if names not in ([], [_PRIVATE_ROOT_OWNER_FILE]):
            raise WorkspaceBranchFencedError(
                "Incomplete durable capture staging contains unowned material."
            )
        if names:
            expected = _private_root_owner_bytes(owner)
            try:
                marker_fd = os.open(
                    _PRIVATE_ROOT_OWNER_FILE,
                    _PRIVATE_OWNER_OPEN_FLAGS,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                _remove_empty_owned_directory(
                    parent_fd,
                    path,
                    directory_fd,
                    owned_identity,
                    detail="Incomplete durable capture ownership changed during cleanup.",
                )
                return
            except OSError as error:
                raise WorkspaceBranchFencedError(
                    "Incomplete durable capture ownership is unavailable."
                ) from error
            marker_info = os.fstat(marker_fd)
            content = bytearray()
            while len(content) <= len(expected):
                chunk = os.read(marker_fd, len(expected) + 1 - len(content))
                if not chunk:
                    break
                content.extend(chunk)
            if (
                not stat.S_ISREG(marker_info.st_mode)
                or len(content) >= len(expected)
                or not expected.startswith(bytes(content))
            ):
                raise WorkspaceBranchFencedError("Incomplete durable capture ownership is invalid.")
            try:
                current_marker = os.stat(
                    _PRIVATE_ROOT_OWNER_FILE,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                _remove_empty_owned_directory(
                    parent_fd,
                    path,
                    directory_fd,
                    owned_identity,
                    detail="Incomplete durable capture ownership changed during cleanup.",
                )
                return
            if not os.path.samestat(marker_info, current_marker):
                raise WorkspaceBranchFencedError(
                    "Incomplete durable capture ownership changed during cleanup."
                )
            try:
                os.unlink(_PRIVATE_ROOT_OWNER_FILE, dir_fd=directory_fd)
            except FileNotFoundError:
                _remove_empty_owned_directory(
                    parent_fd,
                    path,
                    directory_fd,
                    owned_identity,
                    detail="Incomplete durable capture ownership changed during cleanup.",
                )
                return
        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not os.path.samestat(owned_identity, current):
            raise WorkspaceBranchFencedError(
                "Incomplete durable capture ownership changed during cleanup."
            )
        try:
            os.rmdir(path.name, dir_fd=parent_fd)
        except FileNotFoundError:
            return
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


def _discard_owned_capture_staging(path: Path, owner: str) -> None:
    try:
        _remove_private_tree(path, private_root_owner=owner)
    except _CleanupQuarantineReplacement:
        raise
    except WorkspaceBranchFencedError as ownership_error:
        try:
            _remove_incomplete_capture_staging(path, owner)
        except WorkspaceBranchFencedError as incomplete_error:
            raise ownership_error from incomplete_error


def _validate_durable_source_workspace_identity(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
) -> None:
    if str(source.root) != record.source_root or source.id != record.source.workspace_id:
        raise WorkspaceBranchOperationConflict(
            "Durable workspace branch belongs to a different source workspace."
        )


def _validate_durable_source_root_identity(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
) -> None:
    _validate_durable_source_workspace_identity(source, record)
    source_info = source.root.stat()
    if (source_info.st_dev, source_info.st_ino) != (
        record.root_device,
        record.root_inode,
    ):
        raise WorkspaceBranchFencedError("Local workspace root identity changed.")


def _captured_from_private_root(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    private_root: Path,
) -> _CapturedBaseline:
    _validate_durable_source_root_identity(source, record)
    _validate_private_root_owner(private_root, record.private_root_owner)
    baseline_root = private_root / "baseline"
    overlay_root = private_root / "overlay"
    baseline, directories = _scan_private_tree(
        baseline_root,
        record.limits,
        max_total_bytes=record.limits.max_baseline_bytes,
        total_limit_detail_code="baseline_byte_limit_exceeded",
    )
    if _revision_for_files(baseline) != record.baseline_revision:
        raise WorkspaceBranchFencedError("Durable branch baseline evidence is corrupt.")
    overlay, _overlay_directories = _scan_private_tree(
        overlay_root,
        record.limits,
        max_total_bytes=record.limits.max_overlay_bytes,
        total_limit_detail_code="overlay_byte_limit_exceeded",
    )
    expected_overlay = {path: identity.private() for path, identity in record.overlay.items()}
    if overlay.keys() != expected_overlay.keys() or any(
        overlay[path].sha256 != expected.sha256 or overlay[path].bytes != expected.bytes
        for path, expected in expected_overlay.items()
    ):
        raise WorkspaceBranchFencedError("Durable branch overlay evidence is ambiguous.")
    _validate_retained_publication_evidence(record, baseline, overlay)
    source_semantics = _directory_lookup_semantics(source.root)
    if source_semantics is _DirectoryLookupSemantics.UNKNOWN or any(
        _directory_lookup_semantics(path) is not source_semantics
        for path in (baseline_root, overlay_root)
    ):
        raise WorkspaceBranchFencedError("Durable branch path semantics changed.")
    return _CapturedBaseline(
        private_root=private_root,
        private_root_owner=record.private_root_owner,
        baseline_root=baseline_root,
        overlay_root=overlay_root,
        files=baseline,
        directories=directories,
        root_identity=(record.root_device, record.root_inode),
        path_semantics=source_semantics,
    )


def _captured_from_durable_record(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
) -> _CapturedBaseline:
    return _captured_from_private_root(
        source,
        record,
        _validated_durable_private_root(source, record),
    )


def _require_durable_branch_support(source: LocalWorkspace) -> None:
    if source._branch_store is None:
        raise RuntimeError("Durable workspace branches require branch_store.")
    if source._branch_store_durability is not WorkspaceBranchStoreDurability.DURABLE:
        raise RuntimeError(
            "Durable workspace branches require a branch store with durability='durable'."
        )
    provider = source._branch_authority_resolver
    if provider is None:
        raise RuntimeError(
            "Durable workspace branches require LocalWorkspace branch_authority_resolver."
        )
    if source._branch_claim_scope is not WorkspaceBranchBindingAuthorityClaimScope.DURABLE:
        raise RuntimeError(
            "Durable workspace branches require a branch authority provider with "
            "claim_scope='durable'."
        )


def _binding_authority_claim_lease(
    source: LocalWorkspace,
    authority: WorkspaceBranchAuthority,
) -> _BindingAuthorityClaimLease:
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        raise WorkspaceBranchFencedError("Workspace source identity is unavailable.")
    _retry_pending_binding_claim_releases(source_key)
    _require_durable_branch_support(source)
    provider = source._branch_authority_resolver
    if provider is None:  # pragma: no cover - helper invariant
        raise AssertionError("Durable workspace branch authority provider disappeared.")
    expected = WorkspaceBranchBindingAuthority(
        environment_name=authority.environment_name,
        binding_generation=authority.binding_generation,
        binding_identity=authority.binding_identity,
    )
    return _BindingAuthorityClaimLease(provider.claim(expected), expected, source_key)


def _binding_claim_release_key(
    claim: _BindingAuthorityClaimLease,
) -> tuple[tuple[object, ...], int]:
    return claim.source_key, id(claim)


def _retain_failed_binding_claim_release(claim: _BindingAuthorityClaimLease) -> None:
    _BINDING_CLAIM_RELEASES.retain(
        _binding_claim_release_key(claim),
        source_key=claim.source_key,
        payload=claim,
    )


def _retry_pending_binding_claim_releases(source_key: tuple[object, ...]) -> None:
    for key, claim in _BINDING_CLAIM_RELEASES.claim_pending(source_key):
        try:
            claim.release_after_settlement()
        except BaseException:
            _BINDING_CLAIM_RELEASES.release_claim(key)
            raise
        _BINDING_CLAIM_RELEASES.forget(key)


def _release_binding_authority_claim_lease(
    claim: _BindingAuthorityClaimLease,
    *,
    primary: BaseException | None = None,
) -> None:
    try:
        claim.release_owner()
    except BaseException as release_error:
        _retain_failed_binding_claim_release(claim)
        if primary is not None:
            _raise_primary_with_reconciliation_failure(primary, release_error)
        raise


async def _record_durable_creation_failure(
    store: WorkspaceBranchStore,
    current: _DurableBranchRecord,
    storage_key: str,
    failure: _DurableFailure,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> _DurableBranchRecord:
    binding_claim_lease.validate(current.authority)
    failed = current.model_copy(
        update={
            "state": WorkspaceBranchDurableState.FAILED,
            "failure": failure,
            "revision": current.revision + 1,
        }
    )
    return await _publish_or_reconcile_exact_record(
        store,
        session_id=current.authority.session_id,
        storage_key=storage_key,
        expected=current,
        updated=failed,
        expected_run_epoch=current.authority.expected_run_epoch,
    )


async def _complete_durable_creating_record(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    storage_key: str,
    capture_authority: _BaselineCaptureAuthority,
    capacity_lease: _BranchCapacityLease,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> tuple[_DurableBranchRecord, _CapturedBaseline | None]:
    """Complete the exact CREATING capture, including fresh-process replay."""

    if record.state is not WorkspaceBranchDurableState.CREATING:
        raise AssertionError("Durable creation completion requires a CREATING record.")
    store = source._branch_store
    if store is None:  # pragma: no cover - durable caller invariant
        raise AssertionError("Durable branch store disappeared during creation completion.")
    _validate_durable_source_root_identity(source, record)
    private_root = _validated_durable_private_root(source, record)
    captured: _CapturedBaseline | None = None
    capture_failure: BaseException | None = None

    def capture_before_open() -> None:
        nonlocal capture_failure, captured
        private_root_existed = _durable_private_root_exists(private_root)
        try:
            binding_claim_lease.validate(record.authority)
            if private_root_existed:
                captured = _captured_from_durable_record(source, record)
            else:
                captured = _capture_baseline_at_private_root(
                    source.root,
                    capture_authority,
                    record.branch_id,
                    capacity_lease,
                    private_root,
                    record.private_root_owner,
                    record.capture_attempt_id,
                    (record.root_device, record.root_inode),
                    lambda: binding_claim_lease.validate(record.authority),
                )
            try:
                binding_claim_lease.validate(record.authority)
            except BaseException as binding_failure:
                if not private_root_existed and captured is not None:
                    try:
                        _discard_captured_baseline(captured, capacity_lease)
                    except BaseException as cleanup_failure:
                        raise binding_failure from cleanup_failure
                    captured = None
                raise
        except BaseException as error:
            capture_failure = error
            raise

    opened = record.model_copy(
        update={"state": WorkspaceBranchDurableState.OPEN, "revision": record.revision + 1}
    )
    try:
        settled = await _publish_or_reconcile_exact_record(
            store,
            session_id=record.authority.session_id,
            storage_key=storage_key,
            expected=record,
            updated=opened,
            expected_run_epoch=record.authority.expected_run_epoch,
            commit_guard=capture_before_open,
        )
    except WorkspaceBranchOperationConflict as conflict:
        try:
            successor = await _load_durable_branch_record(
                store,
                session_id=record.authority.session_id,
                storage_key=storage_key,
            )
        except BaseException as reconciliation_error:
            _raise_primary_with_reconciliation_failure(conflict, reconciliation_error)
        if successor is None or _durable_branch_creation_identity(
            successor
        ) != _durable_branch_creation_identity(record):
            raise
        binding_claim_lease.validate(successor.authority)
        if successor.state in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
        }:
            reconstructed = await _await_owned_thread(
                _captured_from_durable_record,
                source,
                successor,
            )
            return successor, reconstructed
        if (
            successor.state is WorkspaceBranchDurableState.FAILED
            and successor.failure is not None
            and successor.failure.phase == "creation"
        ):
            return successor, None
        raise
    except _CreationConflict as conflict:
        if conflict is not capture_failure:
            raise
        try:
            evidence = _bounded_workspace_branch_evidence(
                source=record.source,
                baseline_revision=record.baseline_revision,
                branch_id=record.branch_id,
                outcome=WorkspaceBranchOutcomeStatus.CONFLICTED,
                max_bytes=record.limits.max_evidence_bytes,
                paths=tuple(item.path for item in conflict.conflicts),
                detail_code="workspace_branch_baseline_conflicted",
            )
            _validate_conflict_evidence_limit(
                conflict.conflicts,
                record.limits.max_evidence_bytes,
                evidence=evidence,
            )
        except WorkspaceBranchResourceExhaustedError as exhausted:
            failure = _DurableFailure(
                phase="creation",
                outcome="resource_exhausted",
                detail_code=exhausted.detail_code,
            )
        else:
            failure = _DurableFailure(
                phase="creation",
                outcome="conflicted",
                detail_code="workspace_branch_baseline_conflicted",
                conflicts=conflict.conflicts,
            )
        failed = await _record_durable_creation_failure(
            store,
            record,
            storage_key,
            failure,
            binding_claim_lease=binding_claim_lease,
        )
        return failed, None
    except _UnsupportedBranch as unsupported:
        if unsupported is not capture_failure:
            raise
        failed = await _record_durable_creation_failure(
            store,
            record,
            storage_key,
            _DurableFailure(
                phase="creation",
                outcome="unsupported",
                detail_code=unsupported.detail_code,
            ),
            binding_claim_lease=binding_claim_lease,
        )
        return failed, None
    except WorkspaceBranchResourceExhaustedError as exhausted:
        if exhausted is not capture_failure:
            raise
        failed = await _record_durable_creation_failure(
            store,
            record,
            storage_key,
            _DurableFailure(
                phase="creation",
                outcome="resource_exhausted",
                detail_code=exhausted.detail_code,
            ),
            binding_claim_lease=binding_claim_lease,
        )
        return failed, None
    except OSError as error:
        if error is not capture_failure:
            raise
        detail_code = (
            "branch_storage_exhausted"
            if error.errno in _RESOURCE_ERRNOS
            else "durable_workspace_branch_creation_failed"
        )
        outcome = "resource_exhausted" if error.errno in _RESOURCE_ERRNOS else "failed"
        failed = await _record_durable_creation_failure(
            store,
            record,
            storage_key,
            _DurableFailure(
                phase="creation",
                outcome=outcome,
                detail_code=detail_code,
            ),
            binding_claim_lease=binding_claim_lease,
        )
        return failed, None
    if captured is None:
        captured = await _await_owned_thread(_captured_from_durable_record, source, settled)
    return settled, captured


def _validate_durable_creation_authority(
    source: LocalWorkspace,
    request: WorkspaceBranchRequest,
    record: _DurableBranchRecord,
) -> None:
    authority = request.authority
    if authority is None or request.branch_id is None or request.idempotency_key is None:
        raise AssertionError("Durable branch creation authority disappeared.")
    if (
        record.branch_id != request.branch_id
        or record.creation_idempotency_key != request.idempotency_key
        or record.source != request.baseline.identity
        or record.source_root != str(source.root)
        or record.baseline_revision != request.baseline.revision
        or record.authority != authority
        or record.limits != request.limits
    ):
        raise WorkspaceBranchOperationConflict(
            "Durable workspace branch identity was reused with different authority."
        )


def _durable_creation_result(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    captured: _CapturedBaseline,
    source_key: tuple[object, ...],
    capacity_lease: _BranchCapacityLease,
) -> WorkspaceBranchCreationResult:
    branch_store = source._branch_store
    if branch_store is None:  # pragma: no cover - durable caller invariant
        raise AssertionError("Durable branch store disappeared during reconstruction.")
    branch = LocalWorkspaceBranch(
        source=source,
        branch_id=record.branch_id,
        request=WorkspaceBranchRequest(
            baseline=_workspace_revision_from_record(record, captured),
            limits=record.limits,
            branch_id=record.branch_id,
            idempotency_key=record.creation_idempotency_key,
            authority=record.authority,
        ),
        captured=captured,
        source_key=source_key,
        capacity_lease=capacity_lease,
        durable=_DurableBranchContext(
            store=branch_store,
            record=record,
            storage_key=_durable_branch_storage_key(record.branch_id),
        ),
    )
    capacity_lease.activate()
    return WorkspaceBranchCreationResult(
        status=WorkspaceBranchOutcomeStatus.CREATED,
        branch=branch,
        evidence=_bounded_workspace_branch_evidence(
            source=record.source,
            baseline_revision=record.baseline_revision,
            branch_id=record.branch_id,
            outcome=WorkspaceBranchOutcomeStatus.CREATED,
            max_bytes=record.limits.max_evidence_bytes,
            detail_code="durable_workspace_branch_created",
        ),
    )


def _durable_failed_creation_result(
    record: _DurableBranchRecord,
) -> WorkspaceBranchCreationResult:
    failure = record.failure
    if (
        record.state is not WorkspaceBranchDurableState.FAILED
        or failure is None
        or failure.phase != "creation"
    ):
        raise WorkspaceBranchFencedError("Durable creation failure evidence is missing.")
    outcome = WorkspaceBranchOutcomeStatus(failure.outcome)
    return WorkspaceBranchCreationResult(
        status=outcome,
        branch=None,
        evidence=_durable_state_evidence(
            record,
            outcome=outcome,
            paths=tuple(conflict.path for conflict in failure.conflicts),
            detail_code=failure.detail_code,
        ),
        conflicts=failure.conflicts,
    )


def _workspace_revision_from_record(
    record: _DurableBranchRecord,
    captured: _CapturedBaseline,
):
    from cayu.workspaces.revisions import WorkspacePathRevision, WorkspaceRevisionObservation

    paths = tuple(
        WorkspacePathRevision(
            path=path,
            kind="file",
            present=True,
            content_sha256=identity.sha256,
        )
        for path, identity in sorted(captured.files.items())
    )
    return WorkspaceRevisionObservation(
        identity=record.source,
        status=WorkspaceRevisionObservationStatus.SUPPORTED,
        revision=record.baseline_revision,
        paths=paths,
        total_paths=len(paths),
    )


async def _create_durable_local_workspace_branch(
    source: LocalWorkspace,
    request: WorkspaceBranchRequest,
    source_key: tuple[object, ...],
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> WorkspaceBranchCreationResult:
    store = source._branch_store
    if store is None:
        raise RuntimeError("Durable workspace branches require LocalWorkspace branch_store.")
    branch_id = request.branch_id
    create_key = request.idempotency_key
    authority = request.authority
    if branch_id is None or create_key is None or authority is None:
        raise AssertionError("Durable branch authority is incomplete.")
    binding_claim_lease.validate(authority)
    storage_key = _durable_branch_storage_key(branch_id)
    current = await _load_durable_branch_record(
        store,
        session_id=authority.session_id,
        storage_key=storage_key,
    )
    if current is None:
        root_info = source.root.stat()
        now = datetime.now(UTC)
        baseline_revision = request.baseline.revision
        if baseline_revision is None:  # pragma: no cover - request invariant
            raise AssertionError("Durable workspace branch baseline revision disappeared.")
        candidate = _DurableBranchRecord(
            branch_id=branch_id,
            creation_idempotency_key=create_key,
            state=WorkspaceBranchDurableState.CREATING,
            source=request.baseline.identity,
            source_root=str(source.root),
            baseline_revision=baseline_revision,
            authority=authority,
            limits=request.limits,
            private_root=str(_durable_private_root(source.root, branch_id, authority)),
            private_root_owner=uuid.uuid4().hex,
            capture_attempt_id=uuid.uuid4().hex,
            root_device=root_info.st_dev,
            root_inode=root_info.st_ino,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(milliseconds=request.limits.lifetime_ms)).isoformat(),
            revision=0,
        )
        try:
            current = await _publish_or_reconcile_exact_record(
                store,
                session_id=authority.session_id,
                storage_key=storage_key,
                expected=None,
                updated=candidate,
                expected_run_epoch=authority.expected_run_epoch,
            )
        except WorkspaceBranchOperationConflict as conflict:
            try:
                winner = await _load_durable_branch_record(
                    store,
                    session_id=authority.session_id,
                    storage_key=storage_key,
                )
            except BaseException as reconciliation_error:
                _raise_primary_with_reconciliation_failure(conflict, reconciliation_error)
            if winner is None:
                raise
            _validate_durable_creation_authority(source, request, winner)
            current = winner
    if current is None:  # pragma: no cover - store publication invariant
        raise AssertionError("Durable workspace branch creation produced no record.")
    _validate_durable_creation_authority(source, request, current)
    binding_claim_lease.validate(current.authority)
    if current.state is WorkspaceBranchDurableState.FAILED:
        if current.failure is not None and current.failure.phase == "creation":
            return _durable_failed_creation_result(current)
        raise WorkspaceBranchClosedError("Durable workspace branch publication already failed.")
    if current.state not in {
        WorkspaceBranchDurableState.CREATING,
        WorkspaceBranchDurableState.OPEN,
        WorkspaceBranchDurableState.CONFLICTED,
    }:
        raise WorkspaceBranchClosedError(
            f"Durable workspace branch is already {current.state.value}."
        )
    await _await_owned_thread(_retry_pending_branch_cleanups, source_key)
    capacity_lease = _admit_branch(
        source_key,
        request.limits.max_active_branches,
        identity=_durable_branch_capacity_identity(current),
    )
    if capacity_lease is None:
        return _creation_result_without_branch(
            request,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch_id=branch_id,
            detail_code="active_branch_limit_exceeded",
        )
    try:
        if current.state is WorkspaceBranchDurableState.CREATING:
            current, captured = await _complete_durable_creating_record(
                source,
                current,
                storage_key,
                _capture_authority_from_request(request),
                capacity_lease,
                binding_claim_lease=binding_claim_lease,
            )
            if current.state is WorkspaceBranchDurableState.FAILED:
                capacity_lease.release_owner()
                return _durable_failed_creation_result(current)
            if captured is None:  # pragma: no cover - completion invariant
                raise AssertionError("Durable branch creation lost its captured baseline.")
        else:
            captured = await _await_owned_thread(_captured_from_durable_record, source, current)
        capacity_lease.adopt_recovered_generation()
        return _durable_creation_result(
            source,
            current,
            captured,
            source_key,
            capacity_lease,
        )
    except BaseException:
        capacity_lease.release_owner()
        raise


def _validate_recovery_authority(
    source: LocalWorkspace,
    request: WorkspaceBranchRecoveryRequest,
    record: _DurableBranchRecord,
) -> None:
    if (
        request.branch_id != record.branch_id
        or request.session_id != record.authority.session_id
        or request.expected_run_epoch != record.authority.expected_run_epoch
        or request.binding_generation != record.authority.binding_generation
        or request.binding_identity != record.authority.binding_identity
        or source.id != record.source.workspace_id
        or str(source.root) != record.source_root
    ):
        raise WorkspaceBranchOperationConflict(
            "Workspace branch recovery authority does not match durable evidence."
        )


def _durable_state_evidence(
    record: _DurableBranchRecord,
    *,
    outcome: WorkspaceBranchOutcomeStatus,
    paths: Sequence[str] = (),
    change_set_digest: str | None = None,
    detail_code: str,
) -> WorkspaceBranchEvidence:
    return _bounded_workspace_branch_evidence(
        source=record.source,
        baseline_revision=record.baseline_revision,
        branch_id=record.branch_id,
        outcome=outcome,
        change_set_digest=change_set_digest,
        paths=paths,
        detail_code=detail_code,
        max_bytes=record.limits.max_evidence_bytes,
    )


def _durable_record_is_expired(record: _DurableBranchRecord) -> bool:
    try:
        expires_at = _parse_durable_branch_timestamp(record.expires_at, "expires_at")
    except ValueError as error:  # pragma: no cover - validated record invariant
        raise WorkspaceBranchFencedError("Durable branch expiry evidence is invalid.") from error
    return datetime.now(UTC) >= expires_at


def _require_durable_terminal_expiry_eligible(record: _DurableBranchRecord) -> None:
    if record.state is WorkspaceBranchDurableState.EXPIRED:
        _require_durable_expiry_eligible(record)


def _require_durable_expiry_eligible(record: _DurableBranchRecord) -> None:
    if not _durable_record_is_expired(record):
        raise WorkspaceBranchFencedError(
            "Durable branch expiry terminal evidence precedes its deadline."
        )


def _durable_publication_staging_name(
    publication: _DurablePublication,
    path: str,
) -> str:
    name = path.rsplit("/", 1)[-1]
    digest = hashlib.sha256(f"{publication.staging_owner}\0{path}".encode()).hexdigest()
    return f".{name}.cayu-{digest[:48]}"


async def _record_durable_branch_ambiguity(
    store: WorkspaceBranchStore,
    record: _DurableBranchRecord,
    storage_key: str,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> _DurableBranchRecord:
    binding_claim_lease.validate(record.authority)
    if record.state is WorkspaceBranchDurableState.AMBIGUOUS:
        return record
    ambiguous = record.model_copy(
        update={
            "state": WorkspaceBranchDurableState.AMBIGUOUS,
            "revision": record.revision + 1,
        }
    )
    try:
        return await _publish_or_reconcile_exact_record(
            store,
            session_id=record.authority.session_id,
            storage_key=storage_key,
            expected=record,
            updated=ambiguous,
            expected_run_epoch=record.authority.expected_run_epoch,
        )
    except WorkspaceBranchOperationConflict:
        settled = await _load_durable_branch_record(
            store,
            session_id=record.authority.session_id,
            storage_key=storage_key,
        )
        if settled is not None and settled.state is WorkspaceBranchDurableState.AMBIGUOUS:
            return settled
        raise


def _durable_publication_result(
    record: _DurableBranchRecord,
) -> WorkspaceBranchPublicationResult:
    publication = record.publication
    if publication is None:
        raise WorkspaceBranchFencedError("Durable committed publication evidence is missing.")
    paths = tuple(change.path for change in publication.change_set.changes)
    return WorkspaceBranchPublicationResult(
        status=WorkspaceBranchOutcomeStatus.COMMITTED,
        evidence=_durable_state_evidence(
            record,
            outcome=WorkspaceBranchOutcomeStatus.COMMITTED,
            paths=paths,
            change_set_digest=publication.change_set.digest,
            detail_code="durable_workspace_branch_committed",
        ),
    )


def _durable_failed_publication_result(
    record: _DurableBranchRecord,
) -> WorkspaceBranchPublicationResult:
    failure = record.failure
    publication = record.publication
    if (
        record.state is not WorkspaceBranchDurableState.FAILED
        or failure is None
        or failure.phase != "publication"
        or publication is None
    ):
        raise WorkspaceBranchFencedError("Durable publication failure evidence is missing.")
    paths = tuple(change.path for change in publication.change_set.changes)
    return WorkspaceBranchPublicationResult(
        status=WorkspaceBranchOutcomeStatus.FAILED,
        evidence=_durable_state_evidence(
            record,
            outcome=WorkspaceBranchOutcomeStatus.FAILED,
            paths=paths,
            change_set_digest=publication.change_set.digest,
            detail_code=failure.detail_code,
        ),
    )


def _durable_rollback_result(record: _DurableBranchRecord) -> WorkspaceBranchRollbackResult:
    _require_durable_terminal_expiry_eligible(record)
    rollback = record.rollback
    if rollback is None:
        raise WorkspaceBranchFencedError("Durable rollback evidence is missing.")
    status = (
        WorkspaceBranchOutcomeStatus.EXPIRED
        if record.state is WorkspaceBranchDurableState.EXPIRED
        else WorkspaceBranchOutcomeStatus.ROLLED_BACK
    )
    return WorkspaceBranchRollbackResult(
        status=status,
        evidence=_durable_state_evidence(
            record,
            outcome=status,
            paths=rollback.paths,
            detail_code=(
                "durable_workspace_branch_expired"
                if status is WorkspaceBranchOutcomeStatus.EXPIRED
                else "durable_workspace_branch_rolled_back"
            ),
        ),
    )


def _durable_recovery_cleanup_key(
    record: _DurableBranchRecord,
) -> tuple[Path, str]:
    return Path(record.private_root), record.private_root_owner


def _release_durable_recovery_cleanup(record: _DurableBranchRecord) -> None:
    key = _durable_recovery_cleanup_key(record)
    retained = _DURABLE_RECOVERY_CLEANUPS.payload(key)
    if retained is not None:
        retained.binding_claim_lease.release_after_settlement()
        _DURABLE_RECOVERY_CLEANUPS.forget(key)


def _retain_durable_recovery_cleanup(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> None:
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        raise WorkspaceBranchFencedError("Workspace source identity is unavailable.")
    key = _durable_recovery_cleanup_key(record)
    existing = _DURABLE_RECOVERY_CLEANUPS.payload(key)
    if existing is not None:
        if existing.source.resource_key != source_key or existing.record != record:
            raise RuntimeError("Retained recovery cleanup identity changed while owned.")
        return
    retained = _DURABLE_RECOVERY_CLEANUPS.retain(
        key,
        source_key=source_key,
        payload=_RetainedDurableRecoveryCleanup(
            source=source,
            record=record,
            binding_claim_lease=binding_claim_lease,
        ),
    )
    if retained:
        binding_claim_lease.retain_for_settlement()


async def _settle_durable_recovery_private_tree(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> None:
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        raise WorkspaceBranchFencedError("Workspace source identity is unavailable.")
    capacity_owner = _mark_matching_branch_owner_terminal(source_key, record)
    binding_claim_lease.validate(record.authority)
    try:
        await _await_owned_thread(_remove_durable_private_tree, source, record)
    except BaseException as cleanup_error:
        try:
            _retain_durable_recovery_cleanup(source, record, binding_claim_lease)
        except BaseException as retention_error:
            raise cleanup_error from retention_error
        raise
    if capacity_owner is not None:
        _release_branch_attachment(capacity_owner, terminal=True)
    _release_durable_recovery_cleanup(record)


async def _durable_terminal_recovery_result(
    source: LocalWorkspace,
    record: _DurableBranchRecord,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> WorkspaceBranchRecoveryResult:
    if record.state is WorkspaceBranchDurableState.COMMITTED:
        publication = _durable_publication_result(record)
        result = WorkspaceBranchRecoveryResult(
            state=record.state,
            evidence=publication.evidence,
            publication=publication,
        )
    elif record.state in {
        WorkspaceBranchDurableState.ROLLED_BACK,
        WorkspaceBranchDurableState.EXPIRED,
    }:
        rollback = _durable_rollback_result(record)
        result = WorkspaceBranchRecoveryResult(
            state=record.state,
            evidence=rollback.evidence,
            rollback=rollback,
        )
    else:
        raise WorkspaceBranchFencedError("Durable branch is not terminal.")
    await _settle_durable_recovery_private_tree(
        source,
        record,
        binding_claim_lease=binding_claim_lease,
    )
    return result


async def recover_local_workspace_branch(
    source: LocalWorkspace,
    request: WorkspaceBranchRecoveryRequest,
) -> WorkspaceBranchRecoveryResult:
    from cayu.workspaces.local import LocalWorkspace

    if type(source) is not LocalWorkspace:
        raise TypeError("Durable local branch recovery requires an exact LocalWorkspace.")
    if type(request) is not WorkspaceBranchRecoveryRequest:
        raise TypeError("Workspace branch recovery request is invalid.")
    store = source._branch_store
    if store is None:
        raise RuntimeError("Durable workspace branch recovery requires branch_store.")
    _require_durable_branch_support(source)
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        raise WorkspaceBranchFencedError("Workspace source identity is unavailable.")
    await _retry_pending_durable_terminal_settlements(source_key)
    record = await _load_durable_branch_record(
        store,
        session_id=request.session_id,
        storage_key=_durable_branch_storage_key(request.branch_id),
    )
    if record is None:
        raise KeyError(f"Durable workspace branch not found: {request.branch_id}")
    _validate_recovery_authority(source, request, record)
    binding_claim_lease = _binding_authority_claim_lease(source, record.authority)
    try:
        result = await _recover_local_workspace_branch_claimed(
            source,
            request,
            binding_claim_lease=binding_claim_lease,
        )
    except BaseException as primary:
        _release_binding_authority_claim_lease(binding_claim_lease, primary=primary)
        raise
    _release_binding_authority_claim_lease(binding_claim_lease)
    return result


async def _recover_local_workspace_branch_claimed(
    source: LocalWorkspace,
    request: WorkspaceBranchRecoveryRequest,
    *,
    binding_claim_lease: _BindingAuthorityClaimLease,
) -> WorkspaceBranchRecoveryResult:
    from cayu.workspaces.local import LocalWorkspace

    if type(source) is not LocalWorkspace:
        raise TypeError("Durable local branch recovery requires an exact LocalWorkspace.")
    if type(request) is not WorkspaceBranchRecoveryRequest:
        raise TypeError("Workspace branch recovery request is invalid.")
    store = source._branch_store
    if store is None:
        raise RuntimeError("Durable workspace branch recovery requires branch_store.")
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        raise WorkspaceBranchFencedError("Workspace source identity is unavailable.")
    storage_key = _durable_branch_storage_key(request.branch_id)
    record = await _load_durable_branch_record(
        store,
        session_id=request.session_id,
        storage_key=storage_key,
    )
    if record is None:
        raise KeyError(f"Durable workspace branch not found: {request.branch_id}")
    _validate_recovery_authority(source, request, record)
    binding_claim_lease.validate(record.authority)
    _validated_durable_private_root(source, record)

    capacity_lease: _BranchCapacityLease | None = None
    creating_captured: _CapturedBaseline | None = None
    if record.state is WorkspaceBranchDurableState.CREATING:
        await _await_owned_thread(_retry_pending_branch_cleanups, source_key)
        capacity_lease = _admit_branch(
            source_key,
            record.limits.max_active_branches,
            identity=_durable_branch_capacity_identity(record),
        )
        if capacity_lease is None:
            raise WorkspaceBranchResourceExhaustedError("active_branch_limit_exceeded")
        try:
            record, creating_captured = await _complete_durable_creating_record(
                source,
                record,
                storage_key,
                _capture_authority_from_record(record),
                capacity_lease,
                binding_claim_lease=binding_claim_lease,
            )
        except BaseException:
            capacity_lease.release_owner()
            raise
        if record.state is WorkspaceBranchDurableState.FAILED:
            capacity_lease.release_owner()
            capacity_lease = None

    if record.state is WorkspaceBranchDurableState.ROLLBACK_INTENT:
        rollback = record.rollback
        if rollback is None:
            raise WorkspaceBranchFencedError("Durable rollback intent is incomplete.")
        if rollback.reason == "expired":
            _require_durable_expiry_eligible(record)
        terminal_state = (
            WorkspaceBranchDurableState.EXPIRED
            if rollback.reason == "expired"
            else WorkspaceBranchDurableState.ROLLED_BACK
        )
        terminal = record.model_copy(
            update={"state": terminal_state, "revision": record.revision + 1}
        )
        record = await _publish_or_reconcile_exact_record(
            store,
            session_id=record.authority.session_id,
            storage_key=storage_key,
            expected=record,
            updated=terminal,
            expected_run_epoch=record.authority.expected_run_epoch,
        )

    if record.state in {
        WorkspaceBranchDurableState.OPEN,
        WorkspaceBranchDurableState.CONFLICTED,
        WorkspaceBranchDurableState.PUBLICATION_INTENT,
        WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
    }:
        if capacity_lease is None:
            capacity_lease = _admit_branch(
                source_key,
                record.limits.max_active_branches,
                identity=_durable_branch_capacity_identity(record),
            )
            if capacity_lease is None:
                raise WorkspaceBranchResourceExhaustedError("active_branch_limit_exceeded")
        try:
            try:
                captured = creating_captured
                if captured is None:
                    captured = await _await_owned_thread(
                        _captured_from_durable_record,
                        source,
                        record,
                    )
            except WorkspaceBranchFencedError:
                capacity_lease.release_owner()
                settled = await _load_durable_branch_record(
                    store,
                    session_id=record.authority.session_id,
                    storage_key=storage_key,
                )
                if settled is not None and settled.state in {
                    WorkspaceBranchDurableState.COMMITTED,
                    WorkspaceBranchDurableState.ROLLED_BACK,
                    WorkspaceBranchDurableState.EXPIRED,
                }:
                    return await _durable_terminal_recovery_result(
                        source,
                        settled,
                        binding_claim_lease=binding_claim_lease,
                    )
                ambiguous = await _record_durable_branch_ambiguity(
                    store,
                    record,
                    storage_key,
                    binding_claim_lease=binding_claim_lease,
                )
                _mark_matching_branch_owner_terminal(source_key, ambiguous)
                return WorkspaceBranchRecoveryResult(
                    state=WorkspaceBranchDurableState.AMBIGUOUS,
                    evidence=_durable_state_evidence(
                        ambiguous,
                        outcome=WorkspaceBranchOutcomeStatus.AMBIGUOUS,
                        change_set_digest=(
                            None
                            if ambiguous.publication is None
                            else ambiguous.publication.change_set.digest
                        ),
                        detail_code="durable_workspace_branch_private_state_ambiguous",
                    ),
                )
            try:
                capacity_lease.adopt_recovered_generation()
            except (WorkspaceBranchClosedError, WorkspaceBranchFencedError):
                capacity_lease.release_owner()
                settled = await _load_durable_branch_record(
                    store,
                    session_id=record.authority.session_id,
                    storage_key=storage_key,
                )
                if settled is not None and settled != record:
                    return await _recover_local_workspace_branch_claimed(
                        source,
                        request,
                        binding_claim_lease=binding_claim_lease,
                    )
                raise
            creation = _durable_creation_result(
                source,
                record,
                captured,
                source_key,
                capacity_lease,
            )
        except BaseException:
            capacity_lease.release_owner()
            raise
        branch = creation.branch
        if not isinstance(branch, LocalWorkspaceBranch):
            raise AssertionError("Durable branch reconstruction returned no branch.")
        if record.state in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
        } and _durable_record_is_expired(record):
            rollback = await branch._rollback_durable_claimed(
                WorkspaceBranchRollbackRequest(
                    branch_id=record.branch_id,
                    idempotency_key=(
                        "expire:" + hashlib.sha256(record.expires_at.encode()).hexdigest()
                    ),
                    expected_run_epoch=record.authority.expected_run_epoch,
                    binding_generation=record.authority.binding_generation,
                    reason="expired",
                ),
                binding_claim_lease=binding_claim_lease,
            )
            return WorkspaceBranchRecoveryResult(
                state=WorkspaceBranchDurableState.EXPIRED,
                evidence=rollback.evidence,
                rollback=rollback,
            )
        if record.state in {
            WorkspaceBranchDurableState.PUBLICATION_INTENT,
            WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
        }:
            publication = record.publication
            if publication is None:
                raise WorkspaceBranchFencedError("Durable publication intent is missing.")
            result = await branch._publish_durable_claimed(
                WorkspaceBranchPublicationRequest(
                    branch_id=record.branch_id,
                    baseline_revision=record.baseline_revision,
                    change_set_digest=publication.change_set.digest,
                    idempotency_key=publication.idempotency_key,
                    expected_run_epoch=record.authority.expected_run_epoch,
                    binding_generation=record.authority.binding_generation,
                ),
                binding_claim_lease=binding_claim_lease,
            )
            settled = await _load_durable_branch_record(
                store,
                session_id=record.authority.session_id,
                storage_key=storage_key,
            )
            if settled is None:
                raise WorkspaceBranchFencedError(
                    "Durable workspace branch disappeared during recovery."
                )
            if settled.state is WorkspaceBranchDurableState.COMMITTED:
                return WorkspaceBranchRecoveryResult(
                    state=settled.state,
                    evidence=result.evidence,
                    publication=result,
                )
            if settled.state is WorkspaceBranchDurableState.CONFLICTED:
                return WorkspaceBranchRecoveryResult(
                    state=settled.state,
                    evidence=result.evidence,
                    branch=branch,
                )
            return WorkspaceBranchRecoveryResult(
                state=settled.state,
                evidence=result.evidence,
            )
        return WorkspaceBranchRecoveryResult(
            state=record.state,
            evidence=_durable_state_evidence(
                record,
                outcome=WorkspaceBranchOutcomeStatus.CREATED,
                detail_code="durable_workspace_branch_recovered",
            ),
            branch=branch,
        )

    if record.state in {
        WorkspaceBranchDurableState.COMMITTED,
        WorkspaceBranchDurableState.ROLLED_BACK,
        WorkspaceBranchDurableState.EXPIRED,
    }:
        return await _durable_terminal_recovery_result(
            source,
            record,
            binding_claim_lease=binding_claim_lease,
        )
    if record.state is WorkspaceBranchDurableState.AMBIGUOUS:
        _mark_matching_branch_owner_terminal(source_key, record)
        outcome = WorkspaceBranchOutcomeStatus.AMBIGUOUS
        detail_code = "durable_workspace_branch_ambiguous"
        settle_failed_private_tree = False
    elif record.state is WorkspaceBranchDurableState.FAILED and record.failure is not None:
        outcome = WorkspaceBranchOutcomeStatus(record.failure.outcome)
        detail_code = record.failure.detail_code
        settle_failed_private_tree = True
    else:
        outcome = WorkspaceBranchOutcomeStatus.FAILED
        detail_code = f"durable_workspace_branch_{record.state.value}"
        settle_failed_private_tree = False
    result = WorkspaceBranchRecoveryResult(
        state=record.state,
        evidence=_durable_state_evidence(
            record,
            outcome=outcome,
            change_set_digest=(
                record.publication.change_set.digest
                if record.failure is not None
                and record.failure.phase == "publication"
                and record.publication is not None
                else None
            ),
            detail_code=detail_code,
        ),
    )
    if settle_failed_private_tree:
        await _settle_durable_recovery_private_tree(
            source,
            record,
            binding_claim_lease=binding_claim_lease,
        )
    return result


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
    envelope_evidence_limit_violation = _workspace_branch_fixed_authority_evidence_limit_violation(
        source=envelope.source,
        baseline_revision=envelope.baseline_revision,
        branch_id=envelope.branch_id or "wsb_" + "0" * 32,
        limits=envelope.limits,
        created_detail_code="workspace_branch_created",
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
    if copied.authority is not None:
        _require_durable_branch_support(source)
    source_key = source.resource_key
    if type(source_key) is not tuple or not source_key:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            detail_code="source_identity_unavailable",
        )
    await _retry_pending_durable_terminal_settlements(source_key)
    if copied.authority is not None:
        binding_claim_lease = _binding_authority_claim_lease(source, copied.authority)
        try:
            result = await _create_durable_local_workspace_branch(
                source,
                copied,
                source_key,
                binding_claim_lease=binding_claim_lease,
            )
        except BaseException as primary:
            _release_binding_authority_claim_lease(binding_claim_lease, primary=primary)
            raise
        _release_binding_authority_claim_lease(binding_claim_lease)
        return result
    await _await_owned_thread(_retry_pending_branch_cleanups, source_key)
    if local_workspace_source_is_fenced(source.root):
        raise WorkspaceBranchFencedError(
            "Local workspace is fenced after an incomplete branch publication rollback."
        )
    branch_id = f"wsb_{uuid.uuid4().hex}"
    capacity_lease = _admit_branch(
        source_key,
        copied.limits.max_active_branches,
        identity=("ephemeral", branch_id),
    )
    if capacity_lease is None:
        return _creation_result_without_branch(
            copied,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            detail_code="active_branch_limit_exceeded",
        )
    captured: _CapturedBaseline | None = None
    ownership_transferred = False
    try:
        captured_result = await _await_owned_thread(
            _capture_baseline,
            source.root,
            _capture_authority_from_request(copied),
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
        capacity_lease.activate()
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
    return _workspace_branch_fixed_authority_evidence_limit_violation(
        source=request.baseline.identity,
        baseline_revision=baseline_revision,
        branch_id=request.branch_id or "wsb_" + "0" * 32,
        limits=request.limits,
        created_detail_code="workspace_branch_created",
    )


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


def _admit_branch(
    source_key: tuple[object, ...],
    limit: int,
    *,
    identity: Hashable,
) -> _BranchCapacityLease | None:
    with _SOURCE_MANAGER_LOCK:
        active = _ACTIVE_BRANCHES.setdefault(source_key, {})
        owner = active.get(identity)
        if owner is None or owner.released:
            if len(active) >= limit:
                return None
            owner = _BranchCapacityOwner(source_key, identity)
            active[identity] = owner
        owner.attachments += 1
        generation = owner.attachment_generation()
    return _BranchCapacityLease(owner, generation)


def _release_branch_attachment(
    owner: _BranchCapacityOwner,
    *,
    terminal: bool,
) -> None:
    with _SOURCE_MANAGER_LOCK:
        if owner.released:
            return
        active = _ACTIVE_BRANCHES.get(owner.source_key)
        if active is None or active.get(owner.identity) is not owner:
            owner.released = True
            return
        owner.attachments = max(0, owner.attachments - 1)
        if terminal or owner.attachments == 0:
            owner.released = True
            active.pop(owner.identity, None)
            if not active:
                _ACTIVE_BRANCHES.pop(owner.source_key, None)


def _capture_baseline(
    root: Path,
    authority: _BaselineCaptureAuthority,
    branch_id: str,
    capacity_lease: _BranchCapacityLease,
    *,
    private_root_path: Path | None = None,
    private_root_owner: str | None = None,
    expected_root_identity: tuple[int, int] | None = None,
    source_lock_held: bool = False,
) -> _CapturedBaseline:
    private_root: Path | None = None
    private_root_owner_written = False
    captured: _CapturedBaseline | None = None
    primary_error: BaseException | None = None
    try:
        source_lock = (
            nullcontext() if source_lock_held else workspace_source_lock(root, exclusive=True)
        )
        with source_lock:
            before = root.stat()
            if (
                expected_root_identity is not None
                and (
                    before.st_dev,
                    before.st_ino,
                )
                != expected_root_identity
            ):
                raise WorkspaceBranchFencedError("Local workspace root identity changed.")
            if not stat.S_ISDIR(before.st_mode):
                raise _UnsupportedBranch("source_root_is_not_directory")
            private_parent = root.parent
            if private_parent == root:
                raise _UnsupportedBranch("source_root_has_no_private_sibling")
            if private_root_path is None:
                private_root = Path(
                    tempfile.mkdtemp(
                        prefix=f".cayu-{branch_id}-",
                        dir=private_parent,
                    )
                )
            else:
                if private_root_path.parent != private_parent:
                    raise _UnsupportedBranch("private_storage_parent_mismatch")
                private_root_path.mkdir(mode=0o700)
                private_root = private_root_path
            os.chmod(private_root, 0o700)
            if private_root_owner is not None:
                _write_private_root_owner(private_root, private_root_owner)
                private_root_owner_written = True
            private_info = private_root.stat()
            if private_info.st_dev != before.st_dev:
                raise _UnsupportedBranch("private_storage_filesystem_mismatch")
            baseline_root = private_root / "baseline"
            overlay_root = private_root / "overlay"
            baseline_root.mkdir(mode=0o700)
            overlay_root.mkdir(mode=0o700)
            source_semantics = _directory_lookup_semantics(root)
            private_semantics = tuple(
                _directory_lookup_semantics(private_directory)
                for private_directory in (baseline_root, overlay_root)
            )
            if source_semantics is _DirectoryLookupSemantics.UNKNOWN or any(
                semantics is _DirectoryLookupSemantics.UNKNOWN for semantics in private_semantics
            ):
                raise _UnsupportedBranch("path_semantics_unavailable")
            if any(semantics is not source_semantics for semantics in private_semantics):
                raise _UnsupportedBranch("private_storage_path_semantics_mismatch")
            files, directories = _copy_regular_tree(
                root,
                baseline_root,
                authority.limits,
                source_semantics=source_semantics,
            )
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
            conflicts = _baseline_conflicts(authority, files, branch_id=branch_id)
            if actual_revision != authority.revision or conflicts:
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
                private_root_owner=private_root_owner,
                baseline_root=baseline_root,
                overlay_root=overlay_root,
                files=files,
                directories=frozenset(directories),
                root_identity=(before.st_dev, before.st_ino),
                path_semantics=source_semantics,
            )
    except BaseException as error:
        primary_error = error
    if primary_error is not None:
        if private_root is not None:
            try:
                if private_root_owner is not None and not private_root_owner_written:
                    private_root.rmdir()
                    capacity_lease.release_after_cleanup()
                else:
                    _discard_private_tree(
                        private_root,
                        capacity_lease,
                        private_root_owner=private_root_owner,
                    )
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
        raise primary_error
    if captured is None:  # pragma: no cover - successful capture invariant
        raise AssertionError("Workspace branch capture produced no baseline.")
    return captured


def _capture_baseline_at_private_root(
    root: Path,
    authority: _BaselineCaptureAuthority,
    branch_id: str,
    capacity_lease: _BranchCapacityLease,
    private_root: Path,
    private_root_owner: str,
    capture_attempt_id: str,
    expected_root_identity: tuple[int, int],
    publish_guard: Callable[[], None],
) -> _CapturedBaseline:
    staging_root = _capture_staging_root(private_root, capture_attempt_id)
    _discard_owned_capture_staging(staging_root, private_root_owner)
    captured: _CapturedBaseline | None = None
    try:
        captured = _capture_baseline(
            root,
            authority,
            branch_id,
            capacity_lease,
            private_root_path=staging_root,
            private_root_owner=private_root_owner,
            expected_root_identity=expected_root_identity,
        )
        publish_guard()
        parent_fd = os.open(private_root.parent, _PRIVATE_DIRECTORY_OPEN_FLAGS)
        try:
            try:
                _rename_private_root_no_replace(
                    staging_root,
                    private_root,
                    parent_fd=parent_fd,
                )
            except FileExistsError as error:
                raise WorkspaceBranchFencedError(
                    "Durable private branch location became occupied."
                ) from error
            captured = replace(
                captured,
                private_root=private_root,
                baseline_root=private_root / "baseline",
                overlay_root=private_root / "overlay",
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return captured
    except BaseException as primary_error:
        if captured is not None:
            try:
                _discard_captured_baseline(captured, capacity_lease)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
        raise


def _copy_regular_tree(
    source_root: Path,
    baseline_root: Path,
    limits: WorkspaceBranchLimits,
    *,
    source_semantics: _DirectoryLookupSemantics,
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
                            directory_semantics = _directory_lookup_semantics(source_root / path)
                            if directory_semantics is _DirectoryLookupSemantics.UNKNOWN:
                                raise _UnsupportedBranch("path_semantics_unavailable")
                            if directory_semantics is not source_semantics:
                                raise _UnsupportedBranch("source_contains_mixed_path_semantics")
                            directories.add(path)
                            baseline_directory = baseline_root / path
                            baseline_directory.mkdir(mode=0o700, parents=False)
                            baseline_semantics = _directory_lookup_semantics(baseline_directory)
                            if baseline_semantics is _DirectoryLookupSemantics.UNKNOWN:
                                raise _UnsupportedBranch("path_semantics_unavailable")
                            if baseline_semantics is not source_semantics:
                                raise _UnsupportedBranch("private_storage_path_semantics_mismatch")
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
    authority: _BaselineCaptureAuthority,
    actual: dict[str, _FileIdentity],
    *,
    branch_id: str,
) -> tuple[WorkspaceBranchConflict, ...]:
    if authority.expected_paths is None:
        return ()
    expected = {entry.path: entry for entry in authority.expected_paths}
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
            source=authority.source,
            outcome=WorkspaceBranchOutcomeStatus.CONFLICTED,
            baseline_revision=authority.revision,
            branch_id=branch_id,
            change_set_digest=None,
            affected_path_count=len(conflicts) + 1,
            detail_code="workspace_branch_baseline_conflicted",
        )
        if (
            projected_evidence_bytes + serialized_conflicts_bytes + item_bytes
            > authority.limits.max_evidence_bytes
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
        durable: _DurableBranchContext | None = None,
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
        self._private_root_owner = captured.private_root_owner
        self._baseline_root = captured.baseline_root
        self._overlay_root = captured.overlay_root
        self._baseline = dict(captured.files)
        self._baseline_directories = captured.directories
        self._root_identity = captured.root_identity
        self._path_semantics = captured.path_semantics
        self._source_key = source_key
        self._capacity_lease = capacity_lease
        self._durable = durable
        self._overlay: dict[str, _FileIdentity] = (
            {}
            if durable is None
            else {path: value.private() for path, value in durable.record.overlay.items()}
        )
        self._tombstones: set[str] = set() if durable is None else set(durable.record.tombstones)
        self._state_lock = threading.RLock()
        self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
        self._publication_receipt: _PublicationReceipt | None = None
        self._rollback_receipt: _RollbackReceipt | None = None
        if durable is None:
            remaining_lifetime = self._limits.lifetime_ms / 1000
        else:
            expires_at = _parse_durable_branch_timestamp(
                durable.record.expires_at,
                "expires_at",
            )
            remaining_lifetime = max(
                0.0,
                (expires_at - datetime.now(UTC)).total_seconds(),
            )
        self._expires_at = time.monotonic() + remaining_lifetime
        self._timer = threading.Timer(self._limits.lifetime_ms / 1000, self._expire)
        self._timer.daemon = True
        if durable is None:
            self._timer.start()
        self.id = f"{source.id}#branch:{branch_id}"

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def lifecycle_status(self) -> WorkspaceBranchLifecycleStatus:
        shared = self._capacity_lease.terminal_lifecycle()
        if shared is not None:
            return shared
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

    def _validate_live_binding(
        self,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> None:
        durable = self._durable
        if durable is not None:
            binding_claim_lease.validate(durable.record.authority)

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
        if self._durable is not None:
            await self._durable_private_mutation(
                lambda: self._project_write(path, content),
                lambda: self._write(path, content, False, None),
            )
            return
        await _await_owned_thread(self._write, path, content, False, None)

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        if self._durable is None:
            result = await _await_owned_thread(self._write, path, content, True, None)
        else:
            result = await self._durable_private_mutation(
                lambda: self._project_write(path, content),
                lambda: self._write(path, content, True, None),
            )
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
        if self._durable is None:
            result = await _await_owned_thread(
                self._write,
                path,
                content,
                False,
                expected_revision,
            )
        else:
            result = await self._durable_private_mutation(
                lambda: self._project_write(path, content),
                lambda: self._write(path, content, False, expected_revision),
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
            proposed_overlay, proposed_tombstones = _project_overlay_write(
                relative,
                after,
                baseline=self._baseline,
                overlay=self._overlay,
                tombstones=self._tombstones,
            )
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
        if self._durable is not None:
            await self._durable_private_mutation(
                lambda: self._project_delete(path),
                lambda: self._delete(path, None),
            )
            return
        await _await_owned_thread(self._delete, path, None)

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(expected_revision) is not str or not expected_revision.strip():
            raise ValueError("Workspace expected_revision must be a nonblank string.")
        if self._durable is None:
            result = await _await_owned_thread(self._delete, path, expected_revision)
        else:
            result = await self._durable_private_mutation(
                lambda: self._project_delete(path),
                lambda: self._delete(path, expected_revision),
            )
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
            proposed_overlay, proposed_tombstones = _project_overlay_delete(
                relative,
                baseline=self._baseline,
                overlay=self._overlay,
                tombstones=self._tombstones,
            )
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

    def _project_write(
        self,
        path: str,
        content: bytes,
    ) -> tuple[dict[str, _FileIdentity], set[str]]:
        relative = self._validated_path(path)
        after = _identity_for_bytes(content)
        return _project_overlay_write(
            relative,
            after,
            baseline=self._baseline,
            overlay=self._overlay,
            tombstones=self._tombstones,
        )

    def _project_delete(
        self,
        path: str,
    ) -> tuple[dict[str, _FileIdentity], set[str]]:
        relative = self._validated_path(path)
        return _project_overlay_delete(
            relative,
            baseline=self._baseline,
            overlay=self._overlay,
            tombstones=self._tombstones,
        )

    async def _durable_private_mutation(
        self,
        projector: Callable[[], tuple[dict[str, _FileIdentity], set[str]]],
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        durable = self._durable
        if durable is None:
            raise AssertionError("Durable private mutation lost its coordinator.")
        binding_claim_lease = _binding_authority_claim_lease(
            self._source,
            durable.record.authority,
        )
        try:
            result = await self._durable_private_mutation_claimed(
                projector,
                operation,
                binding_claim_lease=binding_claim_lease,
            )
        except BaseException as primary:
            _release_binding_authority_claim_lease(binding_claim_lease, primary=primary)
            raise
        _release_binding_authority_claim_lease(binding_claim_lease)
        return result

    async def _durable_private_mutation_claimed(
        self,
        projector: Callable[[], tuple[dict[str, _FileIdentity], set[str]]],
        operation: Callable[[], _ResultT],
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> _ResultT:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable private mutation lost its coordinator.")
        async with durable.lock:
            self._validate_live_binding(binding_claim_lease)
            current = await _load_durable_branch_record(
                durable.store,
                session_id=durable.record.authority.session_id,
                storage_key=durable.storage_key,
            )
            if current is None:
                raise WorkspaceBranchFencedError("Durable workspace branch record disappeared.")
            if current != durable.record:
                raise WorkspaceBranchOperationConflict(
                    "Durable workspace branch changed under another owner."
                )
            if current.state not in {
                WorkspaceBranchDurableState.OPEN,
                WorkspaceBranchDurableState.CONFLICTED,
            }:
                raise WorkspaceBranchClosedError(
                    f"Durable workspace branch is {current.state.value}."
                )
            with self._state_lock:
                self._require_active()
                proposed_overlay, proposed_tombstones = projector()
                updated = current.model_copy(
                    update={
                        "state": WorkspaceBranchDurableState.OPEN,
                        "overlay": {
                            path: _DurableFileIdentity.from_private(identity)
                            for path, identity in proposed_overlay.items()
                        },
                        "tombstones": tuple(sorted(proposed_tombstones)),
                        "publication": None,
                        "revision": current.revision + 1,
                    }
                )
                updated = _validated_durable_branch_record(updated)
            result: list[_ResultT] = []

            def guarded_operation() -> None:
                self._validate_live_binding(binding_claim_lease)
                result.append(operation())

            try:
                await _publish_durable_branch_record(
                    durable.store,
                    session_id=current.authority.session_id,
                    storage_key=durable.storage_key,
                    expected=current,
                    updated=updated,
                    expected_run_epoch=current.authority.expected_run_epoch,
                    commit_guard=guarded_operation,
                )
            except BaseException as signal:
                try:
                    loaded = await _load_durable_branch_record(
                        durable.store,
                        session_id=current.authority.session_id,
                        storage_key=durable.storage_key,
                    )
                except BaseException as reconciliation_error:
                    with self._state_lock:
                        self._set_shared_uncertain_lifecycle_unlocked()
                    _raise_primary_with_reconciliation_failure(signal, reconciliation_error)
                if loaded is not None and loaded == updated and result:
                    durable.record = loaded
                    if not isinstance(signal, Exception):
                        raise
                    return result[0]
                with self._state_lock:
                    self._set_shared_uncertain_lifecycle_unlocked()
                raise
            durable.record = updated
            if not result:  # pragma: no cover - guarded store invariant
                raise AssertionError("Durable private mutation guard did not run.")
            return result[0]

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
            idempotency_key=request.idempotency_key,
            expected_run_epoch=request.expected_run_epoch,
            binding_generation=request.binding_generation,
        )
        if self._durable is not None:
            return await self._publish_durable(copied)
        return await _await_owned_thread(self._publish, copied)

    async def _publish_durable(
        self,
        request: WorkspaceBranchPublicationRequest,
    ) -> WorkspaceBranchPublicationResult:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable publication lost its coordinator.")
        binding_claim_lease = _binding_authority_claim_lease(
            self._source,
            durable.record.authority,
        )
        try:
            result = await self._publish_durable_claimed(
                request,
                binding_claim_lease=binding_claim_lease,
            )
        except BaseException as primary:
            _release_binding_authority_claim_lease(binding_claim_lease, primary=primary)
            raise
        _release_binding_authority_claim_lease(binding_claim_lease)
        return result

    async def _publish_durable_claimed(
        self,
        request: WorkspaceBranchPublicationRequest,
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> WorkspaceBranchPublicationResult:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable publication lost its coordinator.")
        if (
            request.idempotency_key is None
            or request.expected_run_epoch is None
            or request.binding_generation is None
        ):
            raise ValueError("Durable publication authority is incomplete.")
        self._validate_live_binding(binding_claim_lease)
        attached_record = durable.record
        current = await _load_durable_branch_record(
            durable.store,
            session_id=durable.record.authority.session_id,
            storage_key=durable.storage_key,
        )
        if current is None:
            raise WorkspaceBranchFencedError("Durable workspace branch record disappeared.")
        authority = current.authority
        if request.expected_run_epoch != authority.expected_run_epoch:
            raise SessionRunFenced(
                "Workspace branch publication run epoch does not match its creation authority."
            )
        if request.binding_generation != authority.binding_generation:
            raise WorkspaceBranchOperationConflict(
                "Workspace branch publication binding generation is stale."
            )
        await self._settle_observed_durable_terminal_handle(
            current,
            binding_claim_lease=binding_claim_lease,
        )
        if current != attached_record:
            current = await self._converge_and_settle_observed_durable_operation(
                current,
                binding_claim_lease=binding_claim_lease,
            )
        previous_attempt = next(
            (
                attempt
                for attempt in current.publication_attempts
                if attempt.idempotency_key == request.idempotency_key
            ),
            None,
        )
        if (
            previous_attempt is not None
            and previous_attempt.change_set_digest != request.change_set_digest
        ):
            raise WorkspaceBranchOperationConflict(
                "Workspace branch publication identity was reused with different material."
            )
        if current.publication is not None:
            persisted = current.publication
            if persisted.idempotency_key != request.idempotency_key:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch already has different publication authority."
                )
            if (
                request.branch_id != current.branch_id
                or request.baseline_revision != current.baseline_revision
                or request.change_set_digest != persisted.change_set.digest
            ):
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch publication retry changed its material."
                )
            if current.state is WorkspaceBranchDurableState.COMMITTED:
                return _durable_publication_result(current)
            if current.state is WorkspaceBranchDurableState.FAILED:
                return _durable_failed_publication_result(current)
        elif current != attached_record:
            raise WorkspaceBranchOperationConflict(
                "Workspace branch changed under this publication owner."
            )
        with self._state_lock:
            self._require_active(allow_publishing=True)
            if (
                request.branch_id != self._branch_id
                or request.baseline_revision != self._baseline_revision
            ):
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch publication authority does not match the branch."
                )
            change_set = self._change_set_unlocked()
            if request.change_set_digest != change_set.digest:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch changed after publication authority was issued."
                )
        if current.publication is not None and (
            current.publication.idempotency_key != request.idempotency_key
            or current.publication.change_set != change_set
        ):
            raise WorkspaceBranchOperationConflict(
                "Workspace branch already has different publication authority."
            )
        attempts = current.publication_attempts
        if previous_attempt is None:
            if len(attempts) >= current.limits.max_publication_attempts:
                return WorkspaceBranchPublicationResult(
                    status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                    evidence=self._evidence(
                        WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                        change_set,
                        paths=(),
                        detail_code="publication_attempt_limit_exceeded",
                    ),
                )
            attempts = (
                *attempts,
                _DurablePublicationAttempt(
                    idempotency_key=request.idempotency_key,
                    change_set_digest=change_set.digest,
                ),
            )
        if current.state not in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
            WorkspaceBranchDurableState.PUBLICATION_INTENT,
            WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
        }:
            raise WorkspaceBranchClosedError(f"Durable workspace branch is {current.state.value}.")
        if current.state in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
        }:
            intended = current.model_copy(
                update={
                    "state": WorkspaceBranchDurableState.PUBLICATION_INTENT,
                    "publication": _DurablePublication(
                        idempotency_key=request.idempotency_key,
                        staging_owner=_durable_publication_staging_owner(
                            private_root_owner=current.private_root_owner,
                            idempotency_key=request.idempotency_key,
                            change_set_digest=change_set.digest,
                        ),
                        change_set=change_set,
                    ),
                    "publication_attempts": attempts,
                    "revision": current.revision + 1,
                }
            )
            self._validate_live_binding(binding_claim_lease)
            current = await _publish_or_reconcile_exact_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=intended,
                expected_run_epoch=authority.expected_run_epoch,
            )
            durable.record = current
        with self._state_lock:
            self._lifecycle = WorkspaceBranchLifecycleStatus.PUBLISHING
        try:
            if current.state in {
                WorkspaceBranchDurableState.PUBLICATION_INTENT,
                WorkspaceBranchDurableState.CONFLICTED,
            }:
                await _await_owned_thread(
                    self._preflight_durable_publication,
                    change_set,
                )
                started = current.model_copy(
                    update={
                        "state": WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
                        "revision": current.revision + 1,
                    }
                )
                self._validate_live_binding(binding_claim_lease)
                current = await _publish_or_reconcile_exact_record(
                    durable.store,
                    session_id=authority.session_id,
                    storage_key=durable.storage_key,
                    expected=current,
                    updated=started,
                    expected_run_epoch=authority.expected_run_epoch,
                )
                durable.record = current
            current = await self._converge_and_record_publication_progress(
                current,
                binding_claim_lease=binding_claim_lease,
            )
        except WorkspaceBranchOperationConflict:
            settled = await _load_durable_branch_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
            )
            if settled is not None:
                settled = await self._converge_and_settle_observed_durable_operation(
                    settled,
                    binding_claim_lease=binding_claim_lease,
                )
            if (
                settled is not None
                and settled.state is WorkspaceBranchDurableState.COMMITTED
                and settled.publication is not None
                and settled.publication.idempotency_key == request.idempotency_key
                and settled.publication.change_set == change_set
            ):
                return _durable_publication_result(settled)
            if (
                settled is not None
                and settled.state is WorkspaceBranchDurableState.FAILED
                and settled.publication is not None
                and settled.publication.idempotency_key == request.idempotency_key
                and settled.publication.change_set == change_set
            ):
                return _durable_failed_publication_result(settled)
            raise
        except _DurablePublicationConflict as conflict:
            conflicted = current.model_copy(
                update={
                    "state": WorkspaceBranchDurableState.CONFLICTED,
                    "revision": current.revision + 1,
                }
            )
            self._validate_live_binding(binding_claim_lease)
            conflicted = await _publish_or_reconcile_exact_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=conflicted,
                expected_run_epoch=authority.expected_run_epoch,
            )
            durable.record = conflicted
            with self._state_lock:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
            evidence = self._evidence(
                WorkspaceBranchOutcomeStatus.CONFLICTED,
                change_set,
                paths=tuple(item.path for item in conflict.conflicts),
                detail_code="affected_path_conflicted",
            )
            return WorkspaceBranchPublicationResult(
                status=WorkspaceBranchOutcomeStatus.CONFLICTED,
                evidence=evidence,
                conflicts=conflict.conflicts,
            )
        except _DurablePublicationFailed as publication_failure:
            failed = current.model_copy(
                update={
                    "state": WorkspaceBranchDurableState.FAILED,
                    "failure": _DurableFailure(
                        phase="publication",
                        outcome="failed",
                        detail_code=publication_failure.detail_code,
                    ),
                    "revision": current.revision + 1,
                }
            )
            self._validate_live_binding(binding_claim_lease)
            failed = await _publish_or_reconcile_exact_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=failed,
                expected_run_epoch=authority.expected_run_epoch,
                on_exact_control_signal=lambda record: (
                    self._settle_observed_durable_terminal_handle(
                        record,
                        binding_claim_lease=binding_claim_lease,
                    )
                ),
                retain_uncertain_owner=lambda: self._retain_durable_terminal_settlement(
                    failed,
                    publication_expected=current,
                    binding_claim_lease=binding_claim_lease,
                ),
            )
            await self._settle_observed_durable_terminal_handle(
                failed,
                binding_claim_lease=binding_claim_lease,
                suppress_ordinary_cleanup_failure=True,
            )
            return _durable_failed_publication_result(failed)
        except _DurablePublicationAmbiguous as ambiguous_error:
            ambiguous = current.model_copy(
                update={
                    "state": WorkspaceBranchDurableState.AMBIGUOUS,
                    "revision": current.revision + 1,
                }
            )
            self._validate_live_binding(binding_claim_lease)
            ambiguous = await _publish_or_reconcile_exact_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=ambiguous,
                expected_run_epoch=authority.expected_run_epoch,
            )
            durable.record = ambiguous
            with self._state_lock:
                self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.FENCED)
            return WorkspaceBranchPublicationResult(
                status=WorkspaceBranchOutcomeStatus.AMBIGUOUS,
                evidence=self._evidence(
                    WorkspaceBranchOutcomeStatus.AMBIGUOUS,
                    change_set,
                    paths=ambiguous_error.paths,
                    detail_code="source_publication_ambiguous",
                ),
            )
        committed = current.model_copy(
            update={
                "state": WorkspaceBranchDurableState.COMMITTED,
                "revision": current.revision + 1,
            }
        )
        self._validate_live_binding(binding_claim_lease)
        committed = await _publish_or_reconcile_exact_record(
            durable.store,
            session_id=authority.session_id,
            storage_key=durable.storage_key,
            expected=current,
            updated=committed,
            expected_run_epoch=authority.expected_run_epoch,
            on_exact_control_signal=lambda record: self._settle_observed_durable_terminal_handle(
                record,
                binding_claim_lease=binding_claim_lease,
            ),
            retain_uncertain_owner=lambda: self._retain_durable_terminal_settlement(
                committed,
                publication_expected=current,
                binding_claim_lease=binding_claim_lease,
            ),
        )
        await self._settle_observed_durable_terminal_handle(
            committed,
            binding_claim_lease=binding_claim_lease,
            suppress_ordinary_cleanup_failure=True,
        )
        return _durable_publication_result(committed)

    async def _converge_and_record_publication_progress(
        self,
        current: _DurableBranchRecord,
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> _DurableBranchRecord:
        durable = self._durable
        publication = current.publication
        if durable is None or publication is None:
            raise WorkspaceBranchFencedError("Durable publication intent is incomplete.")
        if current.state is not WorkspaceBranchDurableState.PUBLICATION_PROGRESS:
            raise WorkspaceBranchFencedError("Durable publication was not started.")
        progressed = current.model_copy(
            update={
                "revision": current.revision + 1,
            }
        )

        def converge_publication() -> None:
            self._validate_live_binding(binding_claim_lease)
            self._converge_durable_publication(publication)

        try:
            progressed = await _publish_durable_branch_record(
                durable.store,
                session_id=current.authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=progressed,
                expected_run_epoch=current.authority.expected_run_epoch,
                commit_guard=converge_publication,
            )
        except BaseException as signal:
            staging_error = _publication_staging_cleanup_error(signal)
            if staging_error is None:
                raise
            fence_local_workspace_source(self._source.root)
            try:
                _retain_source_staging_cleanup(
                    staging_error,
                    source_key=self._source_key,
                )
            except BaseException as retention_error:
                _raise_primary_with_reconciliation_failure(signal, retention_error)
            raise
        durable.record = progressed
        return progressed

    def _preflight_durable_publication(
        self,
        change_set: WorkspaceBranchChangeSet,
    ) -> None:
        """Prove source inspection before recording the first mutation dispatch."""

        try:
            with workspace_source_lock(self._source.root, exclusive=True):
                self._validate_source_root()
                for change in change_set.changes:
                    _inspect_source_path(
                        self._source.root,
                        change.path,
                        max_bytes=self._limits.max_file_bytes,
                    )
        except OSError:
            raise _DurablePublicationFailed("source_publication_inspection_failed") from None

    def _converge_durable_publication(
        self,
        publication: _DurablePublication,
    ) -> None:
        change_set = publication.change_set
        with workspace_source_lock(self._source.root, exclusive=True):
            self._validate_source_root()
            for change in change_set.changes:
                if change.after is None:
                    continue
                try:
                    settle_regular_staging(
                        self._source.root,
                        change.path,
                        _durable_publication_staging_name(publication, change.path),
                        expected_sha256=change.after.sha256,
                        expected_bytes=change.after.bytes,
                    )
                except _LocalGuardStagingConflictError:
                    raise _DurablePublicationAmbiguous((change.path,)) from None
            positions: dict[str, Literal["before", "after"]] = {}
            conflicts: list[WorkspaceBranchConflict] = []
            after_count = 0
            for change in change_set.changes:
                try:
                    actual_path, actual_kind, actual = _inspect_source_path(
                        self._source.root,
                        change.path,
                        max_bytes=self._limits.max_file_bytes,
                    )
                except OSError:
                    raise _DurablePublicationAmbiguous(
                        tuple(
                            sorted(
                                {
                                    change.path,
                                    *(
                                        path
                                        for path, position in positions.items()
                                        if position == "after"
                                    ),
                                }
                            )
                        )
                    ) from None
                before_matches = actual_path == change.path and (
                    (change.before is None and actual_kind == "missing")
                    or (
                        change.before is not None
                        and actual_kind == "file"
                        and actual is not None
                        and actual.public() == change.before
                    )
                )
                after_matches = actual_path == change.path and (
                    (change.after is None and actual_kind == "missing")
                    or (
                        change.after is not None
                        and actual_kind == "file"
                        and actual is not None
                        and actual.public() == change.after
                    )
                )
                if after_matches:
                    positions[change.path] = "after"
                    after_count += 1
                elif before_matches:
                    positions[change.path] = "before"
                else:
                    conflicts.append(
                        WorkspaceBranchConflict(
                            path=actual_path,
                            expected=change.before,
                            actual=None if actual is None else actual.public(),
                            actual_kind=actual_kind,
                        )
                    )
            if conflicts:
                paths = tuple(sorted({item.path for item in conflicts}))
                if after_count:
                    raise _DurablePublicationAmbiguous(paths)
                raise _DurablePublicationConflict(tuple(conflicts))
            for change in change_set.changes:
                if positions[change.path] == "after":
                    continue
                with workspace_path_lock(self._source.root, change.path):
                    if change.operation == "created":
                        try:
                            create_regular(
                                self._source.root,
                                change.path,
                                self._overlay_bytes(change.path),
                                staging_name=_durable_publication_staging_name(
                                    publication,
                                    change.path,
                                ),
                            )
                        except FileExistsError:
                            raise _DurablePublicationAmbiguous((change.path,)) from None
                    elif change.operation == "modified":
                        before = change.before
                        if before is None:  # pragma: no cover - model invariant
                            raise AssertionError("Modified change has no before identity.")
                        try:
                            replace_regular_if_revision(
                                self._source.root,
                                change.path,
                                self._overlay_bytes(change.path),
                                f"sha256:{before.sha256}",
                                staging_name=_durable_publication_staging_name(
                                    publication,
                                    change.path,
                                ),
                            )
                        except FileExistsError:
                            raise _DurablePublicationAmbiguous((change.path,)) from None
                    else:
                        before = change.before
                        if before is None:  # pragma: no cover - model invariant
                            raise AssertionError("Deleted change has no before identity.")
                        delete_regular_if_revision(
                            self._source.root,
                            change.path,
                            f"sha256:{before.sha256}",
                        )
            verification = self._published_identity_conflicts(change_set)
            if verification:
                raise _DurablePublicationAmbiguous(
                    tuple(sorted(item.path for item in verification))
                )

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
                    self._set_shared_terminal_lifecycle_unlocked(
                        WorkspaceBranchLifecycleStatus.FENCED
                    )
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
            self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.COMMITTED)
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

        def semantics_changed(path: Path) -> bool:
            try:
                semantics = _directory_lookup_semantics(path)
            except OSError:
                return True
            return (
                semantics is _DirectoryLookupSemantics.UNKNOWN
                or semantics is not self._path_semantics
            )

        if any(
            semantics_changed(path)
            for path in (self._source.root, self._baseline_root, self._overlay_root)
        ):
            conflicts.add(
                WorkspaceBranchConflict(
                    path="__path_semantics__",
                    actual_kind="special",
                )
            )
        overlay_directories = {
            "/".join(parts[:index])
            for path in self._overlay
            for parts in (path.split("/")[:-1],)
            for index in range(1, len(parts) + 1)
        }
        affected_directories = {
            "/".join(parts[:index])
            for change in change_set.changes
            for parts in (change.path.split("/")[:-1],)
            for index in range(1, len(parts) + 1)
        }
        for relative in sorted(self._baseline_directories):
            if semantics_changed(self._source.root / relative) or semantics_changed(
                self._baseline_root / relative
            ):
                conflicts.add(WorkspaceBranchConflict(path=relative, actual_kind="special"))
        for relative in sorted(overlay_directories):
            if semantics_changed(self._overlay_root / relative):
                conflicts.add(WorkspaceBranchConflict(path=relative, actual_kind="special"))
        for relative in sorted(affected_directories - self._baseline_directories):
            source_directory = self._source.root / relative
            try:
                source_info = source_directory.stat(follow_symlinks=False)
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError:
                conflicts.add(WorkspaceBranchConflict(path=relative, actual_kind="special"))
                continue
            if stat.S_ISDIR(source_info.st_mode) and semantics_changed(source_directory):
                conflicts.add(WorkspaceBranchConflict(path=relative, actual_kind="special"))
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

    async def rollback(
        self,
        request: WorkspaceBranchRollbackRequest | None = None,
    ) -> WorkspaceBranchRollbackResult:
        if self._durable is not None:
            if request is None:
                raise ValueError("Durable workspace rollback requires explicit authority.")
            return await self._rollback_durable(request)
        return await _await_owned_thread(self._rollback, "explicit")

    async def _rollback_durable(
        self,
        request: WorkspaceBranchRollbackRequest,
    ) -> WorkspaceBranchRollbackResult:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable rollback lost its coordinator.")
        binding_claim_lease = _binding_authority_claim_lease(
            self._source,
            durable.record.authority,
        )
        try:
            result = await self._rollback_durable_claimed(
                request,
                binding_claim_lease=binding_claim_lease,
            )
        except BaseException as primary:
            _release_binding_authority_claim_lease(binding_claim_lease, primary=primary)
            raise
        _release_binding_authority_claim_lease(binding_claim_lease)
        return result

    async def _rollback_durable_claimed(
        self,
        request: WorkspaceBranchRollbackRequest,
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> WorkspaceBranchRollbackResult:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable rollback lost its coordinator.")
        self._validate_live_binding(binding_claim_lease)
        attached_record = durable.record
        current = await _load_durable_branch_record(
            durable.store,
            session_id=durable.record.authority.session_id,
            storage_key=durable.storage_key,
        )
        if current is None:
            raise WorkspaceBranchFencedError("Durable workspace branch record disappeared.")
        authority = current.authority
        if request.expected_run_epoch != authority.expected_run_epoch:
            raise SessionRunFenced(
                "Workspace branch rollback run epoch does not match creation authority."
            )
        if (
            request.branch_id != current.branch_id
            or request.binding_generation != authority.binding_generation
        ):
            raise WorkspaceBranchOperationConflict(
                "Workspace branch rollback authority does not match the branch."
            )
        await self._settle_observed_durable_terminal_handle(
            current,
            binding_claim_lease=binding_claim_lease,
        )
        if current != attached_record:
            current = await self._converge_and_settle_observed_durable_operation(
                current,
                binding_claim_lease=binding_claim_lease,
            )
        if current.rollback is not None:
            if (
                current.rollback.idempotency_key != request.idempotency_key
                or current.rollback.reason != request.reason
            ):
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch already has different rollback authority."
                )
        elif current != attached_record:
            raise WorkspaceBranchOperationConflict(
                "Workspace branch changed under this rollback owner."
            )
        if request.reason == "expired" and not _durable_record_is_expired(current):
            raise WorkspaceBranchOperationConflict(
                "Workspace branch cannot expire before its durable lifetime ends."
            )
        if current.state in {
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
        }:
            return _durable_rollback_result(current)
        if current.state not in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
            WorkspaceBranchDurableState.ROLLBACK_INTENT,
        }:
            raise WorkspaceBranchClosedError(f"Durable workspace branch is {current.state.value}.")
        try:
            if current.state is not WorkspaceBranchDurableState.ROLLBACK_INTENT:
                paths = tuple(sorted(self._overlay.keys() | self._tombstones))
                intended = current.model_copy(
                    update={
                        "state": WorkspaceBranchDurableState.ROLLBACK_INTENT,
                        "rollback": _DurableRollback(
                            idempotency_key=request.idempotency_key,
                            reason=request.reason,
                            paths=paths,
                        ),
                        "revision": current.revision + 1,
                    }
                )
                self._validate_live_binding(binding_claim_lease)
                current = await _publish_or_reconcile_exact_record(
                    durable.store,
                    session_id=authority.session_id,
                    storage_key=durable.storage_key,
                    expected=current,
                    updated=intended,
                    expected_run_epoch=authority.expected_run_epoch,
                )
            terminal_state = (
                WorkspaceBranchDurableState.EXPIRED
                if request.reason == "expired"
                else WorkspaceBranchDurableState.ROLLED_BACK
            )
            terminal = current.model_copy(
                update={"state": terminal_state, "revision": current.revision + 1}
            )
            self._validate_live_binding(binding_claim_lease)
            terminal = await _publish_or_reconcile_exact_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
                expected=current,
                updated=terminal,
                expected_run_epoch=authority.expected_run_epoch,
                on_exact_control_signal=lambda record: (
                    self._settle_observed_durable_terminal_handle(
                        record,
                        binding_claim_lease=binding_claim_lease,
                    )
                ),
                retain_uncertain_owner=lambda: self._retain_durable_terminal_settlement(
                    terminal,
                    publication_expected=current,
                    binding_claim_lease=binding_claim_lease,
                ),
            )
        except WorkspaceBranchOperationConflict:
            settled = await _load_durable_branch_record(
                durable.store,
                session_id=authority.session_id,
                storage_key=durable.storage_key,
            )
            if settled is not None:
                settled = await self._converge_and_settle_observed_durable_operation(
                    settled,
                    binding_claim_lease=binding_claim_lease,
                )
                if (
                    settled.state
                    in {
                        WorkspaceBranchDurableState.ROLLED_BACK,
                        WorkspaceBranchDurableState.EXPIRED,
                    }
                    and settled.rollback is not None
                    and settled.rollback.idempotency_key == request.idempotency_key
                    and settled.rollback.reason == request.reason
                ):
                    return _durable_rollback_result(settled)
            raise
        durable.record = terminal
        result = _durable_rollback_result(terminal)
        with self._state_lock:
            self._validate_live_binding(binding_claim_lease)
            self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLING_BACK
        await self._settle_observed_durable_terminal_handle(
            terminal,
            binding_claim_lease=binding_claim_lease,
        )
        return result

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
                    private_root_owner=self._private_root_owner,
                )
                raise
            self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.ROLLED_BACK)
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

    def _set_shared_terminal_lifecycle_unlocked(
        self,
        lifecycle: WorkspaceBranchLifecycleStatus,
    ) -> None:
        self._capacity_lease.mark_terminal(lifecycle)
        self._lifecycle = lifecycle

    def _set_shared_uncertain_lifecycle_unlocked(self) -> None:
        self._capacity_lease.mark_uncertain()
        self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED

    def _require_active(self, *, allow_publishing: bool = False) -> None:
        shared = self._capacity_lease.terminal_lifecycle()
        if shared is WorkspaceBranchLifecycleStatus.FENCED:
            raise WorkspaceBranchFencedError("Workspace branch is fenced.")
        if shared is not None:
            raise WorkspaceBranchClosedError(f"Workspace branch is not active: {shared.value}.")
        if self._lifecycle is WorkspaceBranchLifecycleStatus.FENCED:
            raise WorkspaceBranchFencedError("Workspace branch is fenced.")
        if self._lifecycle is WorkspaceBranchLifecycleStatus.PUBLISHING and allow_publishing:
            return
        if self._lifecycle is not WorkspaceBranchLifecycleStatus.ACTIVE:
            raise WorkspaceBranchClosedError(
                f"Workspace branch is not active: {self._lifecycle.value}."
            )
        if time.monotonic() >= self._expires_at:
            if self._durable is not None:
                raise WorkspaceBranchClosedError(
                    "Durable workspace branch expired; recover it to settle cleanup."
                )
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
        return _changes_for_retained_branch_state(self._baseline, overlay, tombstones)

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
                private_root_owner=self._private_root_owner,
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
        _settle_private_tree_cleanup(
            self._private_root,
            self._capacity_lease,
            private_root_owner=self._private_root_owner,
        )

    async def _converge_and_settle_observed_durable_operation(
        self,
        record: _DurableBranchRecord,
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> _DurableBranchRecord:
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable operation convergence lost its coordinator.")
        if record.state in {
            WorkspaceBranchDurableState.PUBLICATION_INTENT,
            WorkspaceBranchDurableState.PUBLICATION_PROGRESS,
        }:
            publication = record.publication
            if publication is None:  # pragma: no cover - record invariant
                raise AssertionError("Durable publication lost its intent evidence.")
            durable.record = record
            await self._publish_durable_claimed(
                WorkspaceBranchPublicationRequest(
                    branch_id=record.branch_id,
                    baseline_revision=record.baseline_revision,
                    change_set_digest=publication.change_set.digest,
                    idempotency_key=publication.idempotency_key,
                    expected_run_epoch=record.authority.expected_run_epoch,
                    binding_generation=record.authority.binding_generation,
                ),
                binding_claim_lease=binding_claim_lease,
            )
        elif record.state is WorkspaceBranchDurableState.ROLLBACK_INTENT:
            rollback = record.rollback
            if rollback is None:  # pragma: no cover - record invariant
                raise AssertionError("Durable rollback lost its intent evidence.")
            durable.record = record
            await self._rollback_durable_claimed(
                WorkspaceBranchRollbackRequest(
                    branch_id=record.branch_id,
                    idempotency_key=rollback.idempotency_key,
                    expected_run_epoch=record.authority.expected_run_epoch,
                    binding_generation=record.authority.binding_generation,
                    reason=rollback.reason,
                ),
                binding_claim_lease=binding_claim_lease,
            )
        else:
            await self._settle_observed_durable_terminal_handle(
                record,
                binding_claim_lease=binding_claim_lease,
            )
            return record
        settled = await _load_durable_branch_record(
            durable.store,
            session_id=record.authority.session_id,
            storage_key=durable.storage_key,
        )
        if settled is None:
            raise WorkspaceBranchFencedError(
                "Durable workspace branch disappeared during operation convergence."
            )
        await self._settle_observed_durable_terminal_handle(
            settled,
            binding_claim_lease=binding_claim_lease,
        )
        return settled

    async def _settle_observed_durable_terminal_handle(
        self,
        record: _DurableBranchRecord,
        *,
        binding_claim_lease: _BindingAuthorityClaimLease,
        suppress_ordinary_cleanup_failure: bool = False,
    ) -> None:
        if record.state not in {
            WorkspaceBranchDurableState.COMMITTED,
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
            WorkspaceBranchDurableState.FAILED,
        }:
            return
        self._record_durable_terminal_handle(record)
        try:
            await _await_owned_thread(
                _settle_durable_private_tree_removal,
                self._source,
                record,
                self._capacity_lease,
                settle_result=lambda _result: self._complete_durable_terminal_settlement(record),
                pre_dispatch_checkpoint=False,
            )
        except BaseException as cleanup_error:
            if not self._capacity_lease.released():
                self._retain_durable_terminal_settlement(
                    record,
                    publication_expected=None,
                    binding_claim_lease=binding_claim_lease,
                )
                if suppress_ordinary_cleanup_failure and isinstance(cleanup_error, Exception):
                    self._record_durable_terminal_handle(record)
                    return
            raise

    def _retain_durable_terminal_settlement(
        self,
        record: _DurableBranchRecord,
        *,
        publication_expected: _DurableBranchRecord | None,
        binding_claim_lease: _BindingAuthorityClaimLease,
    ) -> None:
        if record.state not in {
            WorkspaceBranchDurableState.COMMITTED,
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
            WorkspaceBranchDurableState.FAILED,
        }:
            raise AssertionError("Retained durable branch settlement is not terminal.")
        key = _durable_terminal_settlement_key(self)
        existing = _DURABLE_TERMINAL_SETTLEMENTS.payload(key)
        if existing is not None:
            if existing.branch is not self:
                raise RuntimeError("Retained terminal settlement owner changed.")
            return
        self._capacity_lease.retain_for_cleanup()
        retained = _DURABLE_TERMINAL_SETTLEMENTS.retain(
            key,
            source_key=self._source_key,
            payload=_RetainedDurableTerminalSettlement(
                branch=self,
                terminal_record=record,
                publication_expected=publication_expected,
                binding_claim_lease=binding_claim_lease,
            ),
        )
        if retained:
            binding_claim_lease.retain_for_settlement()

    def _complete_durable_terminal_settlement(
        self,
        record: _DurableBranchRecord,
    ) -> None:
        self._record_durable_terminal_handle(record)
        self._capacity_lease.release_after_cleanup()
        _forget_private_tree_cleanup(self._private_root, self._capacity_lease)
        _release_durable_terminal_settlement(self)

    def _record_durable_terminal_handle(
        self,
        record: _DurableBranchRecord,
    ) -> None:
        if record.state not in {
            WorkspaceBranchDurableState.COMMITTED,
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
            WorkspaceBranchDurableState.FAILED,
        }:
            return
        durable = self._durable
        if durable is None:  # pragma: no cover - caller invariant
            raise AssertionError("Durable terminal settlement lost its coordinator.")
        durable.record = record
        with self._state_lock:
            self._record_durable_terminal_handle_unlocked(record)

    def _record_durable_terminal_handle_unlocked(
        self,
        record: _DurableBranchRecord,
    ) -> None:
        if record.state is WorkspaceBranchDurableState.COMMITTED:
            publication = record.publication
            if publication is None:  # pragma: no cover - record invariant
                raise AssertionError("Durable publication lost terminal evidence.")
            self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.COMMITTED)
            self._publication_receipt = _PublicationReceipt(
                change_set_digest=publication.change_set.digest,
                paths=tuple(change.path for change in publication.change_set.changes),
            )
        elif record.state is WorkspaceBranchDurableState.FAILED:
            self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.FENCED)
        elif record.state in {
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
        }:
            rollback = record.rollback
            if rollback is None:  # pragma: no cover - record invariant
                raise AssertionError("Durable rollback lost terminal evidence.")
            self._set_shared_terminal_lifecycle_unlocked(WorkspaceBranchLifecycleStatus.ROLLED_BACK)
            self._rollback_receipt = _RollbackReceipt(
                paths=rollback.paths,
                detail_code=(
                    "workspace_branch_expired"
                    if rollback.reason == "expired"
                    else "workspace_branch_rolled_back"
                ),
            )
        else:  # pragma: no cover - caller invariant
            raise AssertionError("Durable branch record is not terminal.")

    def _finish_terminal_cleanup_unlocked(self) -> None:
        try:
            self._cleanup_and_release()
        except BaseException as cleanup_error:
            _retain_private_tree_cleanup(
                self._private_root,
                self._capacity_lease,
                private_root_owner=self._private_root_owner,
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
                private_root_owner=self._private_root_owner,
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


def _remove_private_tree(
    path: Path,
    *,
    private_root_owner: str | None = None,
) -> None:
    if private_root_owner is not None:
        _remove_owned_private_tree(path, private_root_owner)
        return
    with suppress(FileNotFoundError):
        shutil.rmtree(path)


@dataclass(frozen=True, slots=True)
class _CleanupQuarantineClaim:
    device: int
    inode: int
    file_identity: os.stat_result = field(compare=False)


class _CleanupQuarantineReplacement(WorkspaceBranchFencedError):
    """A raced unowned cleanup pathname was restored to its public location."""


class _CleanupQuarantineClaimMissing(WorkspaceBranchFencedError):
    """The exact cleanup claim disappeared before it could be inspected."""


def _cleanup_quarantine_claim_path(quarantine: Path) -> Path:
    name_digest = hashlib.sha256(os.fsencode(quarantine.name)).hexdigest()
    return quarantine.with_name(f".cayu-cleanup-{name_digest}.claim")


def _cleanup_quarantine_claim_staging_name(quarantine: Path) -> str:
    return f"{_cleanup_quarantine_claim_path(quarantine).name}.staging"


def _cleanup_quarantine_claim_bytes(
    owner: str,
    identity: os.stat_result,
) -> bytes:
    try:
        return (
            f"{_PRIVATE_CLEANUP_CLAIM_PREFIX}\n{owner}\n{identity.st_dev}\n{identity.st_ino}\n"
        ).encode("ascii")
    except UnicodeEncodeError as error:  # pragma: no cover - generated owner invariant
        raise WorkspaceBranchFencedError("Durable private cleanup identity is invalid.") from error


def _read_cleanup_quarantine_claim(
    parent_fd: int,
    quarantine: Path,
    owner: str,
) -> _CleanupQuarantineClaim:
    claim_path = _cleanup_quarantine_claim_path(quarantine)
    claim_fd = -1
    try:
        claim_fd = os.open(
            claim_path.name,
            _PRIVATE_OWNER_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        claim_identity = os.fstat(claim_fd)
        if (
            not stat.S_ISREG(claim_identity.st_mode)
            or claim_identity.st_size <= 0
            or claim_identity.st_size > _PRIVATE_CLEANUP_CLAIM_MAX_BYTES
        ):
            raise WorkspaceBranchFencedError("Durable private cleanup identity is invalid.")
        content = bytearray()
        while len(content) <= _PRIVATE_CLEANUP_CLAIM_MAX_BYTES:
            chunk = os.read(
                claim_fd,
                _PRIVATE_CLEANUP_CLAIM_MAX_BYTES + 1 - len(content),
            )
            if not chunk:
                break
            content.extend(chunk)
        current = os.stat(
            claim_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not os.path.samestat(claim_identity, current):
            raise WorkspaceBranchFencedError("Durable private cleanup identity changed.")
        try:
            lines = bytes(content).decode("ascii").splitlines()
            if len(lines) != 4 or lines[0] != _PRIVATE_CLEANUP_CLAIM_PREFIX or lines[1] != owner:
                raise ValueError
            device = int(lines[2])
            inode = int(lines[3])
            if device < 0 or inode <= 0 or lines[2] != str(device) or lines[3] != str(inode):
                raise ValueError
            canonical = (f"{_PRIVATE_CLEANUP_CLAIM_PREFIX}\n{owner}\n{device}\n{inode}\n").encode(
                "ascii"
            )
            if bytes(content) != canonical:
                raise ValueError
        except (UnicodeDecodeError, ValueError) as error:
            raise WorkspaceBranchFencedError(
                "Durable private cleanup identity is invalid."
            ) from error
        return _CleanupQuarantineClaim(
            device=device,
            inode=inode,
            file_identity=claim_identity,
        )
    except FileNotFoundError as error:
        raise _CleanupQuarantineClaimMissing(
            "Durable private cleanup identity is missing."
        ) from error
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private cleanup identity is unavailable."
        ) from error
    finally:
        if claim_fd >= 0:
            os.close(claim_fd)


def _lock_cleanup_claim_staging(staging_fd: int) -> None:
    try:
        lock_module = __import__("fcntl")
        lock_module.flock(staging_fd, lock_module.LOCK_EX)
    except (ImportError, OSError) as error:
        raise WorkspaceBranchFencedError(
            "Durable private cleanup staging ownership is unavailable."
        ) from error


def _write_cleanup_claim_staging(staging_fd: int, expected: bytes) -> None:
    os.ftruncate(staging_fd, 0)
    os.lseek(staging_fd, 0, os.SEEK_SET)
    remaining = memoryview(expected)
    while remaining:
        written = os.write(staging_fd, remaining)
        if written <= 0:  # pragma: no cover - os.write contract
            raise OSError(errno.EIO, "cleanup staging write made no progress")
        remaining = remaining[written:]
    os.fsync(staging_fd)


def _cleanup_claim_staging_is_repairable(
    parent_fd: int,
    claim_path: Path,
    staging_name: str,
    staging_fd: int,
    expected: bytes,
) -> bool:
    identity = os.fstat(staging_fd)
    if (
        not stat.S_ISREG(identity.st_mode)
        or stat.S_IMODE(identity.st_mode) != _PRIVATE_FILE_MODE
        or identity.st_uid != os.geteuid()
        or identity.st_nlink != 1
        or identity.st_size >= len(expected)
    ):
        return False
    try:
        current = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if not os.path.samestat(identity, current):
        return False
    try:
        os.stat(
            claim_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        return False
    os.lseek(staging_fd, 0, os.SEEK_SET)
    content = bytearray()
    while len(content) < len(expected):
        chunk = os.read(staging_fd, len(expected) - len(content))
        if not chunk:
            break
        content.extend(chunk)
    return expected.startswith(content)


def _validate_exact_cleanup_claim_staging(
    parent_fd: int,
    staging_name: str,
    staging_fd: int,
    expected: bytes,
) -> tuple[os.stat_result, bool]:
    try:
        staging_identity = os.fstat(staging_fd)
        if (
            not stat.S_ISREG(staging_identity.st_mode)
            or staging_identity.st_size != len(expected)
            or staging_identity.st_size > _PRIVATE_CLEANUP_CLAIM_MAX_BYTES
        ):
            raise WorkspaceBranchFencedError("Durable private cleanup staging identity is invalid.")
        os.lseek(staging_fd, 0, os.SEEK_SET)
        content = bytearray()
        while len(content) <= _PRIVATE_CLEANUP_CLAIM_MAX_BYTES:
            chunk = os.read(
                staging_fd,
                _PRIVATE_CLEANUP_CLAIM_MAX_BYTES + 1 - len(content),
            )
            if not chunk:
                break
            content.extend(chunk)
        try:
            current = os.stat(
                staging_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            path_present = False
        else:
            path_present = True
            if not os.path.samestat(staging_identity, current):
                raise WorkspaceBranchFencedError(
                    "Durable private cleanup staging identity changed."
                )
        if bytes(content) != expected:
            raise WorkspaceBranchFencedError(
                "Durable private cleanup staging identity conflicts with the owned tree."
            )
        return staging_identity, path_present
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private cleanup staging identity is unavailable."
        ) from error


def _remove_exact_cleanup_claim_staging(
    parent_fd: int,
    staging_name: str,
    staging_identity: os.stat_result,
) -> None:
    try:
        current = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not os.path.samestat(staging_identity, current):
        raise WorkspaceBranchFencedError("Durable private cleanup staging identity changed.")
    try:
        os.unlink(staging_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return


def _read_matching_cleanup_quarantine_claim(
    parent_fd: int,
    quarantine: Path,
    owner: str,
    owned_identity: os.stat_result,
) -> _CleanupQuarantineClaim:
    claim = _read_cleanup_quarantine_claim(parent_fd, quarantine, owner)
    if claim.device != owned_identity.st_dev or claim.inode != owned_identity.st_ino:
        raise WorkspaceBranchFencedError(
            "Durable private cleanup identity conflicts with the owned tree."
        )
    return claim


def _matching_cleanup_claim_or_settled(
    parent_fd: int,
    quarantine: Path,
    authoritative_path: Path,
    owner: str,
    owned_identity: os.stat_result,
) -> bool:
    """Return whether another exact cleanup already removed both owned paths."""

    try:
        _read_matching_cleanup_quarantine_claim(
            parent_fd,
            quarantine,
            owner,
            owned_identity,
        )
    except _CleanupQuarantineClaimMissing as missing_claim:
        for path in (authoritative_path, quarantine):
            try:
                os.stat(
                    path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                raise WorkspaceBranchFencedError(
                    "Durable private cleanup settlement is unavailable."
                ) from error
            raise missing_claim
        return True
    return False


def _write_or_require_cleanup_quarantine_claim(
    parent_fd: int,
    quarantine: Path,
    authoritative_path: Path,
    owner: str,
    owned_identity: os.stat_result,
) -> bool:
    claim_path = _cleanup_quarantine_claim_path(quarantine)
    expected = _cleanup_quarantine_claim_bytes(owner, owned_identity)
    staging_name = _cleanup_quarantine_claim_staging_name(quarantine)
    staging_fd = -1
    staging_identity: os.stat_result | None = None
    created_staging = False
    staging_path_present = False
    published = False
    try:
        try:
            staging_fd = os.open(
                staging_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                _PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            try:
                staging_fd = os.open(
                    staging_name,
                    os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                if _matching_cleanup_claim_or_settled(
                    parent_fd,
                    quarantine,
                    authoritative_path,
                    owner,
                    owned_identity,
                ):
                    return True
                published = True
        else:
            created_staging = True
        if staging_fd >= 0:
            _lock_cleanup_claim_staging(staging_fd)
            if created_staging:
                _write_cleanup_claim_staging(staging_fd, expected)
            try:
                staging_identity, staging_path_present = _validate_exact_cleanup_claim_staging(
                    parent_fd,
                    staging_name,
                    staging_fd,
                    expected,
                )
            except WorkspaceBranchFencedError:
                if not _cleanup_claim_staging_is_repairable(
                    parent_fd,
                    claim_path,
                    staging_name,
                    staging_fd,
                    expected,
                ):
                    raise
                _write_cleanup_claim_staging(staging_fd, expected)
                staging_identity, staging_path_present = _validate_exact_cleanup_claim_staging(
                    parent_fd,
                    staging_name,
                    staging_fd,
                    expected,
                )
        if not published and not staging_path_present:
            if _matching_cleanup_claim_or_settled(
                parent_fd,
                quarantine,
                authoritative_path,
                owner,
                owned_identity,
            ):
                return True
            published = True
        elif not published:
            try:
                os.link(
                    staging_name,
                    claim_path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except (FileExistsError, FileNotFoundError):
                if _matching_cleanup_claim_or_settled(
                    parent_fd,
                    quarantine,
                    authoritative_path,
                    owner,
                    owned_identity,
                ):
                    return True
                published = True
            else:
                claim = _read_matching_cleanup_quarantine_claim(
                    parent_fd,
                    quarantine,
                    owner,
                    owned_identity,
                )
                if staging_identity is None or not os.path.samestat(
                    claim.file_identity, staging_identity
                ):
                    raise WorkspaceBranchFencedError(
                        "Durable private cleanup publication changed its staging identity."
                    )
                published = True
    except WorkspaceBranchFencedError:
        raise
    except OSError as error:
        raise WorkspaceBranchFencedError(
            "Durable private cleanup staging ownership is unavailable."
        ) from error
    finally:
        primary = sys.exception()
        cleanup_failures: list[BaseException] = []
        try:
            if staging_identity is not None and (created_staging or published):
                _remove_exact_cleanup_claim_staging(
                    parent_fd,
                    staging_name,
                    staging_identity,
                )
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        try:
            if staging_fd >= 0:
                os.close(staging_fd)
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        try:
            if published:
                os.fsync(parent_fd)
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            cleanup_failure: BaseException = (
                cleanup_failures[0]
                if len(cleanup_failures) == 1
                else BaseExceptionGroup(
                    "Durable private cleanup staging settlement failed.",
                    cleanup_failures,
                )
            )
            if primary is not None:
                _raise_primary_with_reconciliation_failure(primary, cleanup_failure)
            raise cleanup_failure
    return False


def _remove_cleanup_quarantine_claim(
    parent_fd: int,
    quarantine: Path,
    claim: _CleanupQuarantineClaim,
) -> None:
    claim_path = _cleanup_quarantine_claim_path(quarantine)
    try:
        current = os.stat(
            claim_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not os.path.samestat(claim.file_identity, current):
        raise WorkspaceBranchFencedError("Durable private cleanup identity changed.")
    try:
        os.unlink(claim_path.name, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    os.fsync(parent_fd)


def _discard_orphaned_cleanup_quarantine_claim(
    parent_fd: int,
    quarantine: Path,
    authoritative_path: Path,
) -> None:
    try:
        os.stat(quarantine.name, dir_fd=parent_fd, follow_symlinks=False)
        return
    except FileNotFoundError:
        pass
    try:
        os.stat(authoritative_path.name, dir_fd=parent_fd, follow_symlinks=False)
        return
    except FileNotFoundError:
        pass
    claim_path = _cleanup_quarantine_claim_path(quarantine)
    with suppress(FileNotFoundError):
        os.unlink(claim_path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _remove_owned_private_tree(path: Path, owner: str) -> None:
    parent_fd = os.open(path.parent, _PRIVATE_DIRECTORY_OPEN_FLAGS)
    directory_fd = -1
    try:
        quarantine = path.with_name(f"{path.name}.cleanup-{owner}")
        _settle_owned_cleanup_quarantine(
            parent_fd,
            quarantine,
            path,
            owner,
        )
        try:
            directory_fd = os.open(
                path.name,
                _PRIVATE_DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            raise WorkspaceBranchFencedError(
                "Durable private branch ownership is unavailable."
            ) from error
        owned_identity = os.fstat(directory_fd)
        _require_private_root_owner(directory_fd, owner)
        cleanup_already_settled = _write_or_require_cleanup_quarantine_claim(
            parent_fd,
            quarantine,
            path,
            owner,
            owned_identity,
        )
        if cleanup_already_settled:
            return
        try:
            _rename_private_root_no_replace(path, quarantine, parent_fd=parent_fd)
        except FileNotFoundError:
            os.close(directory_fd)
            directory_fd = -1
            _settle_owned_cleanup_quarantine(
                parent_fd,
                quarantine,
                path,
                owner,
            )
            return
        except FileExistsError as error:
            raise WorkspaceBranchFencedError(
                "Durable private cleanup quarantine became occupied."
            ) from error
        try:
            current = os.stat(quarantine.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not os.path.samestat(owned_identity, current):
            claim = _read_cleanup_quarantine_claim(parent_fd, quarantine, owner)
            _restore_unowned_cleanup_quarantine(
                parent_fd,
                quarantine,
                path,
                current,
                claim,
                _CleanupQuarantineReplacement(
                    "Durable private branch ownership changed during quarantine."
                ),
            )
        os.close(directory_fd)
        directory_fd = -1
        _settle_owned_cleanup_quarantine(
            parent_fd,
            quarantine,
            path,
            owner,
        )
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


def _settle_owned_cleanup_quarantine(
    parent_fd: int,
    path: Path,
    authoritative_path: Path,
    owner: str,
) -> None:
    directory_fd = -1
    try:
        try:
            directory_fd = os.open(
                path.name,
                _PRIVATE_DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            _discard_orphaned_cleanup_quarantine_claim(
                parent_fd,
                path,
                authoritative_path,
            )
            return
        except OSError as error:
            raise WorkspaceBranchFencedError(
                "Durable private cleanup quarantine is unavailable."
            ) from error
        owned_identity = os.fstat(directory_fd)
        claim = _read_cleanup_quarantine_claim(parent_fd, path, owner)
        if claim.device != owned_identity.st_dev or claim.inode != owned_identity.st_ino:
            _restore_unowned_cleanup_quarantine(
                parent_fd,
                path,
                authoritative_path,
                owned_identity,
                claim,
                _CleanupQuarantineReplacement(
                    "Durable private cleanup quarantine ownership changed."
                ),
            )
        try:
            _require_private_root_owner(directory_fd, owner)
        except WorkspaceBranchFencedError as ownership_error:
            _remove_empty_owned_directory(
                parent_fd,
                path,
                directory_fd,
                owned_identity,
                detail="Durable private cleanup quarantine ownership is unavailable.",
                ownership_error=ownership_error,
            )
            _remove_cleanup_quarantine_claim(parent_fd, path, claim)
            return
        _remove_private_tree_contents_from_fd(
            directory_fd,
            path=path,
            preserve_owner_marker=True,
        )
        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            _remove_cleanup_quarantine_claim(parent_fd, path, claim)
            return
        if not os.path.samestat(owned_identity, current):
            raise WorkspaceBranchFencedError("Durable private cleanup quarantine changed.")
        try:
            _require_private_root_owner(directory_fd, owner)
        except WorkspaceBranchFencedError as ownership_error:
            _remove_empty_owned_directory(
                parent_fd,
                path,
                directory_fd,
                owned_identity,
                detail="Durable private cleanup quarantine changed.",
                ownership_error=ownership_error,
            )
            _remove_cleanup_quarantine_claim(parent_fd, path, claim)
            return
        try:
            os.unlink(_PRIVATE_ROOT_OWNER_FILE, dir_fd=directory_fd)
        except FileNotFoundError:
            _remove_empty_owned_directory(
                parent_fd,
                path,
                directory_fd,
                owned_identity,
                detail="Durable private cleanup quarantine changed.",
            )
            _remove_cleanup_quarantine_claim(parent_fd, path, claim)
            return
        try:
            os.rmdir(path.name, dir_fd=parent_fd)
        except FileNotFoundError:
            _remove_cleanup_quarantine_claim(parent_fd, path, claim)
            return
        except BaseException as removal_error:
            try:
                _write_private_root_owner_to_fd(directory_fd, owner)
            except BaseException as restoration_error:
                raise removal_error from restoration_error
            raise
        _remove_cleanup_quarantine_claim(parent_fd, path, claim)
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _restore_unowned_cleanup_quarantine(
    parent_fd: int,
    quarantine: Path,
    authoritative_path: Path,
    quarantine_identity: os.stat_result,
    claim: _CleanupQuarantineClaim,
    ownership_error: BaseException,
) -> NoReturn:
    """Restore a raced unowned pathname instead of deleting or stranding it."""

    try:
        _rename_private_root_no_replace(
            quarantine,
            authoritative_path,
            parent_fd=parent_fd,
        )
    except FileExistsError:
        raise ownership_error from None
    except FileNotFoundError:
        raise ownership_error from None
    try:
        restored = os.stat(
            authoritative_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise ownership_error from None
    if not os.path.samestat(quarantine_identity, restored):
        raise WorkspaceBranchFencedError(
            "Durable private cleanup restoration changed ownership."
        ) from ownership_error
    try:
        _remove_cleanup_quarantine_claim(parent_fd, quarantine, claim)
    except BaseException as cleanup_error:
        _raise_primary_with_reconciliation_failure(ownership_error, cleanup_error)
    raise ownership_error


def _remove_empty_owned_directory(
    parent_fd: int,
    path: Path,
    directory_fd: int,
    owned_identity: os.stat_result,
    *,
    detail: str,
    ownership_error: BaseException | None = None,
) -> None:
    with os.scandir(directory_fd) as entries:
        if any(True for _entry in entries):
            if ownership_error is not None:
                raise ownership_error
            raise WorkspaceBranchFencedError(detail)
    try:
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not os.path.samestat(owned_identity, current):
        raise WorkspaceBranchFencedError(detail) from ownership_error
    try:
        os.rmdir(path.name, dir_fd=parent_fd)
    except FileNotFoundError:
        return


def _remove_private_tree_contents_from_fd(
    directory_fd: int,
    *,
    path: Path,
    preserve_owner_marker: bool = False,
) -> None:
    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    if preserve_owner_marker:
        names = [name for name in names if name != _PRIVATE_ROOT_OWNER_FILE]
    for name in names:
        child_path = path / name
        try:
            identity = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(identity.st_mode):
            try:
                child_fd = os.open(
                    name,
                    _PRIVATE_DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                continue
            try:
                opened_identity = os.fstat(child_fd)
                if not os.path.samestat(identity, opened_identity):
                    raise WorkspaceBranchFencedError(
                        f"Durable private branch path changed during cleanup: {child_path}"
                    )
                _remove_private_tree_contents_from_fd(child_fd, path=child_path)
            finally:
                os.close(child_fd)
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not os.path.samestat(opened_identity, current):
                raise WorkspaceBranchFencedError(
                    f"Durable private branch path changed during cleanup: {child_path}"
                )
            try:
                os.rmdir(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            continue
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not os.path.samestat(identity, current):
            raise WorkspaceBranchFencedError(
                f"Durable private branch path changed during cleanup: {child_path}"
            )
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            continue


def _discard_captured_baseline(
    captured: _CapturedBaseline,
    capacity_lease: _BranchCapacityLease,
) -> None:
    _discard_private_tree(
        captured.private_root,
        capacity_lease,
        private_root_owner=captured.private_root_owner,
    )


def _discard_private_tree(
    path: Path,
    capacity_lease: _BranchCapacityLease,
    *,
    private_root_owner: str | None = None,
) -> None:
    try:
        _settle_private_tree_cleanup(
            path,
            capacity_lease,
            private_root_owner=private_root_owner,
        )
    except BaseException:
        _retain_private_tree_cleanup(
            path,
            capacity_lease,
            private_root_owner=private_root_owner,
        )
        raise


def _retain_source_staging_cleanup(
    error: _LocalGuardStagingCleanupError,
    *,
    source_key: tuple[object, ...],
) -> None:
    key = (source_key, error.staging_name)
    inserted, _retained = _SOURCE_STAGING_CLEANUPS.retain_or_existing(
        key,
        source_key=source_key,
        payload=error,
        replace_unclaimed=lambda retained: not retained.cleanup_owned and error.cleanup_owned,
    )
    if not inserted:
        error.release_cleanup_owner()
    _schedule_source_staging_cleanup(key)


def _schedule_source_staging_cleanup(
    key: tuple[tuple[object, ...], str],
) -> None:
    _schedule_retained_cleanup(
        _SOURCE_STAGING_CLEANUPS,
        key,
        _retry_source_staging_cleanup,
        eligible=lambda retained: retained.cleanup_owned,
    )


def _retry_source_staging_cleanup(key: tuple[tuple[object, ...], str]) -> None:
    error = _SOURCE_STAGING_CLEANUPS.payload(key)
    if error is None:
        return
    settled = error.retry_cleanup()
    if settled:
        _SOURCE_STAGING_CLEANUPS.forget(key)
        return
    _SOURCE_STAGING_CLEANUPS.release_claim(key)
    _schedule_source_staging_cleanup(key)


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
    path, capacity_lease, private_root_owner = retained
    try:
        _settle_private_tree_cleanup(
            path,
            capacity_lease,
            private_root_owner=private_root_owner,
        )
    except BaseException as cleanup_error:
        _PRIVATE_TREE_CLEANUPS.release_claim(key)
        _schedule_private_tree_cleanup(path, capacity_lease)
        if not isinstance(cleanup_error, Exception):
            raise


def _settle_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
    *,
    private_root_owner: str | None = None,
) -> None:
    """Join the exact lease owner before removing and releasing one private tree."""

    with capacity_lease._cleanup_settlement_lock:
        if private_root_owner is None:
            _remove_private_tree(path)
        else:
            _remove_private_tree(path, private_root_owner=private_root_owner)
        capacity_lease.release_after_cleanup()
        _forget_private_tree_cleanup(path, capacity_lease)


def _retain_private_tree_cleanup(
    path: Path,
    capacity_lease: _BranchCapacityLease,
    *,
    private_root_owner: str | None = None,
) -> None:
    capacity_lease.retain_for_cleanup()
    key = _private_tree_cleanup_key(path, capacity_lease)
    _PRIVATE_TREE_CLEANUPS.retain(
        key,
        source_key=capacity_lease._source_key,
        payload=(path, capacity_lease, private_root_owner),
    )
    _schedule_private_tree_cleanup(path, capacity_lease)


def _retry_pending_branch_cleanups(source_key: tuple[object, ...]) -> None:
    _retry_pending_private_tree_cleanups(source_key)
    _retry_pending_source_staging_cleanups(source_key)


async def _retry_pending_durable_terminal_settlements(
    source_key: tuple[object, ...],
) -> None:
    """Assist process-local terminal owners before admitting replacement work."""

    await _retry_pending_durable_recovery_cleanups(source_key)
    pending_keys = _DURABLE_TERMINAL_SETTLEMENTS.pending_keys(source_key)
    for key in pending_keys:
        if not _DURABLE_TERMINAL_SETTLEMENTS.claim(key):
            continue
        retained = _DURABLE_TERMINAL_SETTLEMENTS.payload(key)
        if retained is None:  # pragma: no cover - claim invariant
            continue
        try:
            await _retry_durable_terminal_settlement(retained)
            _release_durable_terminal_settlement(retained.branch)
        except BaseException:
            _DURABLE_TERMINAL_SETTLEMENTS.release_claim(key)
            raise


async def _retry_pending_durable_recovery_cleanups(
    source_key: tuple[object, ...],
) -> None:
    pending_keys = _DURABLE_RECOVERY_CLEANUPS.pending_keys(source_key)
    for key in pending_keys:
        if not _DURABLE_RECOVERY_CLEANUPS.claim(key):
            continue
        retained = _DURABLE_RECOVERY_CLEANUPS.payload(key)
        if retained is None:  # pragma: no cover - claim invariant
            continue
        try:
            await _settle_durable_recovery_private_tree(
                retained.source,
                retained.record,
                binding_claim_lease=retained.binding_claim_lease,
            )
        except BaseException:
            _DURABLE_RECOVERY_CLEANUPS.release_claim(key)
            raise


async def _retry_durable_terminal_settlement(
    retained: _RetainedDurableTerminalSettlement,
) -> None:
    branch = retained.branch
    durable = branch._durable
    if durable is None:  # pragma: no cover - retained-owner invariant
        raise AssertionError("Retained durable terminal owner lost its coordinator.")
    if retained.publication_expected is None:
        terminal = await _load_durable_branch_record(
            durable.store,
            session_id=retained.terminal_record.authority.session_id,
            storage_key=durable.storage_key,
        )
        if terminal is None:
            raise WorkspaceBranchOperationConflict(
                "Retained durable terminal evidence changed before cleanup."
            )
        if terminal != retained.terminal_record:
            if terminal.state not in {
                WorkspaceBranchDurableState.COMMITTED,
                WorkspaceBranchDurableState.ROLLED_BACK,
                WorkspaceBranchDurableState.EXPIRED,
                WorkspaceBranchDurableState.FAILED,
            }:
                raise WorkspaceBranchOperationConflict(
                    "Retained durable terminal evidence changed before cleanup."
                )
            await branch._settle_observed_durable_terminal_handle(
                terminal,
                binding_claim_lease=retained.binding_claim_lease,
            )
            return
    else:
        terminal = await _publish_or_reconcile_exact_record(
            durable.store,
            session_id=retained.terminal_record.authority.session_id,
            storage_key=durable.storage_key,
            expected=retained.publication_expected,
            updated=retained.terminal_record,
            expected_run_epoch=retained.terminal_record.authority.expected_run_epoch,
        )
    await branch._settle_observed_durable_terminal_handle(
        terminal,
        binding_claim_lease=retained.binding_claim_lease,
    )


def _retry_pending_source_staging_cleanups(source_key: tuple[object, ...]) -> None:
    pending = _SOURCE_STAGING_CLEANUPS.claim_pending(
        source_key,
        eligible=lambda retained: retained.cleanup_owned,
    )
    for key, _payload in pending:
        _retry_source_staging_cleanup(key)


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


def _durable_terminal_settlement_key(
    branch: LocalWorkspaceBranch,
) -> tuple[Path, int]:
    return branch._private_root, id(branch._capacity_lease)


def _release_durable_terminal_settlement(branch: LocalWorkspaceBranch) -> None:
    key = _durable_terminal_settlement_key(branch)
    retained = _DURABLE_TERMINAL_SETTLEMENTS.payload(key)
    if retained is not None:
        retained.binding_claim_lease.release_after_settlement()
        _DURABLE_TERMINAL_SETTLEMENTS.forget(key)


async def _await_owned_thread(
    operation: Callable[..., _ResultT],
    *args: object,
    abandon_result: Callable[[_ResultT], None] | None = None,
    settle_result: Callable[[_ResultT], None] | None = None,
    pre_dispatch_checkpoint: bool = True,
) -> _ResultT:
    """Do not release branch ownership while a cancelled to_thread call still runs."""

    if pre_dispatch_checkpoint:
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
    if settle_result is not None:
        try:
            settle_result(result)
        except BaseException as settlement_error:
            if caller_signal is not None:
                raise caller_signal from settlement_error
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

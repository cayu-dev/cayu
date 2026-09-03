"""Crash-recoverable publication for generator edits across independent paths.

This owner deliberately does not reuse the whole-tree publication policy.  A
generator plan edits independent paths in a user-owned repository, so recovery
must authenticate every before/after image instead of treating the repository as
one replaceable tree.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, cast

from cayu._filesystem_lock import cooperative_path_lock
from cayu.cli._guarded_tree_publication import (
    GuardedTreePublicationError,
    _assert_windows_directory_dacl_is_protected,
    _capture_stable_identity,
    _capture_tree_authority,
    _CleanupEntry,
    _close_descriptor,
    _create_private_windows_directory,
    _delete_windows_entry_by_handle,
    _Identity,
    _is_unsafe_windows_component,
    _is_windows_reparse_point,
    _Parent,
    _pinned_parent,
    _PreparedStageDirectory,
    _PreparedStageFile,
    _remove_directory_contents_from_fd,
    _restore_windows_directory_inheritance,
    _sync_windows_path,
    _windows_directory_namespace_fence,
    _write_stage_files,
)
from cayu.runtime.manifest import APP_MANIFEST_SCHEMA_VERSION

_SCHEMA_VERSION = 1
_STATE_DIRECTORY = "generator-transactions"
_ACTIVE_DIRECTORY = "active"
_RECEIPT_FILE = "receipt.json"
_NEXT_RECEIPT_FILE = "next-receipt.json"
_PREVIOUS_RECEIPT_FILE = "previous-receipt.json"
_JOURNAL_FILE = "journal.jsonl"
_OWNER_FILE = "owner.json"
_CLEANUP_CLAIM_FILE = "cleanup-claim.json"
_PREPARE_PATTERN = re.compile(r"\Aprepare-(?P<token>[0-9a-f]{64})\Z")
_PREPARATION_OWNER_PATTERN = re.compile(r"\Aprepare-(?P<token>[0-9a-f]{64})\.owner\.json\Z")
_CLEANUP_PATTERN = re.compile(r"\Acleanup-(?P<token>[0-9a-f]{64})\Z")
_CLEANUP_CLAIM_PATTERN = re.compile(r"\Acleanup-(?P<token>[0-9a-f]{64})\.claim\.json\Z")
_CLEANUP_OWNER_PATTERN = re.compile(r"\Acleanup-(?P<token>[0-9a-f]{64})\.owner\.json\Z")
_MAX_EDITS = 256
_MAX_PRECONDITIONS = 1024
_MAX_PATH_BYTES = 1024
_MAX_PATH_DEPTH = 128
_MAX_STAGED_BYTES = 64 * 1024 * 1024
_MAX_JOURNAL_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_CLEANUP_CLAIM_BYTES = 16 * 1024 * 1024
_MAX_STATE_ENTRIES = 1024
_MAX_CREATED_TREE_ENTRIES = _MAX_EDITS * _MAX_PATH_DEPTH
_MAX_PRIVATE_TREE_ENTRIES = 16_384
_MAX_IDENTITY_DEVICE = (1 << 128) - 1
_MAX_IDENTITY_INODE = (1 << 128) - 1
_MAX_IDENTITY_KIND = 0o170000
_MAX_IDENTITY_INCARNATION = (1 << 256) - 1
_MAX_DIAGNOSTIC_PATHS = 8
_MAX_DIAGNOSTIC_PATH_LENGTH = 256
_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)
_LOCK_DIRECTORY = "cayu-generator-transaction-locks"


class GeneratorTransactionError(RuntimeError):
    """A generator transaction could not prove a safe filesystem outcome."""

    def __init__(self, code: str, message: str, *, paths: tuple[str, ...] = ()) -> None:
        self.code = code
        self.paths = tuple(
            path[:_MAX_DIAGNOSTIC_PATH_LENGTH] for path in paths[:_MAX_DIAGNOSTIC_PATHS]
        )
        super().__init__(message)


def encode_generator_transaction_content(value: str, *, remaining_bytes: int) -> bytes:
    """Encode one plan body without first allocating an over-limit UTF-8 value."""

    if (
        not isinstance(value, str)
        or type(remaining_bytes) is not int
        or not 0 <= remaining_bytes <= _MAX_STAGED_BYTES
    ):
        raise GeneratorTransactionError(
            "invalid_content",
            "Generator edit content has an unsupported type or byte budget.",
        )
    chunks: list[bytes] = []
    total = 0
    if len(value) > remaining_bytes:
        raise GeneratorTransactionError(
            "staged_content_limit",
            "Generator plan exceeds the aggregate staged-content limit.",
        )
    try:
        for offset in range(0, len(value), 64 * 1024):
            encoded = value[offset : offset + 64 * 1024].encode("utf-8")
            total += len(encoded)
            if total > remaining_bytes:
                raise GeneratorTransactionError(
                    "staged_content_limit",
                    "Generator plan exceeds the aggregate staged-content limit.",
                )
            chunks.append(encoded)
    except UnicodeEncodeError as exc:
        raise GeneratorTransactionError(
            "invalid_content",
            "Generator edit content is not valid UTF-8.",
        ) from exc
    return b"".join(chunks)


def generator_transaction_staged_byte_limit() -> int:
    """Return the internal aggregate plan-content bound for the generator facade."""

    return _MAX_STAGED_BYTES


def validate_generator_transaction_collection_bounds(
    *,
    edit_count: int,
    precondition_count: int,
) -> None:
    """Reject an oversized public plan before its values are defensively copied."""

    if type(edit_count) is not int or not 1 <= edit_count <= _MAX_EDITS:
        raise GeneratorTransactionError(
            "edit_limit",
            f"Generator plans must contain between 1 and {_MAX_EDITS} edits.",
        )
    if type(precondition_count) is not int or not 0 <= precondition_count <= _MAX_PRECONDITIONS:
        raise GeneratorTransactionError(
            "precondition_limit",
            "Generator plan exceeds the precondition-count limit.",
        )


@dataclass(frozen=True)
class GeneratorTransactionEdit:
    path: str
    operation: Literal["create", "update_region"]
    content: bytes
    content_sha256: str
    preimage_sha256: str | None


@dataclass(frozen=True)
class GeneratorTransactionPrecondition:
    path: str
    content_sha256: str


@dataclass(frozen=True)
class GeneratorTransactionRequest:
    schema_version: str
    slice_name: str
    tool_name: str
    effect: str
    authoring_state: str
    edits: tuple[GeneratorTransactionEdit, ...]
    preconditions: tuple[GeneratorTransactionPrecondition, ...]
    verification_commands: tuple[str, ...]

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "status": "ready",
            "slice_name": self.slice_name,
            "tool_name": self.tool_name,
            "effect": self.effect,
            "authoring_state": self.authoring_state,
            "edits": [
                {
                    "path": edit.path,
                    "operation": edit.operation,
                    "content_sha256": edit.content_sha256,
                    "preimage_sha256": edit.preimage_sha256,
                }
                for edit in self.edits
            ],
            "preconditions": [
                {
                    "path": precondition.path,
                    "content_sha256": precondition.content_sha256,
                }
                for precondition in self.preconditions
            ],
            "verification_commands": list(self.verification_commands),
        }
        return _sha256(_canonical_json(payload))


class _Phase(StrEnum):
    PREPARED = "prepared"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class _UnexpectedRenamePolicy(StrEnum):
    RESTORE_SOURCE = "restore_source"
    PRESERVE_DESTINATION = "preserve_destination"


@dataclass(frozen=True)
class _FileSnapshot:
    identity: _Identity
    content_sha256: str
    mode: int
    size: int

    def payload(self) -> dict[str, object]:
        return {
            "identity": self.identity.as_json(),
            "content_sha256": self.content_sha256,
            "mode": self.mode,
            "size": self.size,
        }

    @classmethod
    def parse(cls, value: object, *, field: str) -> _FileSnapshot:
        if not isinstance(value, dict) or set(value) != {
            "identity",
            "content_sha256",
            "mode",
            "size",
        }:
            raise _invalid_record(f"invalid {field} file snapshot")
        value = cast("dict[str, object]", value)
        identity = _identity_from_json(value["identity"], field=f"{field}.identity")
        if identity.kind != stat.S_IFREG:
            raise _invalid_record(f"invalid {field} file identity")
        digest = value["content_sha256"]
        mode = value["mode"]
        size = value["size"]
        if not _is_sha256(digest):
            raise _invalid_record(f"invalid {field} content digest")
        if type(mode) is not int or not 0 <= mode <= 0o7777:
            raise _invalid_record(f"invalid {field} mode")
        if type(size) is not int or not 0 <= size <= _MAX_STAGED_BYTES:
            raise _invalid_record(f"invalid {field} size")
        return cls(
            identity=identity,
            content_sha256=cast("str", digest),
            mode=mode,
            size=size,
        )


@dataclass(frozen=True)
class _EditRecord:
    index: int
    path: str
    operation: Literal["create", "update_region"]
    before: _FileSnapshot | None
    after: _FileSnapshot
    parent_identity: _Identity
    created_root_index: int | None
    stage_name: str
    backup_name: str
    quarantine_name: str

    def payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "path": self.path,
            "operation": self.operation,
            "before": None if self.before is None else self.before.payload(),
            "after": self.after.payload(),
            "parent_identity": self.parent_identity.as_json(),
            "created_root_index": self.created_root_index,
            "stage_name": self.stage_name,
            "backup_name": self.backup_name,
            "quarantine_name": self.quarantine_name,
        }


@dataclass(frozen=True)
class _PreconditionRecord:
    path: str
    snapshot: _FileSnapshot
    parent_identity: _Identity

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "snapshot": self.snapshot.payload(),
            "parent_identity": self.parent_identity.as_json(),
        }


@dataclass(frozen=True)
class _TreeSnapshot:
    identity: _Identity
    fingerprint: str
    entries: int

    def payload(self) -> dict[str, object]:
        return {
            "identity": self.identity.as_json(),
            "fingerprint": self.fingerprint,
            "entries": self.entries,
        }

    @classmethod
    def parse(cls, value: object, *, field: str) -> _TreeSnapshot:
        if not isinstance(value, dict) or set(value) != {
            "identity",
            "fingerprint",
            "entries",
        }:
            raise _invalid_record(f"invalid {field} tree snapshot")
        value = cast("dict[str, object]", value)
        fingerprint = value["fingerprint"]
        entries = value["entries"]
        if not _is_sha256(fingerprint):
            raise _invalid_record(f"invalid {field} tree fingerprint")
        if type(entries) is not int or not 1 <= entries <= _MAX_CREATED_TREE_ENTRIES:
            raise _invalid_record(f"invalid {field} tree entry count")
        identity = _identity_from_json(value["identity"], field=f"{field}.identity")
        if identity.kind != stat.S_IFDIR:
            raise _invalid_record(f"invalid {field} tree identity")
        return cls(
            identity=identity,
            fingerprint=cast("str", fingerprint),
            entries=entries,
        )


@dataclass(frozen=True)
class _CreatedRootRecord:
    index: int
    path: str
    stage_name: str
    parent_identity: _Identity
    after: _TreeSnapshot
    edit_indexes: tuple[int, ...]

    def payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "path": self.path,
            "stage_name": self.stage_name,
            "parent_identity": self.parent_identity.as_json(),
            "after": self.after.payload(),
            "edit_indexes": list(self.edit_indexes),
        }


@dataclass(frozen=True)
class _PrivateIdentities:
    state: _Identity
    new: _Identity
    trees: _Identity
    backup: _Identity
    quarantine: _Identity

    def payload(self) -> dict[str, object]:
        return {
            "state": self.state.as_json(),
            "new": self.new.as_json(),
            "trees": self.trees.as_json(),
            "backup": self.backup.as_json(),
            "quarantine": self.quarantine.as_json(),
        }


@dataclass
class _Transaction:
    directory: Path
    directory_identity: _Identity
    journal_identity: _Identity
    private_identities: _PrivateIdentities
    token: str
    request_digest: str
    root_identity: _Identity
    edits: tuple[_EditRecord, ...]
    preconditions: tuple[_PreconditionRecord, ...]
    created_roots: tuple[_CreatedRootRecord, ...]
    prior_receipt: _FileSnapshot | None
    receipt_after: _FileSnapshot
    phase: _Phase
    sequence: int
    entry_sha256: str
    valid_bytes: int


@dataclass(frozen=True)
class _Receipt:
    request_digest: str
    root_identity: _Identity
    edits: tuple[_PreconditionRecord, ...]
    preconditions: tuple[_PreconditionRecord, ...]
    created_roots: tuple[tuple[str, _Identity, _TreeSnapshot], ...]


@dataclass(frozen=True)
class _CreatedRootPlan:
    path: str
    parent_identity: _Identity
    edit_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _RequestAuthority:
    before: tuple[_FileSnapshot | None, ...]
    parent_identities: tuple[_Identity | None, ...]
    preconditions: tuple[_PreconditionRecord, ...]
    created_roots: tuple[_CreatedRootPlan, ...]


@dataclass(frozen=True)
class _PreparationAuthority:
    token: str
    request_digest: str
    root_identity: _Identity
    transaction_identity: _Identity
    marker: _FileSnapshot


@dataclass(frozen=True)
class _CleanupClaim:
    owner_kind: Literal["preparation", "transaction"]
    token: str
    request_digest: str
    root_identity: _Identity
    transaction_identity: _Identity
    phase: str
    preparation_owner: _FileSnapshot
    journal: _FileSnapshot | None
    transaction_owner: _FileSnapshot | None
    entries: tuple[_CleanupEntry, ...]
    identity: _Identity


def _fault(phase: str) -> None:
    """Test seam for real-process interruption at durable transitions."""


@contextmanager
def generator_planning_guard(root: Path) -> Iterator[None]:
    """Hold a read-side project fence while one generator plan is constructed."""

    root = _canonical_root(root)
    with cooperative_path_lock(
        root,
        _STATE_DIRECTORY,
        lock_directory_name=_LOCK_DIRECTORY,
        shared=True,
    ):
        root_identity = _capture_directory_identity(root, label="project root")
        _require_root(root, root_identity)
        state = _state_directory(root, create=False)
        if state is not None:
            _require_state_census(state)
            if (
                _entry_exists(state / _ACTIVE_DIRECTORY)
                or _prepare_directories(state)
                or _cleanup_directories(state)
                or _cleanup_claim_files(state)
                or _cleanup_owner_files(state)
                or _preparation_owner_files(state)
            ):
                raise GeneratorTransactionError(
                    "recovery_required",
                    "A pending generator transaction must be recovered before planning.",
                )
        yield
        _require_root(root, root_identity)


def recover_generator_transaction(root: Path, *, dry_run: bool = False) -> bool:
    """Recover exact pending work; return whether private work was present."""

    root = _canonical_root(root)
    with cooperative_path_lock(
        root,
        _STATE_DIRECTORY,
        lock_directory_name=_LOCK_DIRECTORY,
        shared=dry_run,
    ):
        root_identity = _capture_directory_identity(root, label="project root")
        state = _state_directory(root, create=False)
        if state is None:
            return False
        _require_state_census(state)
        pending = (
            bool(_prepare_directories(state))
            or bool(_cleanup_directories(state))
            or bool(_cleanup_claim_files(state))
            or bool(_cleanup_owner_files(state))
            or bool(_preparation_owner_files(state))
            or _entry_exists(state / _ACTIVE_DIRECTORY)
        )
        if not pending:
            return False
        if dry_run:
            raise GeneratorTransactionError(
                "recovery_required",
                "A pending generator transaction requires recovery; rerun without --dry-run.",
            )
        _settle_cleanups(state)
        _settle_preparations(root, state, root_identity)
        active = state / _ACTIVE_DIRECTORY
        if _entry_exists(active):
            transaction = _load_transaction(active)
            _require_root(root, root_identity)
            if transaction.root_identity != root_identity:
                raise _conflict(
                    "The pending generator transaction belongs to another project root."
                )
            _recover_transaction(root, state, transaction)
        _require_no_orphan_owner_markers(state)
        return True


def apply_generator_transaction(root: Path, request: GeneratorTransactionRequest) -> None:
    """Apply one exact request or reconcile its exact terminal receipt."""

    root = _canonical_root(root)
    _validate_request(request)
    with cooperative_path_lock(
        root,
        _STATE_DIRECTORY,
        lock_directory_name=_LOCK_DIRECTORY,
    ):
        root_identity = _capture_directory_identity(root, label="project root")
        state = _state_directory(root, create=False)
        if state is not None:
            _require_state_census(state)
            _settle_cleanups(state)
            _settle_preparations(root, state, root_identity)
            active_path = state / _ACTIVE_DIRECTORY
            if _entry_exists(active_path):
                active = _load_transaction(active_path)
                if active.root_identity != root_identity:
                    raise _conflict(
                        "The active generator transaction belongs to another project root."
                    )
                _recover_transaction(root, state, active)
            _require_no_orphan_owner_markers(state)
            receipt = _load_receipt(state, required=False)
            if receipt is not None and receipt.request_digest == request.digest:
                _require_receipt_state(root, root_identity, receipt, request=request)
                return

        _require_root(root, root_identity)
        # Reject a stale or unsafe request before allocating private state. The
        # preparation repeats this census after allocation so a concurrent
        # non-generator writer cannot consume the earlier observation.
        authority = _capture_request_authority(root, request)
        _validate_durable_representation_bounds(
            root_identity,
            request,
            authority,
        )
        if state is None:
            state = _state_directory(root, create=True)
            assert state is not None
            _require_state_census(state)
        prior_receipt = _snapshot_optional(
            state / _RECEIPT_FILE,
            label="previous generator receipt",
        )
        prepared = _prepare_transaction(
            root,
            state,
            root_identity,
            request,
            prior_receipt=prior_receipt,
        )
        transaction = prepared
        primary: BaseException | None = None
        try:
            transaction = _promote_preparation(state, transaction)
            _append_phase(transaction, _Phase.COMMITTING)
            _commit_transaction(root, state, transaction)
            _append_phase(transaction, _Phase.COMMITTED)
            _publish_receipt(root, state, transaction)
            _cleanup_transaction(state, transaction)
            return
        except BaseException as exc:
            primary = exc

        assert primary is not None
        settlement_errors: list[BaseException] = []
        try:
            transaction = _reconcile_transaction_after_failure(state, transaction)
            if transaction.phase is _Phase.PREPARED:
                _require_prepared(root, transaction)
                _cleanup_transaction(state, transaction)
            elif transaction.phase is _Phase.COMMITTED:
                _require_committed(root, transaction)
                _publish_receipt(root, state, transaction)
                # Once cleanup has been durably claimed, that directory is the
                # sole retry owner.  Do not consume it again in the same call
                # that is reporting a cleanup failure; recovery will settle it
                # before the next plan or apply operation.
                if transaction.directory.name == _ACTIVE_DIRECTORY:
                    _cleanup_transaction(state, transaction)
            else:
                _append_phase(transaction, _Phase.ROLLING_BACK)
                _rollback_transaction(root, transaction)
                _append_phase(transaction, _Phase.ROLLED_BACK)
                _cleanup_transaction(state, transaction)
        except BaseException as settlement_error:
            settlement_errors.append(settlement_error)
        _raise_primary(primary, settlement_errors)


def _reconcile_transaction_after_failure(
    state: Path,
    transaction: _Transaction,
) -> _Transaction:
    candidates = tuple(
        dict.fromkeys(
            (
                transaction.directory,
                state / _ACTIVE_DIRECTORY,
                state / f"prepare-{transaction.token}",
                state / f"cleanup-{transaction.token}",
            )
        )
    )
    present: list[Path] = []
    for candidate in candidates:
        if not _entry_exists(candidate):
            continue
        if (
            _capture_directory_identity(candidate, label="generator transaction")
            != transaction.directory_identity
        ):
            raise _conflict(
                "Generator transaction namespace changed during failure settlement.",
                paths=(candidate.name,),
            )
        present.append(candidate)
    if not present:
        return transaction
    if len(present) != 1:
        raise _conflict(
            "Generator transaction authority appears in multiple namespaces.",
            paths=tuple(path.name for path in present),
        )
    loaded = _load_transaction(present[0])
    if (
        loaded.directory_identity != transaction.directory_identity
        or loaded.journal_identity != transaction.journal_identity
        or loaded.private_identities != transaction.private_identities
        or loaded.token != transaction.token
        or loaded.request_digest != transaction.request_digest
        or loaded.root_identity != transaction.root_identity
        or loaded.edits != transaction.edits
        or loaded.preconditions != transaction.preconditions
        or loaded.created_roots != transaction.created_roots
        or loaded.prior_receipt != transaction.prior_receipt
        or loaded.receipt_after != transaction.receipt_after
    ):
        raise _conflict("Generator transaction authority changed during failure settlement.")
    return loaded


def _prepare_transaction(
    root: Path,
    state: Path,
    root_identity: _Identity,
    request: GeneratorTransactionRequest,
    *,
    prior_receipt: _FileSnapshot | None,
) -> _Transaction:
    _require_root(root, root_identity)
    authorities = _capture_request_authority(root, request)
    token = secrets.token_hex(32)
    state_identity = _capture_directory_identity(state, label="generator transaction state")
    prepare = state / f"prepare-{token}"
    prepare_identity = _create_owned_directory(
        prepare,
        expected_parent=state_identity,
        mode=0o700,
        label="generator preparation",
    )
    _fault("preparation_directory_allocated")
    _write_private_file(
        _preparation_owner_path(state, token),
        _preparation_owner_content(
            token=token,
            request_digest=request.digest,
            root_identity=root_identity,
            transaction_identity=prepare_identity,
        ),
        expected_parent=state_identity,
    )
    _sync_directory(state)
    _fault("preparation_directory_created")
    new_root = prepare / "new"
    trees_root = prepare / "trees"
    backup_root = prepare / "backup"
    quarantine_root = prepare / "quarantine"
    root_index_by_edit: dict[int, int] = {}
    directory_modes: dict[tuple[str, ...], int | None] = {
        ("new",): 0o700,
        ("trees",): 0o700,
        ("backup",): 0o700,
        ("quarantine",): 0o700,
    }
    prepared_files: list[_PreparedStageFile] = []
    for root_index, root_plan in enumerate(authorities.created_roots):
        root_prefix = ("trees", f"{root_index:04d}")
        directory_modes.setdefault(root_prefix, None)
        for edit_index in root_plan.edit_indexes:
            root_index_by_edit[edit_index] = root_index
            edit = request.edits[edit_index]
            relative = PurePosixPath(edit.path).relative_to(PurePosixPath(root_plan.path))
            for depth in range(1, len(relative.parts)):
                directory_modes.setdefault((*root_prefix, *relative.parts[:depth]), None)
            prepared_files.append(
                _PreparedStageFile(
                    path=PurePosixPath(*root_prefix, *relative.parts),
                    content=edit.content,
                    mode=None,
                )
            )
    for index, (edit, before) in enumerate(zip(request.edits, authorities.before, strict=True)):
        if index in root_index_by_edit:
            continue
        prepared_files.append(
            _PreparedStageFile(
                path=PurePosixPath("new", f"{index:04d}"),
                content=edit.content,
                mode=None if before is None else before.mode,
            )
        )
    prepared_directories = tuple(
        _PreparedStageDirectory(path=PurePosixPath(*parts), mode=mode)
        for parts, mode in sorted(
            directory_modes.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )
    try:
        _write_stage_files(
            prepare,
            expected=prepare_identity,
            files=tuple(sorted(prepared_files, key=lambda item: item.path.as_posix())),
            directories=prepared_directories,
            root_mode=0o700,
        )
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc
    private_identities = _PrivateIdentities(
        state=state_identity,
        new=_capture_directory_identity(new_root, label="generator new-file stage"),
        trees=_capture_directory_identity(trees_root, label="generator tree stage"),
        backup=_capture_directory_identity(backup_root, label="generator backup stage"),
        quarantine=_capture_directory_identity(
            quarantine_root,
            label="generator rollback quarantine",
        ),
    )
    created_root_stages = {
        root_index: trees_root / f"{root_index:04d}"
        for root_index in range(len(authorities.created_roots))
    }
    edits: list[_EditRecord] = []
    total = 0
    try:
        for index, (edit, before, recorded_parent) in enumerate(
            zip(
                request.edits,
                authorities.before,
                authorities.parent_identities,
                strict=True,
            )
        ):
            stage_name = f"{index:04d}"
            created_root_index = root_index_by_edit.get(index)
            if created_root_index is None:
                stage_path = new_root / stage_name
            else:
                root_plan = authorities.created_roots[created_root_index]
                relative = PurePosixPath(edit.path).relative_to(PurePosixPath(root_plan.path))
                stage_path = created_root_stages[created_root_index].joinpath(*relative.parts)
            after = _snapshot_regular(stage_path, label=f"staged {edit.path}")
            total += after.size
            if total > _MAX_STAGED_BYTES:
                raise GeneratorTransactionError(
                    "staged_content_limit",
                    "Generator plan exceeds the aggregate staged-content limit.",
                )
            if after.content_sha256 != edit.content_sha256:
                raise GeneratorTransactionError(
                    "staged_content_changed",
                    "Staged generator content does not match its planned digest.",
                    paths=(edit.path,),
                )
            if created_root_index is None:
                _sync_file(stage_path, expected=after.identity)
            parent_identity = (
                recorded_parent
                if recorded_parent is not None
                else _capture_directory_identity(
                    stage_path.parent,
                    label=f"staged parent of {edit.path}",
                )
            )
            edits.append(
                _EditRecord(
                    index=index,
                    path=edit.path,
                    operation=edit.operation,
                    before=before,
                    after=after,
                    parent_identity=parent_identity,
                    created_root_index=created_root_index,
                    stage_name=stage_name,
                    backup_name=stage_name,
                    quarantine_name=stage_name,
                )
            )
        for stage in created_root_stages.values():
            _sync_tree(stage)
        created_roots = tuple(
            _CreatedRootRecord(
                index=index,
                path=plan.path,
                stage_name=f"{index:04d}",
                parent_identity=plan.parent_identity,
                after=_snapshot_tree(
                    created_root_stages[index],
                    label=f"staged created root {plan.path}",
                ),
                edit_indexes=plan.edit_indexes,
            )
            for index, plan in enumerate(authorities.created_roots)
        )
        receipt_content = _receipt_content(
            request_digest=request.digest,
            root_identity=root_identity,
            edits=tuple(edits),
            preconditions=authorities.preconditions,
            created_roots=created_roots,
        )
        next_receipt = prepare / _NEXT_RECEIPT_FILE
        _write_private_file(
            next_receipt,
            receipt_content,
            expected_parent=prepare_identity,
        )
        receipt_after = _snapshot_regular(
            next_receipt,
            label="staged generator receipt",
        )
        _sync_directory(new_root)
        _sync_directory(trees_root)
        _sync_directory(backup_root)
        _sync_directory(quarantine_root)
        _sync_directory(prepare)
        manifest_payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "manifest",
            "sequence": 0,
            "previous_sha256": None,
            "token": token,
            "request_digest": request.digest,
            "root_identity": root_identity.as_json(),
            "transaction_identity": prepare_identity.as_json(),
            "private_identities": private_identities.payload(),
            "phase": _Phase.PREPARED.value,
            "edits": [edit.payload() for edit in edits],
            "preconditions": [item.payload() for item in authorities.preconditions],
            "created_roots": [item.payload() for item in created_roots],
            "prior_receipt": (None if prior_receipt is None else prior_receipt.payload()),
            "receipt_after": receipt_after.payload(),
        }
        journal_path = prepare / _JOURNAL_FILE
        entry, entry_sha256 = _journal_entry(manifest_payload)
        if len(entry) > _MAX_JOURNAL_BYTES:
            raise GeneratorTransactionError(
                "journal_limit",
                "Generator transaction manifest exceeds its encoded-size limit.",
            )
        _write_private_file(journal_path, entry, expected_parent=prepare_identity)
        journal_identity = _capture_file_identity(journal_path, label="generator journal")
        owner_payload = {
            "schema_version": _SCHEMA_VERSION,
            "token": token,
            "transaction_identity": prepare_identity.as_json(),
            "journal_identity": journal_identity.as_json(),
            "manifest_sha256": entry_sha256,
        }
        _write_private_file(
            prepare / _OWNER_FILE,
            _canonical_json(owner_payload) + b"\n",
            expected_parent=prepare_identity,
        )
        _sync_directory(prepare)
        _sync_directory(state)
        _fault("preparation_synced")
        return _Transaction(
            directory=prepare,
            directory_identity=prepare_identity,
            journal_identity=journal_identity,
            private_identities=private_identities,
            token=token,
            request_digest=request.digest,
            root_identity=root_identity,
            edits=tuple(edits),
            preconditions=authorities.preconditions,
            created_roots=created_roots,
            prior_receipt=prior_receipt,
            receipt_after=receipt_after,
            phase=_Phase.PREPARED,
            sequence=0,
            entry_sha256=entry_sha256,
            valid_bytes=len(entry),
        )
    except BaseException as primary:
        # Before promotion no source path can have changed.  An incomplete
        # preparation has no sealed inventory, so fail closed and preserve it
        # instead of adopting raced content into a recursive cleanup claim.
        _raise_primary(primary, [])


def _promote_preparation(state: Path, transaction: _Transaction) -> _Transaction:
    active = state / _ACTIVE_DIRECTORY
    _rename_no_replace(
        transaction.directory,
        active,
        expected=transaction.directory_identity,
        expected_source_parent=transaction.private_identities.state,
        expected_destination_parent=transaction.private_identities.state,
        label="generator preparation",
    )
    _fault("preparation_promoted_before_sync")
    transaction.directory = active
    _sync_directory(state)
    _fault("preparation_promoted")
    return transaction


def _commit_transaction(root: Path, state: Path, transaction: _Transaction) -> None:
    del state
    _require_root(root, transaction.root_identity)
    classifications = _classify_transaction(root, transaction)
    _require_commit_prefix(classifications)
    applied_paths: set[str] = {
        edit.path
        for edit, classification in zip(transaction.edits, classifications, strict=True)
        if classification == "applied"
    }
    for edit, classification in zip(transaction.edits, classifications, strict=True):
        if edit.created_root_index is not None:
            continue
        _require_root(root, transaction.root_identity)
        _require_preconditions(root, transaction, applied_paths=applied_paths)
        target = _target_path(root, edit.path)
        _require_existing_parent(
            root,
            target.parent,
            path=edit.path,
            expected=edit.parent_identity,
        )
        stage = _edit_stage_path(transaction, edit)
        backup = transaction.directory / "backup" / edit.backup_name
        if classification == "applied":
            _finalize_windows_published_path(target, expected=edit.after.identity)
            continue
        if edit.operation == "update_region" and classification == "pending":
            assert edit.before is not None
            _rename_no_replace(
                target,
                backup,
                expected=edit.before.identity,
                expected_source_parent=edit.parent_identity,
                expected_destination_parent=transaction.private_identities.backup,
                label="generator original",
            )
            _fault(f"original_moved_before_sync:{edit.index}")
            _sync_directory(target.parent)
            _sync_directory(backup.parent)
            _fault(f"original_moved_before_record:{edit.index}")
            _append_event(transaction, "original_moved", edit.index)
            _fault(f"original_moved:{edit.index}")
            classification = "original_moved"
        if classification not in {"pending", "original_moved"}:
            raise _conflict(
                "Generator transaction is not in a commit-compatible state.",
                paths=(edit.path,),
            )
        _rename_no_replace(
            stage,
            target,
            expected=edit.after.identity,
            expected_source_parent=transaction.private_identities.new,
            expected_destination_parent=edit.parent_identity,
            label="generator stage",
            unexpected_policy=_UnexpectedRenamePolicy.PRESERVE_DESTINATION,
        )
        _finalize_windows_published_path(target, expected=edit.after.identity)
        _fault(f"new_published_before_sync:{edit.index}")
        _sync_file(target, expected=edit.after.identity)
        _sync_directory(target.parent)
        _sync_directory(stage.parent)
        _fault(f"new_published_before_record:{edit.index}")
        _append_event(transaction, "new_published", edit.index)
        _fault(f"new_published:{edit.index}")
        applied_paths.add(edit.path)
    for created_root in transaction.created_roots:
        _require_root(root, transaction.root_identity)
        _require_preconditions(root, transaction, applied_paths=applied_paths)
        target = _target_path(root, created_root.path)
        _require_existing_parent(
            root,
            target.parent,
            path=created_root.path,
            expected=created_root.parent_identity,
        )
        _reject_name_alias(
            _target_path(root, created_root.path).parent, Path(created_root.path).name
        )
        stage = transaction.directory / "trees" / created_root.stage_name
        target_tree = _snapshot_tree_optional(target, label=created_root.path)
        stage_tree = _snapshot_tree_optional(
            stage,
            label=f"staged created root {created_root.path}",
        )
        if _tree_snapshot_matches(target_tree, created_root.after) and stage_tree is None:
            _finalize_windows_published_path(
                target,
                expected=created_root.after.identity,
            )
            applied_paths.update(
                transaction.edits[index].path for index in created_root.edit_indexes
            )
            continue
        if target_tree is not None or not _tree_snapshot_matches(stage_tree, created_root.after):
            raise _conflict(
                "A transaction-created directory changed before publication.",
                paths=(created_root.path,),
            )
        _rename_no_replace(
            stage,
            target,
            expected=created_root.after.identity,
            expected_source_parent=transaction.private_identities.trees,
            expected_destination_parent=created_root.parent_identity,
            label="generator created subtree",
            unexpected_policy=_UnexpectedRenamePolicy.PRESERVE_DESTINATION,
        )
        _finalize_windows_published_path(
            target,
            expected=created_root.after.identity,
        )
        _fault(f"created_root_published_before_sync:{created_root.index}")
        _require_tree_snapshot(target, created_root.after, label=created_root.path)
        _sync_directory(target.parent)
        _sync_directory(stage.parent)
        _fault(f"created_root_published_before_record:{created_root.index}")
        _append_event(transaction, "created_root_published", created_root.index)
        _fault(f"created_root_published:{created_root.index}")
        applied_paths.update(transaction.edits[index].path for index in created_root.edit_indexes)
    _require_preconditions(root, transaction, applied_paths=applied_paths)


def _rollback_transaction(root: Path, transaction: _Transaction) -> None:
    _require_root(root, transaction.root_identity)
    for created_root in reversed(transaction.created_roots):
        target = _target_path(root, created_root.path)
        stage = transaction.directory / "trees" / created_root.stage_name
        target_tree = _snapshot_tree_optional(target, label=created_root.path)
        stage_tree = _snapshot_tree_optional(
            stage,
            label=f"staged created root {created_root.path}",
        )
        if _tree_snapshot_matches(stage_tree, created_root.after) and target_tree is None:
            continue
        if not _tree_snapshot_matches(target_tree, created_root.after) or stage_tree is not None:
            raise _conflict(
                "A transaction-created directory changed before rollback.",
                paths=(created_root.path,),
            )
        _require_existing_parent(
            root,
            target.parent,
            path=created_root.path,
            expected=created_root.parent_identity,
        )
        _rename_no_replace(
            target,
            stage,
            expected=created_root.after.identity,
            expected_source_parent=created_root.parent_identity,
            expected_destination_parent=transaction.private_identities.trees,
            label="generator created subtree",
        )
        _fault(f"created_root_restored_before_sync:{created_root.index}")
        _require_tree_snapshot(stage, created_root.after, label=created_root.path)
        _sync_directory(target.parent)
        _sync_directory(stage.parent)
        _fault(f"created_root_restored_before_record:{created_root.index}")
        _append_event(transaction, "created_root_restored", created_root.index)
        _fault(f"created_root_restored:{created_root.index}")
    classifications = _classify_transaction(root, transaction)
    for edit, classification in reversed(
        tuple(zip(transaction.edits, classifications, strict=True))
    ):
        target = _target_path(root, edit.path)
        if edit.created_root_index is not None:
            continue
        stage = _edit_stage_path(transaction, edit)
        backup = transaction.directory / "backup" / edit.backup_name
        quarantine = transaction.directory / "quarantine" / edit.quarantine_name
        if classification == "pending":
            continue
        if classification == "applied":
            _rename_no_replace(
                target,
                quarantine,
                expected=edit.after.identity,
                expected_source_parent=edit.parent_identity,
                expected_destination_parent=transaction.private_identities.quarantine,
                label="generator after-image",
            )
            _fault(f"new_removed_before_sync:{edit.index}")
            _sync_directory(target.parent)
            _sync_directory(quarantine.parent)
            _fault(f"new_removed_before_record:{edit.index}")
            _append_event(transaction, "new_removed", edit.index)
            _fault(f"new_removed:{edit.index}")
            classification = "original_moved" if edit.before is not None else "rolled_back"
        if edit.before is not None and classification == "original_moved":
            _rename_no_replace(
                backup,
                target,
                expected=edit.before.identity,
                expected_source_parent=transaction.private_identities.backup,
                expected_destination_parent=edit.parent_identity,
                label="generator original backup",
                unexpected_policy=_UnexpectedRenamePolicy.PRESERVE_DESTINATION,
            )
            _fault(f"original_restored_before_sync:{edit.index}")
            _sync_file(target, expected=edit.before.identity)
            _sync_directory(target.parent)
            _sync_directory(backup.parent)
            _fault(f"original_restored_before_record:{edit.index}")
            _append_event(transaction, "original_restored", edit.index)
            _fault(f"original_restored:{edit.index}")
        elif (edit.before is not None and classification == "restored") or (
            edit.before is None and classification == "rolled_back"
        ):
            pass
        else:
            raise _conflict(
                "Generator transaction is not in a rollback-compatible state.",
                paths=(edit.path,),
            )
        if not _entry_exists(stage) and _entry_exists(quarantine):
            _rename_no_replace(
                quarantine,
                stage,
                expected=edit.after.identity,
                expected_source_parent=transaction.private_identities.quarantine,
                expected_destination_parent=transaction.private_identities.new,
                label="generator rollback stage",
            )
            _fault(f"rollback_stage_restored_before_sync:{edit.index}")
            _sync_directory(stage.parent)
            _sync_directory(quarantine.parent)
    _require_rolled_back(root, transaction)


def _recover_transaction(root: Path, state: Path, transaction: _Transaction) -> None:
    if transaction.phase is _Phase.PREPARED:
        _require_prepared(root, transaction)
        _cleanup_transaction(state, transaction)
        return
    if transaction.phase is _Phase.COMMITTING:
        _commit_transaction(root, state, transaction)
        _append_phase(transaction, _Phase.COMMITTED)
        _publish_receipt(root, state, transaction)
        _cleanup_transaction(state, transaction)
        return
    if transaction.phase is _Phase.ROLLING_BACK:
        _rollback_transaction(root, transaction)
        _append_phase(transaction, _Phase.ROLLED_BACK)
        _cleanup_transaction(state, transaction)
        return
    if transaction.phase is _Phase.COMMITTED:
        _require_committed(root, transaction)
        _publish_receipt(root, state, transaction)
        _cleanup_transaction(state, transaction)
        return
    if transaction.phase is _Phase.ROLLED_BACK:
        _require_rolled_back(root, transaction)
        _cleanup_transaction(state, transaction)
        return
    raise _invalid_record("unknown generator transaction phase")


def _classify_transaction(root: Path, transaction: _Transaction) -> tuple[str, ...]:
    created_root_states: dict[int, str] = {}
    for created_root in transaction.created_roots:
        _require_existing_parent(
            root,
            _target_path(root, created_root.path).parent,
            path=created_root.path,
            expected=created_root.parent_identity,
        )
        target_tree = _snapshot_tree_optional(
            _target_path(root, created_root.path),
            label=created_root.path,
        )
        stage_tree = _snapshot_tree_optional(
            transaction.directory / "trees" / created_root.stage_name,
            label=f"staged created root {created_root.path}",
        )
        if target_tree is None and _tree_snapshot_matches(stage_tree, created_root.after):
            created_root_states[created_root.index] = "pending"
        elif stage_tree is None and _tree_snapshot_matches(target_tree, created_root.after):
            created_root_states[created_root.index] = "applied"
        else:
            raise _conflict(
                "A transaction-created directory has ambiguous recovery state.",
                paths=(created_root.path,),
            )
    results: list[str] = []
    for edit in transaction.edits:
        target = _target_path(root, edit.path)
        stage = _edit_stage_path(transaction, edit)
        parent_to_check = target.parent if target.parent.exists() else stage.parent
        if target.parent.exists():
            _reject_name_alias(target.parent, target.name)
        if (
            _capture_directory_identity(
                parent_to_check,
                label=f"parent of {edit.path}",
            )
            != edit.parent_identity
        ):
            raise _conflict(
                "A generator target parent changed while recovery was active.",
                paths=(edit.path,),
            )
        target_snapshot = _snapshot_optional(target, label=edit.path)
        stage_snapshot = _snapshot_optional(
            stage,
            label=f"staged {edit.path}",
        )
        backup_snapshot = _snapshot_optional(
            transaction.directory / "backup" / edit.backup_name,
            label=f"backup {edit.path}",
        )
        quarantine_snapshot = _snapshot_optional(
            transaction.directory / "quarantine" / edit.quarantine_name,
            label=f"rollback {edit.path}",
        )
        target_before = _snapshot_matches(target_snapshot, edit.before)
        target_after = _snapshot_matches(target_snapshot, edit.after)
        stage_after = _snapshot_matches(stage_snapshot, edit.after)
        backup_before = _snapshot_matches(backup_snapshot, edit.before)
        quarantine_after = _snapshot_matches(quarantine_snapshot, edit.after)
        if (
            target_before
            and stage_after
            and backup_snapshot is None
            and quarantine_snapshot is None
        ):
            results.append("pending")
        elif (
            target_snapshot is None
            and stage_after
            and backup_before
            and quarantine_snapshot is None
        ):
            results.append("original_moved")
        elif (
            target_after
            and stage_snapshot is None
            and (backup_before if edit.before is not None else backup_snapshot is None)
            and quarantine_snapshot is None
        ):
            results.append("applied")
        elif (
            edit.before is not None
            and target_before
            and backup_snapshot is None
            and stage_snapshot is None
            and quarantine_after
        ):
            results.append("restored")
        elif (
            edit.before is not None
            and target_snapshot is None
            and backup_before
            and stage_snapshot is None
            and quarantine_after
        ):
            results.append("original_moved")
        elif (
            target_before
            and backup_snapshot is None
            and stage_after
            and quarantine_snapshot is None
        ):
            results.append("pending")
        elif (
            edit.before is None
            and target_snapshot is None
            and stage_snapshot is None
            and backup_snapshot is None
            and quarantine_after
        ):
            results.append("rolled_back")
        else:
            raise _conflict(
                "A generator path changed outside the pending transaction; recovery preserved all state.",
                paths=(edit.path,),
            )
    for created_root in transaction.created_roots:
        expected = created_root_states[created_root.index]
        if any(results[index] != expected for index in created_root.edit_indexes):
            raise _conflict(
                "A transaction-created directory and its files disagree.",
                paths=(created_root.path,),
            )
    return tuple(results)


def _require_commit_prefix(classifications: tuple[str, ...]) -> None:
    for classification in classifications:
        if classification not in {"pending", "original_moved", "applied"}:
            raise _conflict("Generator transaction cannot continue its commit.")


def _edit_stage_path(transaction: _Transaction, edit: _EditRecord) -> Path:
    if edit.created_root_index is None:
        return transaction.directory / "new" / edit.stage_name
    created_root = transaction.created_roots[edit.created_root_index]
    relative = PurePosixPath(edit.path).relative_to(PurePosixPath(created_root.path))
    return (transaction.directory / "trees" / created_root.stage_name).joinpath(*relative.parts)


def _require_prepared(root: Path, transaction: _Transaction) -> None:
    classifications = _classify_transaction(root, transaction)
    if any(classification != "pending" for classification in classifications):
        raise _conflict("A prepared generator transaction has unexpected project mutations.")
    _require_preconditions(root, transaction, applied_paths=set())


def _require_committed(root: Path, transaction: _Transaction) -> None:
    classifications = _classify_transaction(root, transaction)
    if any(classification != "applied" for classification in classifications):
        raise _conflict("A committed generator transaction no longer matches its after-images.")
    _require_preconditions(
        root,
        transaction,
        applied_paths={edit.path for edit in transaction.edits},
    )


def _require_rolled_back(root: Path, transaction: _Transaction) -> None:
    classifications = _classify_transaction(root, transaction)
    if any(classification not in {"pending", "rolled_back"} for classification in classifications):
        raise _conflict("A rolled-back generator transaction no longer matches its before-images.")
    _require_preconditions(root, transaction, applied_paths=set())


def _capture_request_authority(
    root: Path,
    request: GeneratorTransactionRequest,
) -> _RequestAuthority:
    root_device = _capture_directory_identity(root, label="project root").device
    before: list[_FileSnapshot | None] = []
    parent_identities: list[_Identity | None] = []
    created_members: dict[str, list[int]] = {}
    total_observed = 0
    observed_paths: set[str] = set()
    edits_by_path = {edit.path: edit for edit in request.edits}
    for index, edit in enumerate(request.edits):
        target = _target_path(root, edit.path)
        parent_identity, created_root = _capture_parent_authority(
            root,
            target.parent,
            path=edit.path,
        )
        if parent_identity is not None:
            if parent_identity.device != root_device:
                raise GeneratorTransactionError(
                    "cross_device_target",
                    "Generator targets must share the project filesystem.",
                    paths=(edit.path,),
                )
            _reject_name_alias(target.parent, target.name)
        snapshot = _snapshot_optional(target, label=edit.path)
        if edit.operation == "create":
            if snapshot is not None:
                raise GeneratorTransactionError(
                    "preimage_changed",
                    f"{edit.path} changed after the plan was created.",
                    paths=(edit.path,),
                )
            if created_root is not None:
                created_members.setdefault(created_root, []).append(index)
        else:
            if created_root is not None:
                raise GeneratorTransactionError(
                    "preimage_changed",
                    f"{edit.path} changed after the plan was created.",
                    paths=(edit.path,),
                )
            if snapshot is None or snapshot.content_sha256 != edit.preimage_sha256:
                raise GeneratorTransactionError(
                    "preimage_changed",
                    f"{edit.path} changed after the plan was created.",
                    paths=(edit.path,),
                )
            if edit.path not in observed_paths:
                total_observed += snapshot.size
                observed_paths.add(edit.path)
                if total_observed > _MAX_STAGED_BYTES:
                    raise GeneratorTransactionError(
                        "observed_content_limit",
                        "Generator preimages exceed the bounded inspection limit.",
                    )
        before.append(snapshot)
        parent_identities.append(parent_identity)
    preconditions: list[_PreconditionRecord] = []
    before_by_path = dict(zip((edit.path for edit in request.edits), before, strict=True))
    for precondition in request.preconditions:
        target = _target_path(root, precondition.path)
        parent_identity = _capture_existing_parent_identity(
            root,
            target.parent,
            path=precondition.path,
        )
        if parent_identity.device != root_device:
            raise GeneratorTransactionError(
                "cross_device_precondition",
                "Generator preconditions must share the project filesystem.",
                paths=(precondition.path,),
            )
        _reject_name_alias(target.parent, target.name)
        snapshot = _snapshot_optional(
            target,
            label=precondition.path,
        )
        if snapshot is None or snapshot.content_sha256 != precondition.content_sha256:
            raise GeneratorTransactionError(
                "precondition_changed",
                f"{precondition.path} changed after the plan was created.",
                paths=(precondition.path,),
            )
        edit = edits_by_path.get(precondition.path)
        if edit is not None:
            expected = before_by_path[precondition.path]
            if (
                expected is None
                or snapshot != expected
                or (edit.preimage_sha256 != precondition.content_sha256)
            ):
                raise GeneratorTransactionError(
                    "precondition_conflict",
                    "Generator edit and precondition authority conflict.",
                    paths=(precondition.path,),
                )
        if precondition.path not in observed_paths:
            total_observed += snapshot.size
            observed_paths.add(precondition.path)
        if total_observed > _MAX_STAGED_BYTES:
            raise GeneratorTransactionError(
                "observed_content_limit",
                "Generator preimages exceed the bounded inspection limit.",
            )
        preconditions.append(
            _PreconditionRecord(
                path=precondition.path,
                snapshot=snapshot,
                parent_identity=parent_identity,
            )
        )
    created_roots = tuple(
        _CreatedRootPlan(
            path=path,
            parent_identity=_capture_existing_parent_identity(
                root,
                _target_path(root, path).parent,
                path=path,
            ),
            edit_indexes=tuple(indexes),
        )
        for path, indexes in sorted(created_members.items())
    )
    if any(item.parent_identity.device != root_device for item in created_roots):
        raise GeneratorTransactionError(
            "cross_device_target",
            "Generator targets must share the project filesystem.",
        )
    return _RequestAuthority(
        before=tuple(before),
        parent_identities=tuple(parent_identities),
        preconditions=tuple(preconditions),
        created_roots=created_roots,
    )


def _record_bound_identity(kind: int) -> _Identity:
    return _Identity(
        device=_MAX_IDENTITY_DEVICE,
        inode=_MAX_IDENTITY_INODE,
        kind=kind,
        incarnation=_MAX_IDENTITY_INCARNATION,
    )


def _require_identity_record_bound(identity: _Identity) -> None:
    if (
        type(identity.device) is not int
        or not 0 <= identity.device <= _MAX_IDENTITY_DEVICE
        or type(identity.inode) is not int
        or not 0 <= identity.inode <= _MAX_IDENTITY_INODE
        or type(identity.kind) is not int
        or not 0 <= identity.kind <= _MAX_IDENTITY_KIND
        or type(identity.incarnation) is not int
        or not 0 <= identity.incarnation <= _MAX_IDENTITY_INCARNATION
    ):
        raise GeneratorTransactionError(
            "record_limit",
            "Generator filesystem identity exceeds its durable record bounds.",
        )


def _cleanup_entry_bound(
    path: str,
    *,
    directory: bool,
) -> _CleanupEntry:
    return _CleanupEntry(
        path=path,
        identity=_record_bound_identity(stat.S_IFDIR if directory else stat.S_IFREG),
        mode=0o7777,
        size=None if directory else _MAX_STAGED_BYTES,
        content_sha256=None if directory else f"sha256:{'f' * 64}",
    )


def _bounded_cleanup_entries(
    request: GeneratorTransactionRequest,
    authority: _RequestAuthority,
) -> tuple[_CleanupEntry, ...]:
    paths: dict[str, bool] = {
        "new": True,
        "trees": True,
        "backup": True,
        "quarantine": True,
        _JOURNAL_FILE: False,
        _OWNER_FILE: False,
        _NEXT_RECEIPT_FILE: False,
        _PREVIOUS_RECEIPT_FILE: False,
    }
    created_root_by_edit = {
        edit_index: (root_index, root_plan)
        for root_index, root_plan in enumerate(authority.created_roots)
        for edit_index in root_plan.edit_indexes
    }
    for index, edit in enumerate(request.edits):
        created = created_root_by_edit.get(index)
        if created is None:
            stage_name = f"{index:04d}"
            for prefix in ("new", "backup", "quarantine"):
                paths[f"{prefix}/{stage_name}"] = False
            continue
        root_index, root_plan = created
        prefix = PurePosixPath("trees", f"{root_index:04d}")
        paths[prefix.as_posix()] = True
        relative = PurePosixPath(edit.path).relative_to(PurePosixPath(root_plan.path))
        for depth in range(1, len(relative.parts)):
            paths[(prefix / PurePosixPath(*relative.parts[:depth])).as_posix()] = True
        paths[(prefix / relative).as_posix()] = False
    if len(paths) > _MAX_PRIVATE_TREE_ENTRIES:
        raise GeneratorTransactionError(
            "cleanup_claim_limit",
            "Generator cleanup authority exceeds its entry-count limit.",
        )
    return tuple(
        _cleanup_entry_bound(path, directory=directory) for path, directory in sorted(paths.items())
    )


def _validate_durable_representation_bounds(
    root_identity: _Identity,
    request: GeneratorTransactionRequest,
    authority: _RequestAuthority,
) -> None:
    identities = [root_identity]
    identities.extend(item for item in authority.parent_identities if item is not None)
    identities.extend(snapshot.identity for snapshot in authority.before if snapshot is not None)
    identities.extend(item.parent_identity for item in authority.preconditions)
    identities.extend(item.snapshot.identity for item in authority.preconditions)
    identities.extend(item.parent_identity for item in authority.created_roots)
    for identity in identities:
        _require_identity_record_bound(identity)

    file_identity = _record_bound_identity(stat.S_IFREG)
    directory_identity = _record_bound_identity(stat.S_IFDIR)
    created_root_by_edit = {
        edit_index: root_index
        for root_index, root_plan in enumerate(authority.created_roots)
        for edit_index in root_plan.edit_indexes
    }
    edits = tuple(
        _EditRecord(
            index=index,
            path=edit.path,
            operation=edit.operation,
            before=authority.before[index],
            after=_FileSnapshot(
                identity=file_identity,
                content_sha256=edit.content_sha256,
                mode=0o7777,
                size=len(edit.content),
            ),
            parent_identity=(authority.parent_identities[index] or directory_identity),
            created_root_index=created_root_by_edit.get(index),
            stage_name=f"{index:04d}",
            backup_name=f"{index:04d}",
            quarantine_name=f"{index:04d}",
        )
        for index, edit in enumerate(request.edits)
    )
    created_roots = tuple(
        _CreatedRootRecord(
            index=index,
            path=plan.path,
            stage_name=f"{index:04d}",
            parent_identity=plan.parent_identity,
            after=_TreeSnapshot(
                identity=directory_identity,
                fingerprint="f" * 64,
                entries=max(1, len(plan.edit_indexes) * _MAX_PATH_DEPTH),
            ),
            edit_indexes=plan.edit_indexes,
        )
        for index, plan in enumerate(authority.created_roots)
    )
    receipt = _receipt_content(
        request_digest=request.digest,
        root_identity=root_identity,
        edits=edits,
        preconditions=authority.preconditions,
        created_roots=created_roots,
    )
    receipt_snapshot = _FileSnapshot(
        identity=file_identity,
        content_sha256=_sha256(receipt),
        mode=0o7777,
        size=len(receipt),
    )
    private_identities = _PrivateIdentities(
        state=directory_identity,
        new=directory_identity,
        trees=directory_identity,
        backup=directory_identity,
        quarantine=directory_identity,
    )
    manifest_payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "manifest",
        "sequence": 0,
        "previous_sha256": None,
        "token": "f" * 64,
        "request_digest": request.digest,
        "root_identity": root_identity.as_json(),
        "transaction_identity": directory_identity.as_json(),
        "private_identities": private_identities.payload(),
        "phase": _Phase.PREPARED.value,
        "edits": [edit.payload() for edit in edits],
        "preconditions": [item.payload() for item in authority.preconditions],
        "created_roots": [item.payload() for item in created_roots],
        "prior_receipt": receipt_snapshot.payload(),
        "receipt_after": receipt_snapshot.payload(),
    }
    manifest, _manifest_sha256 = _journal_entry(manifest_payload)
    if len(manifest) > _MAX_JOURNAL_BYTES:
        raise GeneratorTransactionError(
            "journal_limit",
            "Generator transaction manifest exceeds its encoded-size limit.",
        )
    preparation_owner = _FileSnapshot(
        identity=file_identity,
        content_sha256="f" * 64,
        mode=0o7777,
        size=4096,
    )
    _cleanup_claim_content(
        owner_kind="transaction",
        token="f" * 64,
        request_digest=request.digest,
        root_identity=root_identity,
        transaction_identity=directory_identity,
        phase=_Phase.ROLLING_BACK.value,
        preparation_owner=preparation_owner,
        journal=_FileSnapshot(
            identity=file_identity,
            content_sha256="f" * 64,
            mode=0o7777,
            size=_MAX_JOURNAL_BYTES,
        ),
        transaction_owner=_FileSnapshot(
            identity=file_identity,
            content_sha256="f" * 64,
            mode=0o7777,
            size=4096,
        ),
        entries=_bounded_cleanup_entries(request, authority),
    )


def _require_preconditions(
    root: Path,
    transaction: _Transaction,
    *,
    applied_paths: set[str],
) -> None:
    edits = {edit.path: edit for edit in transaction.edits}
    for precondition in transaction.preconditions:
        expected = precondition.snapshot
        edit = edits.get(precondition.path)
        if edit is not None and precondition.path in applied_paths:
            expected = edit.after
        target = _target_path(root, precondition.path)
        _require_existing_parent(
            root,
            target.parent,
            path=precondition.path,
            expected=precondition.parent_identity,
        )
        _reject_name_alias(target.parent, target.name)
        current = _snapshot_optional(
            target,
            label=precondition.path,
        )
        if (
            edit is not None
            and precondition.path not in applied_paths
            and current is None
            and edit.before is not None
            and _snapshot_matches(
                _snapshot_optional(
                    transaction.directory / "backup" / edit.backup_name,
                    label=f"backup {edit.path}",
                ),
                edit.before,
            )
        ):
            continue
        if not _snapshot_matches(current, expected):
            raise _conflict(
                "A generator precondition changed while the transaction was active.",
                paths=(precondition.path,),
            )


def _publish_receipt(root: Path, state: Path, transaction: _Transaction) -> None:
    _require_committed(root, transaction)
    next_receipt = transaction.directory / _NEXT_RECEIPT_FILE
    receipt = state / _RECEIPT_FILE
    previous = transaction.directory / _PREVIOUS_RECEIPT_FILE
    current_receipt = _snapshot_optional(receipt, label="generator receipt")
    current_previous = _snapshot_optional(previous, label="previous generator receipt")
    current_next = _snapshot_optional(next_receipt, label="staged generator receipt")
    if _snapshot_matches(current_receipt, transaction.receipt_after) and current_next is None:
        return
    if transaction.prior_receipt is None:
        if current_receipt is not None or current_previous is not None:
            raise _conflict("Generator receipt authority changed before publication.")
    elif _snapshot_matches(current_receipt, transaction.prior_receipt):
        if current_previous is not None:
            raise _conflict("Generator previous-receipt authority is ambiguous.")
        _rename_no_replace(
            receipt,
            previous,
            expected=transaction.prior_receipt.identity,
            expected_source_parent=transaction.private_identities.state,
            expected_destination_parent=transaction.directory_identity,
            label="generator previous receipt",
        )
        _fault("previous_receipt_moved_before_sync")
        _sync_directory(state)
        _sync_directory(transaction.directory)
        _fault("previous_receipt_moved")
    elif current_receipt is None and _snapshot_matches(
        current_previous,
        transaction.prior_receipt,
    ):
        pass
    else:
        raise _conflict("Generator receipt authority changed before publication.")
    if not _snapshot_matches(current_next, transaction.receipt_after):
        raise _conflict("Staged generator receipt changed before publication.")
    _rename_no_replace(
        next_receipt,
        receipt,
        expected=transaction.receipt_after.identity,
        expected_source_parent=transaction.directory_identity,
        expected_destination_parent=transaction.private_identities.state,
        label="generator receipt",
    )
    _fault("receipt_published_before_sync")
    _sync_directory(state)
    _sync_directory(transaction.directory)
    _fault("receipt_published")


def _receipt_content(
    *,
    request_digest: str,
    root_identity: _Identity,
    edits: tuple[_EditRecord, ...],
    preconditions: tuple[_PreconditionRecord, ...],
    created_roots: tuple[_CreatedRootRecord, ...],
) -> bytes:
    edit_receipts = tuple(
        _PreconditionRecord(
            path=edit.path,
            snapshot=edit.after,
            parent_identity=edit.parent_identity,
        )
        for edit in edits
    )
    edit_paths = {edit.path for edit in edits}
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "request_digest": request_digest,
        "root_identity": root_identity.as_json(),
        "edits": [item.payload() for item in edit_receipts],
        "preconditions": [item.payload() for item in preconditions if item.path not in edit_paths],
        "created_roots": [
            {
                "path": item.path,
                "parent_identity": item.parent_identity.as_json(),
                "snapshot": item.after.payload(),
            }
            for item in created_roots
        ],
    }
    canonical = _canonical_json(payload)
    encoded = _canonical_json({**payload, "receipt_sha256": _sha256(canonical)}) + b"\n"
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise GeneratorTransactionError(
            "receipt_limit",
            "Generator transaction receipt exceeds its encoded-size limit.",
        )
    return encoded


def _load_receipt(state: Path, *, required: bool) -> _Receipt | None:
    path = state / _RECEIPT_FILE
    if not _entry_exists(path):
        if required:
            raise _invalid_record("generator receipt is missing")
        return None
    value = _read_json_file(path, limit=_MAX_RECEIPT_BYTES, label="generator receipt")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "request_digest",
        "root_identity",
        "edits",
        "preconditions",
        "created_roots",
        "receipt_sha256",
    }:
        raise _invalid_record("generator receipt has unexpected fields")
    value = cast("dict[str, object]", value)
    supplied = value.pop("receipt_sha256")
    if not _is_sha256(supplied) or supplied != _sha256(_canonical_json(value)):
        raise _invalid_record("generator receipt digest does not match")
    if value["schema_version"] != _SCHEMA_VERSION or type(value["schema_version"]) is not int:
        raise _invalid_record("unsupported generator receipt schema")
    request_digest = value["request_digest"]
    if not _is_sha256(request_digest):
        raise _invalid_record("invalid generator receipt request digest")
    edits = _parse_receipt_snapshots(
        value["edits"],
        field="edits",
        limit=_MAX_EDITS,
        require_nonempty=True,
    )
    preconditions = _parse_receipt_snapshots(
        value["preconditions"],
        field="preconditions",
        limit=_MAX_PRECONDITIONS,
        require_nonempty=False,
    )
    created_roots = _parse_receipt_created_roots(value["created_roots"])
    receipt = _Receipt(
        request_digest=cast("str", request_digest),
        root_identity=_identity_from_json(
            value["root_identity"],
            field="root_identity",
            expected_kind=stat.S_IFDIR,
        ),
        edits=edits,
        preconditions=preconditions,
        created_roots=created_roots,
    )
    _require_receipt_topology(receipt)
    return receipt


def _parse_receipt_snapshots(
    value: object,
    *,
    field: str,
    limit: int,
    require_nonempty: bool,
) -> tuple[_PreconditionRecord, ...]:
    if not isinstance(value, list) or len(value) > limit or (require_nonempty and not value):
        raise _invalid_record(f"invalid generator receipt {field}")
    result: list[_PreconditionRecord] = []
    for index, item in enumerate(value):
        result.append(_parse_precondition_record(item, index=index))
    return tuple(result)


def _require_receipt_topology(receipt: _Receipt) -> None:
    edit_paths: dict[str, str] = {}
    for item in receipt.edits:
        key = _normalized_path(item.path)
        if key in edit_paths:
            raise _invalid_record("generator receipt contains duplicate or aliased edits")
        edit_paths[key] = item.path
    ordered_edit_keys = sorted(edit_paths)
    for index, key in enumerate(ordered_edit_keys):
        prefix = f"{key}/"
        if any(later.startswith(prefix) for later in ordered_edit_keys[index + 1 :]):
            raise _invalid_record("generator receipt contains overlapping edit paths")

    precondition_paths: set[str] = set()
    for item in receipt.preconditions:
        key = _normalized_path(item.path)
        if key in edit_paths or key in precondition_paths:
            raise _invalid_record("generator receipt contains duplicate or aliased preconditions")
        precondition_paths.add(key)

    created_paths: dict[str, str] = {}
    for path, _parent_identity, _snapshot in receipt.created_roots:
        key = _normalized_path(path)
        if key in created_paths:
            raise _invalid_record("generator receipt contains duplicate or aliased created roots")
        created_paths[key] = path
        prefix = f"{key}/"
        if not any(edit_key.startswith(prefix) for edit_key in edit_paths):
            raise _invalid_record(
                "generator receipt created root does not contain a published edit"
            )
    ordered_created_keys = sorted(created_paths)
    for index, key in enumerate(ordered_created_keys):
        prefix = f"{key}/"
        if any(later.startswith(prefix) for later in ordered_created_keys[index + 1 :]):
            raise _invalid_record("generator receipt contains overlapping created roots")


def _parse_receipt_created_roots(
    value: object,
) -> tuple[tuple[str, _Identity, _TreeSnapshot], ...]:
    if not isinstance(value, list) or len(value) > _MAX_EDITS:
        raise _invalid_record("invalid generator receipt created roots")
    result: list[tuple[str, _Identity, _TreeSnapshot]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "parent_identity",
            "snapshot",
        }:
            raise _invalid_record("invalid generator receipt created-root item")
        item = cast("dict[str, object]", item)
        result.append(
            (
                _parse_record_path(item["path"], field=f"created_roots[{index}].path"),
                _identity_from_json(
                    item["parent_identity"],
                    field=f"created_roots[{index}].parent_identity",
                    expected_kind=stat.S_IFDIR,
                ),
                _TreeSnapshot.parse(
                    item["snapshot"],
                    field=f"created_roots[{index}].snapshot",
                ),
            )
        )
    return tuple(result)


def _require_receipt_state(
    root: Path,
    root_identity: _Identity,
    receipt: _Receipt,
    *,
    request: GeneratorTransactionRequest,
) -> None:
    if receipt.root_identity != root_identity:
        raise _conflict("Generator receipt belongs to another project root.")
    expected_preconditions = tuple(
        item
        for item in request.preconditions
        if item.path not in {edit.path for edit in request.edits}
    )
    if (
        receipt.request_digest != request.digest
        or len(receipt.edits) != len(request.edits)
        or len(receipt.preconditions) != len(expected_preconditions)
        or any(
            recorded.path != expected.path
            or recorded.snapshot.content_sha256 != expected.content_sha256
            or recorded.snapshot.size != len(expected.content)
            for recorded, expected in zip(receipt.edits, request.edits, strict=True)
        )
        or any(
            recorded.path != expected.path
            or recorded.snapshot.content_sha256 != expected.content_sha256
            for recorded, expected in zip(
                receipt.preconditions,
                expected_preconditions,
                strict=True,
            )
        )
    ):
        raise _invalid_record("generator receipt does not match the exact request")
    for item in (*receipt.edits, *receipt.preconditions):
        target = _target_path(root, item.path)
        _require_existing_parent(
            root,
            target.parent,
            path=item.path,
            expected=item.parent_identity,
        )
        _reject_name_alias(target.parent, target.name)
        current = _snapshot_optional(target, label=item.path)
        if not _snapshot_matches(current, item.snapshot):
            raise _conflict(
                "Generator exact retry conflicts with the current project state.",
                paths=(item.path,),
            )
    for path, parent_identity, expected in receipt.created_roots:
        target = _target_path(root, path)
        _require_existing_parent(
            root,
            target.parent,
            path=path,
            expected=parent_identity,
        )
        _reject_name_alias(target.parent, target.name)
        _require_tree_snapshot(target, expected, label=path)


def _preparation_owner_path(state: Path, token: str) -> Path:
    return state / f"prepare-{token}.owner.json"


def _preparation_owner_content(
    *,
    token: str,
    request_digest: str,
    root_identity: _Identity,
    transaction_identity: _Identity,
) -> bytes:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "token": token,
        "request_digest": request_digest,
        "root_identity": root_identity.as_json(),
        "transaction_identity": transaction_identity.as_json(),
    }
    return (
        _canonical_json(
            {
                **payload,
                "owner_sha256": _sha256(_canonical_json(payload)),
            }
        )
        + b"\n"
    )


def _load_preparation_authority(
    directory: Path,
    *,
    expected_token: str | None = None,
) -> _PreparationAuthority:
    token = expected_token
    if token is None:
        prepare_match = _PREPARE_PATTERN.fullmatch(directory.name)
        cleanup_match = _CLEANUP_PATTERN.fullmatch(directory.name)
        if prepare_match is not None:
            token = prepare_match.group("token")
        elif cleanup_match is not None:
            token = cleanup_match.group("token")
    if token is None:
        raise _invalid_record("generator preparation owner token is unavailable")
    path = _preparation_owner_path(directory.parent, token)
    content, marker_identity = _read_regular_bytes(
        path,
        limit=4096,
        label="generator preparation owner",
    )
    try:
        value = json.loads(content)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _invalid_record("generator preparation owner contains invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "token",
        "request_digest",
        "root_identity",
        "transaction_identity",
        "owner_sha256",
    }:
        raise _invalid_record("generator preparation owner has unexpected fields")
    value = cast("dict[str, object]", value)
    supplied = value.pop("owner_sha256")
    if not _is_sha256(supplied) or supplied != _sha256(_canonical_json(value)):
        raise _invalid_record("generator preparation owner digest does not match")
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        raise _invalid_record("unsupported generator preparation-owner schema")
    stored_token = value["token"]
    if (
        not isinstance(stored_token, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored_token) is None
        or stored_token != token
    ):
        raise _invalid_record("generator preparation owner token is invalid")
    request_digest = value["request_digest"]
    if not _is_sha256(request_digest):
        raise _invalid_record("generator preparation request digest is invalid")
    transaction_identity = _identity_from_json(
        value["transaction_identity"],
        field="preparation_owner.transaction_identity",
        expected_kind=stat.S_IFDIR,
    )
    _require_directory_identity(directory, transaction_identity)
    marker = _snapshot_regular(path, label="generator preparation owner")
    if marker.identity != marker_identity or marker.content_sha256 != _sha256(content):
        raise _invalid_record("generator preparation owner changed while it was read")
    return _PreparationAuthority(
        token=stored_token,
        request_digest=cast("str", request_digest),
        root_identity=_identity_from_json(
            value["root_identity"],
            field="preparation_owner.root_identity",
            expected_kind=stat.S_IFDIR,
        ),
        transaction_identity=transaction_identity,
        marker=marker,
    )


def _cleanup_claim_content(
    *,
    owner_kind: Literal["preparation", "transaction"],
    token: str,
    request_digest: str,
    root_identity: _Identity,
    transaction_identity: _Identity,
    phase: str,
    preparation_owner: _FileSnapshot,
    journal: _FileSnapshot | None,
    transaction_owner: _FileSnapshot | None,
    entries: tuple[_CleanupEntry, ...],
) -> bytes:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "owner_kind": owner_kind,
        "token": token,
        "request_digest": request_digest,
        "root_identity": root_identity.as_json(),
        "transaction_identity": transaction_identity.as_json(),
        "phase": phase,
        "preparation_owner": preparation_owner.payload(),
        "journal": None if journal is None else journal.payload(),
        "transaction_owner": (None if transaction_owner is None else transaction_owner.payload()),
        "entries": [entry.as_json() for entry in entries],
    }
    content = (
        _canonical_json({**payload, "claim_sha256": _sha256(_canonical_json(payload))}) + b"\n"
    )
    if len(content) > _MAX_CLEANUP_CLAIM_BYTES:
        raise GeneratorTransactionError(
            "cleanup_claim_limit",
            "Generator cleanup authority exceeds its encoded-size limit.",
        )
    return content


def _cleanup_claim_path(state: Path, token: str) -> Path:
    return state / f"cleanup-{token}.claim.json"


def _cleanup_owner_path(state: Path, token: str) -> Path:
    return state / f"cleanup-{token}.owner.json"


def _publish_cleanup_owner(
    state: Path,
    directory: Path,
    claim: _CleanupClaim,
) -> _FileSnapshot:
    source = _preparation_owner_path(state, claim.token)
    published = _cleanup_owner_path(state, claim.token)
    current_source = _snapshot_optional(source, label="generator preparation owner")
    current_published = _snapshot_optional(published, label="generator cleanup owner")
    if current_published is not None:
        if current_source is not None or current_published != claim.preparation_owner:
            raise _conflict("Generator cleanup-owner authority is ambiguous.")
        return current_published
    if current_source != claim.preparation_owner:
        raise _conflict("Generator cleanup owner changed before publication.")
    _rename_no_replace(
        source,
        published,
        expected=claim.preparation_owner.identity,
        expected_source_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        expected_destination_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        label="generator cleanup owner",
    )
    _sync_directory(state)
    current_published = _snapshot_regular(published, label="generator cleanup owner")
    if current_published != claim.preparation_owner:
        raise _conflict("Generator cleanup owner changed during publication.")
    return current_published


def _require_cleanup_owner(
    state: Path,
    directory: Path | None,
    claim: _CleanupClaim,
) -> _FileSnapshot | None:
    published = _snapshot_optional(
        _cleanup_owner_path(state, claim.token),
        label="generator cleanup owner",
    )
    if directory is None:
        if published is not None and published != claim.preparation_owner:
            raise _conflict("Generator cleanup owner changed after tree cleanup.")
        return published
    if published is None:
        if not _entry_exists(_preparation_owner_path(state, claim.token)):
            raise _conflict("Generator cleanup lacks its durable owner authority.")
        if claim.owner_kind == "transaction":
            transaction = _load_transaction(directory)
            _require_cleanup_claim_matches_transaction(claim, transaction)
        else:
            preparation = _load_preparation_authority(
                directory,
                expected_token=claim.token,
            )
            _require_cleanup_claim_matches_preparation(claim, preparation)
        return _publish_cleanup_owner(state, directory, claim)
    if (
        _entry_exists(_preparation_owner_path(state, claim.token))
        or published != claim.preparation_owner
    ):
        raise _conflict("Generator cleanup-owner authority is ambiguous.")
    return published


def _remove_cleanup_owner(
    state: Path,
    token: str,
    *,
    expected: _FileSnapshot,
) -> None:
    _unlink_owned_file(
        _cleanup_owner_path(state, token),
        expected=expected.identity,
        expected_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        label="generator cleanup owner",
    )


def _transaction_cleanup_claim_content(
    transaction: _Transaction,
    entries: tuple[_CleanupEntry, ...],
) -> bytes:
    preparation = _load_preparation_authority(
        transaction.directory,
        expected_token=transaction.token,
    )
    journal = _snapshot_regular(
        transaction.directory / _JOURNAL_FILE,
        label="generator journal",
    )
    owner = _snapshot_regular(
        transaction.directory / _OWNER_FILE,
        label="generator owner marker",
    )
    if (
        preparation.request_digest != transaction.request_digest
        or preparation.root_identity != transaction.root_identity
        or preparation.transaction_identity != transaction.directory_identity
        or journal.identity != transaction.journal_identity
    ):
        raise _conflict("Generator cleanup authority does not match its transaction.")
    return _cleanup_claim_content(
        owner_kind="transaction",
        token=transaction.token,
        request_digest=transaction.request_digest,
        root_identity=transaction.root_identity,
        transaction_identity=transaction.directory_identity,
        phase=transaction.phase.value,
        preparation_owner=preparation.marker,
        journal=journal,
        transaction_owner=owner,
        entries=entries,
    )


def _require_cleanup_claim_matches_transaction(
    claim: _CleanupClaim,
    transaction: _Transaction,
) -> None:
    preparation = _load_preparation_authority(
        transaction.directory,
        expected_token=transaction.token,
    )
    journal = _snapshot_regular(
        transaction.directory / _JOURNAL_FILE,
        label="generator journal",
    )
    owner = _snapshot_regular(
        transaction.directory / _OWNER_FILE,
        label="generator owner marker",
    )
    if (
        claim.owner_kind != "transaction"
        or claim.token != transaction.token
        or claim.request_digest != transaction.request_digest
        or claim.root_identity != transaction.root_identity
        or claim.transaction_identity != transaction.directory_identity
        or claim.phase != transaction.phase.value
        or claim.journal is None
        or claim.journal.identity != transaction.journal_identity
        or claim.transaction_owner is None
        or claim.preparation_owner != preparation.marker
        or claim.journal != journal
        or claim.transaction_owner != owner
    ):
        raise _conflict("Generator cleanup claim belongs to another transaction.")


def _preparation_cleanup_claim_content(
    preparation: _PreparationAuthority,
    entries: tuple[_CleanupEntry, ...],
) -> bytes:
    return _cleanup_claim_content(
        owner_kind="preparation",
        token=preparation.token,
        request_digest=preparation.request_digest,
        root_identity=preparation.root_identity,
        transaction_identity=preparation.transaction_identity,
        phase="preparing",
        preparation_owner=preparation.marker,
        journal=None,
        transaction_owner=None,
        entries=entries,
    )


def _require_cleanup_claim_matches_preparation(
    claim: _CleanupClaim,
    preparation: _PreparationAuthority,
) -> None:
    if (
        claim.owner_kind != "preparation"
        or claim.token != preparation.token
        or claim.request_digest != preparation.request_digest
        or claim.root_identity != preparation.root_identity
        or claim.transaction_identity != preparation.transaction_identity
        or claim.phase != "preparing"
        or claim.preparation_owner != preparation.marker
        or claim.journal is not None
        or claim.transaction_owner is not None
    ):
        raise _conflict("Generator cleanup claim belongs to another preparation.")


def _publish_cleanup_claim(
    state: Path,
    transaction: _Transaction,
) -> _CleanupClaim:
    staged = transaction.directory / _CLEANUP_CLAIM_FILE
    published = _cleanup_claim_path(state, transaction.token)
    current_staged = _snapshot_optional(staged, label="staged generator cleanup claim")
    current_published = _snapshot_optional(published, label="generator cleanup claim")
    if current_published is not None:
        if current_staged is not None:
            raise _conflict("Generator cleanup-claim authority is ambiguous.")
        claim = _load_cleanup_claim(published)
        _require_cleanup_claim_matches_transaction(claim, transaction)
        return claim
    if current_staged is None:
        entries = _capture_transaction_cleanup_authority(transaction)
        _write_private_file(
            staged,
            _transaction_cleanup_claim_content(transaction, entries),
            expected_parent=transaction.directory_identity,
        )
        current_staged = _snapshot_regular(
            staged,
            label="staged generator cleanup claim",
        )
    claim = _load_cleanup_claim(
        staged,
        expected_token=transaction.token,
    )
    _require_cleanup_claim_matches_transaction(claim, transaction)
    if current_staged.identity != claim.identity:
        raise _conflict("Generator cleanup-claim authority changed before publication.")
    _rename_no_replace(
        staged,
        published,
        expected=claim.identity,
        expected_source_parent=transaction.directory_identity,
        expected_destination_parent=transaction.private_identities.state,
        label="generator cleanup claim",
    )
    _fault("cleanup_claim_published_before_sync")
    _sync_directory(state)
    _sync_directory(transaction.directory)
    _fault("cleanup_claim_published")
    return claim


def _load_cleanup_claim(
    path: Path,
    *,
    expected_token: str | None = None,
) -> _CleanupClaim:
    content, claim_identity = _read_regular_bytes(
        path,
        limit=_MAX_CLEANUP_CLAIM_BYTES,
        label="generator cleanup claim",
    )
    try:
        value = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise _invalid_record("generator cleanup claim contains invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "owner_kind",
        "token",
        "request_digest",
        "root_identity",
        "transaction_identity",
        "phase",
        "preparation_owner",
        "journal",
        "transaction_owner",
        "entries",
        "claim_sha256",
    }:
        raise _invalid_record("generator cleanup claim has unexpected fields")
    value = cast("dict[str, object]", value)
    supplied = value.pop("claim_sha256")
    if not _is_sha256(supplied) or supplied != _sha256(_canonical_json(value)):
        raise _invalid_record("generator cleanup claim digest does not match")
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        raise _invalid_record("unsupported generator cleanup-claim schema")
    owner_kind = value["owner_kind"]
    if owner_kind not in {"preparation", "transaction"}:
        raise _invalid_record("generator cleanup claim owner kind is invalid")
    token = value["token"]
    match = _CLEANUP_CLAIM_PATTERN.fullmatch(path.name)
    name_token = (
        expected_token
        if expected_token is not None
        else (None if match is None else match.group("token"))
    )
    if not isinstance(token, str) or name_token != token:
        raise _invalid_record("generator cleanup claim token is invalid")
    request_digest = value["request_digest"]
    if not _is_sha256(request_digest):
        raise _invalid_record("generator cleanup claim request digest is invalid")
    phase = value["phase"]
    if not isinstance(phase, str):
        raise _invalid_record("generator cleanup claim phase is invalid")
    preparation_owner = _FileSnapshot.parse(
        value["preparation_owner"],
        field="cleanup_claim.preparation_owner",
    )
    raw_journal = value["journal"]
    raw_transaction_owner = value["transaction_owner"]
    journal = (
        None
        if raw_journal is None
        else _FileSnapshot.parse(raw_journal, field="cleanup_claim.journal")
    )
    transaction_owner = (
        None
        if raw_transaction_owner is None
        else _FileSnapshot.parse(
            raw_transaction_owner,
            field="cleanup_claim.transaction_owner",
        )
    )
    if owner_kind == "preparation":
        if phase != "preparing" or journal is not None or transaction_owner is not None:
            raise _invalid_record("generator preparation cleanup authority is invalid")
    else:
        try:
            _Phase(phase)
        except ValueError as exc:
            raise _invalid_record("generator cleanup claim phase is invalid") from exc
        if journal is None or transaction_owner is None:
            raise _invalid_record("generator transaction cleanup authority is incomplete")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_PRIVATE_TREE_ENTRIES:
        raise _invalid_record("generator cleanup claim entries are invalid")
    entries = tuple(
        _parse_cleanup_entry(item, index=index) for index, item in enumerate(raw_entries)
    )
    paths = [entry.path for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise _invalid_record("generator cleanup claim entry paths are not canonical")
    return _CleanupClaim(
        owner_kind=cast('Literal["preparation", "transaction"]', owner_kind),
        token=token,
        request_digest=cast("str", request_digest),
        root_identity=_identity_from_json(
            value["root_identity"],
            field="cleanup_claim.root_identity",
            expected_kind=stat.S_IFDIR,
        ),
        transaction_identity=_identity_from_json(
            value["transaction_identity"],
            field="cleanup_claim.transaction_identity",
            expected_kind=stat.S_IFDIR,
        ),
        phase=phase,
        preparation_owner=preparation_owner,
        journal=journal,
        transaction_owner=transaction_owner,
        entries=entries,
        identity=claim_identity,
    )


def _parse_cleanup_entry(value: object, *, index: int) -> _CleanupEntry:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "identity",
        "mode",
        "size",
        "content_sha256",
    }:
        raise _invalid_record(f"invalid cleanup entry {index}")
    value = cast("dict[str, object]", value)
    path = value["path"]
    if not isinstance(path, str):
        raise _invalid_record(f"invalid cleanup entry {index} path")
    relative = PurePosixPath(path)
    if (
        not path
        or path == "."
        or "\\" in path
        or relative.is_absolute()
        or relative.as_posix() != path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) > _MAX_PATH_DEPTH + 3
        or not _utf8_within(path, _MAX_PATH_BYTES + 64)
        or (os.name == "nt" and any(_is_unsafe_windows_component(part) for part in relative.parts))
    ):
        raise _invalid_record(f"invalid cleanup entry {index} path")
    identity = _identity_from_json(value["identity"], field=f"cleanup[{index}].identity")
    mode = value["mode"]
    size = value["size"]
    digest = value["content_sha256"]
    if type(mode) is not int or not 0 <= mode <= 0o7777:
        raise _invalid_record(f"invalid cleanup entry {index} mode")
    if identity.kind == stat.S_IFDIR:
        if size is not None or digest is not None:
            raise _invalid_record(f"invalid cleanup directory entry {index}")
    elif identity.kind == stat.S_IFREG:
        if (
            type(size) is not int
            or not 0 <= size <= _MAX_STAGED_BYTES
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or not _is_sha256(digest.removeprefix("sha256:"))
        ):
            raise _invalid_record(f"invalid cleanup file entry {index}")
    else:
        raise _invalid_record(f"unsupported cleanup entry {index}")
    return _CleanupEntry(
        path=path,
        identity=identity,
        mode=mode,
        size=size,
        content_sha256=digest,
    )


def _cleanup_entry_matches_file(entry: _CleanupEntry, expected: _FileSnapshot) -> bool:
    return (
        entry.identity == expected.identity
        and entry.mode == expected.mode
        and entry.size == expected.size
        and entry.content_sha256 == f"sha256:{expected.content_sha256}"
    )


def _capture_transaction_cleanup_authority(
    transaction: _Transaction,
) -> tuple[_CleanupEntry, ...]:
    owner_snapshot = _snapshot_regular(
        transaction.directory / _OWNER_FILE,
        label="generator owner marker",
    )
    loaded = _load_transaction(transaction.directory)
    if (
        loaded.directory_identity != transaction.directory_identity
        or loaded.token != transaction.token
        or loaded.request_digest != transaction.request_digest
        or loaded.phase is not transaction.phase
        or loaded.entry_sha256 != transaction.entry_sha256
    ):
        raise _conflict("Generator transaction changed before cleanup was sealed.")
    journal_snapshot = _snapshot_regular(
        transaction.directory / _JOURNAL_FILE,
        label="generator journal",
    )
    if journal_snapshot.identity != transaction.journal_identity:
        raise _conflict("Generator journal changed before cleanup was sealed.")
    expected_files: dict[str, _FileSnapshot] = {
        _JOURNAL_FILE: journal_snapshot,
        _OWNER_FILE: owner_snapshot,
    }
    next_receipt = _snapshot_optional(
        transaction.directory / _NEXT_RECEIPT_FILE,
        label="staged generator receipt",
    )
    if next_receipt is not None:
        if next_receipt != transaction.receipt_after:
            raise _conflict("Staged generator receipt changed before cleanup was sealed.")
        expected_files[_NEXT_RECEIPT_FILE] = next_receipt
    previous_receipt = _snapshot_optional(
        transaction.directory / _PREVIOUS_RECEIPT_FILE,
        label="previous generator receipt",
    )
    if previous_receipt is not None:
        if transaction.prior_receipt is None or previous_receipt != transaction.prior_receipt:
            raise _conflict("Previous generator receipt changed before cleanup was sealed.")
        expected_files[_PREVIOUS_RECEIPT_FILE] = previous_receipt
    for edit in transaction.edits:
        if edit.created_root_index is not None:
            continue
        for prefix, name, expected in (
            ("new", edit.stage_name, edit.after),
            ("backup", edit.backup_name, edit.before),
            ("quarantine", edit.quarantine_name, edit.after),
        ):
            current = _snapshot_optional(
                transaction.directory / prefix / name,
                label=f"generator private {prefix} entry",
            )
            if current is None:
                continue
            if expected is None or current != expected:
                raise _conflict(
                    "Generator private file changed before cleanup was sealed.",
                    paths=(f"{prefix}/{name}",),
                )
            expected_files[f"{prefix}/{name}"] = current
    present_tree_prefixes: list[str] = []
    for created_root in transaction.created_roots:
        prefix = f"trees/{created_root.stage_name}"
        stage = transaction.directory / "trees" / created_root.stage_name
        current = _snapshot_tree_optional(stage, label=f"staged {created_root.path}")
        if current is not None:
            if current != created_root.after:
                raise _conflict(
                    "Generator private tree changed before cleanup was sealed.",
                    paths=(created_root.path,),
                )
            present_tree_prefixes.append(prefix)
    try:
        _digest, entries = _capture_tree_authority(
            transaction.directory,
            expected=transaction.directory_identity,
            require_cleanup_access=True,
            entry_limit=_MAX_PRIVATE_TREE_ENTRIES,
        )
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc
    expected_directories = {
        "new": transaction.private_identities.new,
        "trees": transaction.private_identities.trees,
        "backup": transaction.private_identities.backup,
        "quarantine": transaction.private_identities.quarantine,
    }
    seen_directories: set[str] = set()
    seen_files: set[str] = set()
    seen_tree_roots: set[str] = set()
    for entry in entries:
        expected_file = expected_files.get(entry.path)
        if expected_file is not None:
            if not _cleanup_entry_matches_file(entry, expected_file):
                raise _conflict(
                    "Generator private file changed while cleanup was sealed.",
                    paths=(entry.path,),
                )
            seen_files.add(entry.path)
            continue
        expected_directory = expected_directories.get(entry.path)
        if expected_directory is not None:
            if entry.identity != expected_directory or (os.name != "nt" and entry.mode != 0o700):
                raise _conflict(
                    "Generator private directory changed while cleanup was sealed.",
                    paths=(entry.path,),
                )
            seen_directories.add(entry.path)
            continue
        if any(
            entry.path == prefix or entry.path.startswith(f"{prefix}/")
            for prefix in present_tree_prefixes
        ):
            if entry.path in present_tree_prefixes:
                seen_tree_roots.add(entry.path)
            continue
        raise _conflict(
            "Generator private state contains content absent from its transaction authority.",
            paths=(entry.path,),
        )
    if seen_directories != set(expected_directories):
        raise _conflict("Generator private directories changed before cleanup was sealed.")
    if seen_files != set(expected_files) or seen_tree_roots != set(present_tree_prefixes):
        raise _conflict("Generator private state changed while cleanup was sealed.")
    for created_root in transaction.created_roots:
        stage = transaction.directory / "trees" / created_root.stage_name
        if _entry_exists(stage):
            _require_tree_snapshot(stage, created_root.after, label=created_root.path)
    return entries


def _remove_cleanup_claim(
    state: Path,
    token: str,
    *,
    expected_transaction_identity: _Identity,
) -> None:
    path = _cleanup_claim_path(state, token)
    claim = _load_cleanup_claim(path)
    if claim.transaction_identity != expected_transaction_identity:
        raise _conflict("Generator cleanup claim belongs to another transaction.")
    _unlink_owned_file(
        path,
        expected=claim.identity,
        expected_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        label="generator cleanup claim",
    )


def _publish_preparation_cleanup_claim(
    state: Path,
    preparation: _PreparationAuthority,
    directory: Path,
) -> _CleanupClaim:
    staged = directory / _CLEANUP_CLAIM_FILE
    published = _cleanup_claim_path(state, preparation.token)
    current_staged = _snapshot_optional(staged, label="staged generator cleanup claim")
    current_published = _snapshot_optional(published, label="generator cleanup claim")
    if current_published is not None:
        if current_staged is not None:
            raise _conflict("Generator cleanup-claim authority is ambiguous.")
        claim = _load_cleanup_claim(published)
        _require_cleanup_claim_matches_preparation(claim, preparation)
        return claim
    if current_staged is None:
        try:
            _digest, entries = _capture_tree_authority(
                directory,
                expected=preparation.transaction_identity,
                require_cleanup_access=True,
                entry_limit=_MAX_PRIVATE_TREE_ENTRIES,
            )
        except GuardedTreePublicationError as exc:
            raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc
        _write_private_file(
            staged,
            _preparation_cleanup_claim_content(preparation, entries),
            expected_parent=preparation.transaction_identity,
        )
        current_staged = _snapshot_regular(staged, label="staged generator cleanup claim")
    claim = _load_cleanup_claim(staged, expected_token=preparation.token)
    _require_cleanup_claim_matches_preparation(claim, preparation)
    if current_staged.identity != claim.identity:
        raise _conflict("Generator cleanup-claim authority changed before publication.")
    _rename_no_replace(
        staged,
        published,
        expected=claim.identity,
        expected_source_parent=preparation.transaction_identity,
        expected_destination_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        label="generator cleanup claim",
    )
    _sync_directory(state)
    _sync_directory(directory)
    return claim


def _cleanup_preparation(
    state: Path,
    directory: Path,
    preparation: _PreparationAuthority,
) -> None:
    cleanup = state / f"cleanup-{preparation.token}"
    if directory != cleanup:
        _rename_no_replace(
            directory,
            cleanup,
            expected=preparation.transaction_identity,
            expected_source_parent=_capture_directory_identity(
                state,
                label="generator transaction state",
            ),
            expected_destination_parent=_capture_directory_identity(
                state,
                label="generator transaction state",
            ),
            label="generator preparation",
        )
        directory = cleanup
        _sync_directory(state)
        preparation = _load_preparation_authority(
            directory,
            expected_token=preparation.token,
        )
    claim = _publish_preparation_cleanup_claim(state, preparation, directory)
    owner = _publish_cleanup_owner(state, directory, claim)
    _remove_owned_directory(
        directory,
        expected=preparation.transaction_identity,
        expected_parent=_capture_directory_identity(
            state,
            label="generator transaction state",
        ),
        authority=claim.entries,
    )
    _sync_directory(state)
    _remove_cleanup_owner(state, preparation.token, expected=owner)
    _remove_cleanup_claim(
        state,
        preparation.token,
        expected_transaction_identity=preparation.transaction_identity,
    )


def _cleanup_transaction(state: Path, transaction: _Transaction) -> None:
    _require_directory_identity(transaction.directory, transaction.directory_identity)
    cleanup = state / f"cleanup-{transaction.token}"
    if transaction.directory != cleanup:
        _rename_no_replace(
            transaction.directory,
            cleanup,
            expected=transaction.directory_identity,
            expected_source_parent=transaction.private_identities.state,
            expected_destination_parent=transaction.private_identities.state,
            label="generator terminal transaction",
        )
        _fault("cleanup_claimed_before_sync")
        transaction.directory = cleanup
        _sync_directory(state)
        _fault("cleanup_claimed")
    claim = _publish_cleanup_claim(state, transaction)
    owner = _publish_cleanup_owner(state, transaction.directory, claim)
    _remove_owned_directory(
        transaction.directory,
        expected=transaction.directory_identity,
        expected_parent=transaction.private_identities.state,
        authority=claim.entries,
    )
    _fault("cleanup_directory_removed_before_sync")
    _sync_directory(state)
    _fault("cleanup_directory_removed")
    _remove_cleanup_owner(state, transaction.token, expected=owner)
    _remove_cleanup_claim(
        state,
        transaction.token,
        expected_transaction_identity=transaction.directory_identity,
    )
    _fault("transaction_cleaned")


def _settle_cleanups(state: Path) -> None:
    directories = {
        cast("re.Match[str]", _CLEANUP_PATTERN.fullmatch(path.name)).group("token"): path
        for path in _cleanup_directories(state)
    }
    claims = {
        cast("re.Match[str]", _CLEANUP_CLAIM_PATTERN.fullmatch(path.name)).group("token"): path
        for path in _cleanup_claim_files(state)
    }
    owners = {
        cast("re.Match[str]", _CLEANUP_OWNER_PATTERN.fullmatch(path.name)).group("token"): path
        for path in _cleanup_owner_files(state)
    }
    state_identity = _capture_directory_identity(state, label="generator transaction state")
    for token in sorted(directories.keys() | claims.keys() | owners.keys()):
        directory = directories.get(token)
        claim = claims.get(token)
        owner_path = owners.get(token)
        if claim is None:
            if directory is None or owner_path is not None:
                raise _conflict(
                    "Generator cleanup state lacks its durable claim authority.",
                    paths=tuple(path.name for path in (directory, owner_path) if path is not None),
                )
            try:
                transaction = _load_transaction(directory)
            except GeneratorTransactionError as invalid_record:
                try:
                    preparation = _load_preparation_authority(
                        directory,
                        expected_token=token,
                    )
                except GeneratorTransactionError:
                    raise _conflict(
                        "A generator cleanup name conflicts with its transaction.",
                        paths=(directory.name,),
                    ) from invalid_record
                _cleanup_preparation(state, directory, preparation)
            else:
                if transaction.token != token:
                    raise _conflict(
                        "A generator cleanup name conflicts with its transaction.",
                        paths=(directory.name,),
                    )
                _cleanup_transaction(state, transaction)
            continue
        loaded_claim = _load_cleanup_claim(claim)
        if loaded_claim.token != token:
            raise _conflict(
                "A generator cleanup claim conflicts with its name.",
                paths=(claim.name,),
            )
        cleanup_owner = _require_cleanup_owner(state, directory, loaded_claim)
        if directory is not None:
            _remove_owned_directory(
                directory,
                expected=loaded_claim.transaction_identity,
                expected_parent=state_identity,
                authority=loaded_claim.entries,
            )
            _sync_directory(state)
        if cleanup_owner is not None:
            _remove_cleanup_owner(state, token, expected=cleanup_owner)
        _unlink_owned_file(
            claim,
            expected=loaded_claim.identity,
            expected_parent=state_identity,
            label="generator cleanup claim",
        )


def _settle_preparations(
    root: Path,
    state: Path,
    root_identity: _Identity,
) -> None:
    for path in _prepare_directories(state):
        try:
            transaction = _load_transaction(path)
        except GeneratorTransactionError as invalid_record:
            try:
                preparation = _load_preparation_authority(path)
            except GeneratorTransactionError:
                raise _conflict(
                    "An unauthenticated generator preparation was preserved.",
                    paths=(path.name,),
                ) from invalid_record
            _require_root(root, root_identity)
            if preparation.root_identity != root_identity:
                raise _conflict(
                    "A generator preparation belongs to another project root.",
                    paths=(path.name,),
                ) from None
            _cleanup_preparation(state, path, preparation)
            continue
        if transaction.phase is not _Phase.PREPARED:
            raise _conflict(
                "An unpromoted generator preparation contains mutation intent.",
                paths=(path.name,),
            )
        _cleanup_transaction(state, transaction)


def _prepare_directories(state: Path) -> tuple[Path, ...]:
    return _matching_state_directories(state, pattern=_PREPARE_PATTERN)


def _cleanup_directories(state: Path) -> tuple[Path, ...]:
    return _matching_state_directories(state, pattern=_CLEANUP_PATTERN)


def _cleanup_claim_files(state: Path) -> tuple[Path, ...]:
    return _matching_state_directories(state, pattern=_CLEANUP_CLAIM_PATTERN)


def _cleanup_owner_files(state: Path) -> tuple[Path, ...]:
    return _matching_state_directories(state, pattern=_CLEANUP_OWNER_PATTERN)


def _preparation_owner_files(state: Path) -> tuple[Path, ...]:
    return _matching_state_directories(state, pattern=_PREPARATION_OWNER_PATTERN)


def _require_no_orphan_owner_markers(state: Path) -> None:
    preparation_owners = _preparation_owner_files(state)
    cleanup_owners = _cleanup_owner_files(state)
    if not preparation_owners and not cleanup_owners:
        return
    raise _conflict(
        "Generator transaction state contains an owner without recoverable work.",
        paths=tuple(path.name for path in (*preparation_owners, *cleanup_owners)),
    )


def _matching_state_directories(
    state: Path,
    *,
    pattern: re.Pattern[str],
) -> tuple[Path, ...]:
    result: list[Path] = []
    try:
        with os.scandir(state) as entries:
            for count, entry in enumerate(entries, start=1):
                if count > _MAX_STATE_ENTRIES:
                    raise GeneratorTransactionError(
                        "state_census_limit",
                        "Generator transaction state exceeds its bounded entry census.",
                    )
                if pattern.fullmatch(entry.name):
                    result.append(state / entry.name)
    except OSError as exc:
        raise GeneratorTransactionError(
            "state_inspection_failed",
            "Could not inspect generator transaction state.",
        ) from exc
    return tuple(sorted(result))


def _load_transaction(directory: Path) -> _Transaction:
    marker = _load_owner_marker(directory, journal_required=True)
    assert marker is not None
    directory_identity, journal_identity, token, manifest_digest = marker
    _require_directory_identity(directory, directory_identity)
    journal_path = directory / _JOURNAL_FILE
    content, observed_journal_identity = _read_regular_bytes(
        journal_path,
        limit=_MAX_JOURNAL_BYTES,
        label="generator transaction journal",
    )
    if observed_journal_identity != journal_identity:
        raise _invalid_record("generator journal identity changed")
    valid_bytes = 0
    previous: str | None = None
    manifest: dict[str, object] | None = None
    phase: _Phase | None = None
    sequence = -1
    events: list[dict[str, object]] = []
    for raw_line in content.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            break
        value, digest = _parse_journal_line(raw_line, previous=previous, sequence=sequence + 1)
        if sequence == -1:
            manifest = value
            if digest != manifest_digest:
                raise _invalid_record("generator owner marker does not bind its manifest")
            phase = _Phase.PREPARED
        else:
            if value.get("kind") != "event" or value.get("token") != token:
                raise _invalid_record("invalid generator journal event")
            events.append(value)
            event_phase = value.get("phase")
            if event_phase is not None:
                try:
                    candidate = _Phase(event_phase)
                except (TypeError, ValueError) as exc:
                    raise _invalid_record("unknown generator journal phase") from exc
                _require_phase_successor(cast("_Phase", phase), candidate)
                phase = candidate
        valid_bytes += len(raw_line)
        previous = digest
        sequence += 1
    if manifest is None or phase is None or previous is None:
        raise _invalid_record("generator journal has no complete manifest")
    parsed = _parse_manifest(manifest, directory=directory)
    if parsed[0] != token or parsed[1] != directory_identity:
        raise _invalid_record("generator manifest does not match its owner marker")
    preparation = _load_preparation_authority(directory, expected_token=token)
    if (
        preparation.transaction_identity != directory_identity
        or preparation.request_digest != parsed[2]
        or preparation.root_identity != parsed[3]
    ):
        raise _invalid_record("generator manifest does not match its preparation owner")
    _validate_journal_events(events, edit_count=len(parsed[4]), root_count=len(parsed[6]))
    return _Transaction(
        directory=directory,
        directory_identity=directory_identity,
        journal_identity=journal_identity,
        private_identities=parsed[9],
        token=token,
        request_digest=parsed[2],
        root_identity=parsed[3],
        edits=parsed[4],
        preconditions=parsed[5],
        created_roots=parsed[6],
        prior_receipt=parsed[7],
        receipt_after=parsed[8],
        phase=phase,
        sequence=sequence,
        entry_sha256=previous,
        valid_bytes=valid_bytes,
    )


def _load_owner_marker(
    directory: Path,
    *,
    journal_required: bool,
) -> tuple[_Identity, _Identity, str, str] | None:
    path = directory / _OWNER_FILE
    if not _entry_exists(path):
        if journal_required:
            raise _invalid_record("generator transaction owner marker is missing")
        return None
    value = _read_json_file(path, limit=4096, label="generator owner marker")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "token",
        "transaction_identity",
        "journal_identity",
        "manifest_sha256",
    }:
        raise _invalid_record("invalid generator owner marker")
    value = cast("dict[str, object]", value)
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        raise _invalid_record("unsupported generator owner-marker schema")
    token = value["token"]
    manifest = value["manifest_sha256"]
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise _invalid_record("invalid generator transaction token")
    if not _is_sha256(manifest):
        raise _invalid_record("invalid generator manifest digest")
    expected_prepare = _PREPARE_PATTERN.fullmatch(directory.name)
    expected_cleanup = _CLEANUP_PATTERN.fullmatch(directory.name)
    if directory.name != _ACTIVE_DIRECTORY and not (
        (expected_prepare is not None and expected_prepare.group("token") == token)
        or (expected_cleanup is not None and expected_cleanup.group("token") == token)
    ):
        raise _invalid_record("generator preparation name does not match its token")
    return (
        _identity_from_json(
            value["transaction_identity"],
            field="transaction_identity",
            expected_kind=stat.S_IFDIR,
        ),
        _identity_from_json(
            value["journal_identity"],
            field="journal_identity",
            expected_kind=stat.S_IFREG,
        ),
        token,
        cast("str", manifest),
    )


def _parse_manifest(
    value: dict[str, object],
    *,
    directory: Path,
) -> tuple[
    str,
    _Identity,
    str,
    _Identity,
    tuple[_EditRecord, ...],
    tuple[_PreconditionRecord, ...],
    tuple[_CreatedRootRecord, ...],
    _FileSnapshot | None,
    _FileSnapshot,
    _PrivateIdentities,
]:
    expected = {
        "schema_version",
        "kind",
        "sequence",
        "previous_sha256",
        "token",
        "request_digest",
        "root_identity",
        "transaction_identity",
        "private_identities",
        "phase",
        "edits",
        "preconditions",
        "created_roots",
        "prior_receipt",
        "receipt_after",
    }
    if set(value) != expected or value.get("kind") != "manifest":
        raise _invalid_record("generator manifest has unexpected fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        raise _invalid_record("unsupported generator manifest schema")
    if value["sequence"] != 0 or value["previous_sha256"] is not None:
        raise _invalid_record("generator manifest sequence is invalid")
    if value["phase"] != _Phase.PREPARED.value:
        raise _invalid_record("generator manifest has invalid initial phase")
    token = value["token"]
    request_digest = value["request_digest"]
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise _invalid_record("invalid generator manifest token")
    if not _is_sha256(request_digest):
        raise _invalid_record("invalid generator request digest")
    edits_raw = value["edits"]
    preconditions_raw = value["preconditions"]
    created_roots_raw = value["created_roots"]
    if not isinstance(edits_raw, list) or not 1 <= len(edits_raw) <= _MAX_EDITS:
        raise _invalid_record("invalid generator manifest edits")
    if not isinstance(preconditions_raw, list) or len(preconditions_raw) > _MAX_PRECONDITIONS:
        raise _invalid_record("invalid generator manifest preconditions")
    if not isinstance(created_roots_raw, list) or len(created_roots_raw) > _MAX_EDITS:
        raise _invalid_record("invalid generator manifest created roots")
    edits = tuple(_parse_edit_record(item, index=index) for index, item in enumerate(edits_raw))
    preconditions = tuple(
        _parse_precondition_record(item, index=index)
        for index, item in enumerate(preconditions_raw)
    )
    created_roots = tuple(
        _parse_created_root_record(item, index=index, edit_count=len(edits))
        for index, item in enumerate(created_roots_raw)
    )
    prior_receipt_raw = value["prior_receipt"]
    prior_receipt = (
        None
        if prior_receipt_raw is None
        else _FileSnapshot.parse(prior_receipt_raw, field="prior_receipt")
    )
    receipt_after = _FileSnapshot.parse(value["receipt_after"], field="receipt_after")
    referenced_indexes = [
        edit_index for created_root in created_roots for edit_index in created_root.edit_indexes
    ]
    if len(referenced_indexes) != len(set(referenced_indexes)):
        raise _invalid_record("generator created roots contain duplicate edit membership")
    for edit in edits:
        if edit.created_root_index is None:
            if edit.index in referenced_indexes:
                raise _invalid_record("generator edit has conflicting created-root membership")
            continue
        if (
            edit.created_root_index >= len(created_roots)
            or edit.index not in created_roots[edit.created_root_index].edit_indexes
        ):
            raise _invalid_record("generator edit created-root membership is invalid")
        created_root = created_roots[edit.created_root_index]
        try:
            relative = PurePosixPath(edit.path).relative_to(PurePosixPath(created_root.path))
        except ValueError as exc:
            raise _invalid_record("generator edit is outside its created root") from exc
        if not relative.parts or edit.operation != "create" or edit.before is not None:
            raise _invalid_record("generator created-root edit authority is invalid")
    root_keys = sorted(_normalized_path(item.path) for item in created_roots)
    for index, key in enumerate(root_keys):
        for later_key in root_keys[index + 1 :]:
            if later_key == key or later_key.startswith(f"{key}/"):
                raise _invalid_record("generator created-root topology is invalid")
    transaction_identity = _identity_from_json(
        value["transaction_identity"],
        field="transaction_identity",
        expected_kind=stat.S_IFDIR,
    )
    _require_directory_identity(directory, transaction_identity)
    private_identities = _parse_private_identities(
        value["private_identities"],
        directory=directory,
    )
    return (
        token,
        transaction_identity,
        cast("str", request_digest),
        _identity_from_json(
            value["root_identity"],
            field="root_identity",
            expected_kind=stat.S_IFDIR,
        ),
        edits,
        preconditions,
        created_roots,
        prior_receipt,
        receipt_after,
        private_identities,
    )


def _parse_private_identities(value: object, *, directory: Path) -> _PrivateIdentities:
    expected_fields = {"state", "new", "trees", "backup", "quarantine"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise _invalid_record("generator private-directory authority is invalid")
    value = cast("dict[str, object]", value)
    identities = _PrivateIdentities(
        state=_identity_from_json(
            value["state"],
            field="private_identities.state",
            expected_kind=stat.S_IFDIR,
        ),
        new=_identity_from_json(
            value["new"],
            field="private_identities.new",
            expected_kind=stat.S_IFDIR,
        ),
        trees=_identity_from_json(
            value["trees"],
            field="private_identities.trees",
            expected_kind=stat.S_IFDIR,
        ),
        backup=_identity_from_json(
            value["backup"],
            field="private_identities.backup",
            expected_kind=stat.S_IFDIR,
        ),
        quarantine=_identity_from_json(
            value["quarantine"],
            field="private_identities.quarantine",
            expected_kind=stat.S_IFDIR,
        ),
    )
    paths = {
        "state": directory.parent,
        "new": directory / "new",
        "trees": directory / "trees",
        "backup": directory / "backup",
        "quarantine": directory / "quarantine",
    }
    for name, path in paths.items():
        _require_directory_identity(path, getattr(identities, name))
    return identities


def _parse_edit_record(value: object, *, index: int) -> _EditRecord:
    if not isinstance(value, dict) or set(value) != {
        "index",
        "path",
        "operation",
        "before",
        "after",
        "parent_identity",
        "created_root_index",
        "stage_name",
        "backup_name",
        "quarantine_name",
    }:
        raise _invalid_record("invalid generator edit record")
    value = cast("dict[str, object]", value)
    if type(value["index"]) is not int or value["index"] != index:
        raise _invalid_record("invalid generator edit index")
    path = _parse_record_path(value["path"], field="edit.path")
    operation = value["operation"]
    if operation not in {"create", "update_region"}:
        raise _invalid_record("invalid generator edit operation")
    before_raw = value["before"]
    before = None if before_raw is None else _FileSnapshot.parse(before_raw, field="before")
    if (operation == "create") != (before is None):
        raise _invalid_record("generator edit preimage conflicts with its operation")
    expected_name = f"{index:04d}"
    created_root_index = value["created_root_index"]
    if created_root_index is not None and (
        type(created_root_index) is not int or created_root_index < 0
    ):
        raise _invalid_record("invalid generator edit created-root index")
    for field in ("stage_name", "backup_name", "quarantine_name"):
        if value[field] != expected_name:
            raise _invalid_record(f"invalid generator {field}")
    return _EditRecord(
        index=index,
        path=path,
        operation=cast("Literal['create', 'update_region']", operation),
        before=before,
        after=_FileSnapshot.parse(value["after"], field="after"),
        parent_identity=_identity_from_json(
            value["parent_identity"],
            field="edit.parent_identity",
            expected_kind=stat.S_IFDIR,
        ),
        created_root_index=created_root_index,
        stage_name=expected_name,
        backup_name=expected_name,
        quarantine_name=expected_name,
    )


def _parse_precondition_record(value: object, *, index: int) -> _PreconditionRecord:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "snapshot",
        "parent_identity",
    }:
        raise _invalid_record(f"invalid generator precondition record {index}")
    value = cast("dict[str, object]", value)
    return _PreconditionRecord(
        path=_parse_record_path(value["path"], field="precondition.path"),
        snapshot=_FileSnapshot.parse(value["snapshot"], field="precondition"),
        parent_identity=_identity_from_json(
            value["parent_identity"],
            field="precondition.parent_identity",
            expected_kind=stat.S_IFDIR,
        ),
    )


def _parse_created_root_record(
    value: object,
    *,
    index: int,
    edit_count: int,
) -> _CreatedRootRecord:
    if not isinstance(value, dict) or set(value) != {
        "index",
        "path",
        "stage_name",
        "parent_identity",
        "after",
        "edit_indexes",
    }:
        raise _invalid_record("invalid generator created-root record")
    value = cast("dict[str, object]", value)
    if type(value["index"]) is not int or value["index"] != index:
        raise _invalid_record("invalid generator created-root index")
    stage_name = f"{index:04d}"
    if value["stage_name"] != stage_name:
        raise _invalid_record("invalid generator created-root stage name")
    edit_indexes = value["edit_indexes"]
    if (
        not isinstance(edit_indexes, list)
        or not edit_indexes
        or len(edit_indexes) > edit_count
        or any(type(item) is not int or not 0 <= item < edit_count for item in edit_indexes)
        or len(edit_indexes) != len(set(edit_indexes))
    ):
        raise _invalid_record("invalid generator created-root edit indexes")
    return _CreatedRootRecord(
        index=index,
        path=_parse_record_path(value["path"], field="created_root.path"),
        stage_name=stage_name,
        parent_identity=_identity_from_json(
            value["parent_identity"],
            field="created_root.parent_identity",
            expected_kind=stat.S_IFDIR,
        ),
        after=_TreeSnapshot.parse(value["after"], field="created_root.after"),
        edit_indexes=tuple(cast("list[int]", edit_indexes)),
    )


def _append_phase(transaction: _Transaction, phase: _Phase) -> None:
    _require_phase_successor(transaction.phase, phase)
    _append_event(transaction, "phase", None, phase=phase)
    transaction.phase = phase
    _fault(f"phase:{phase.value}")


def _append_event(
    transaction: _Transaction,
    event: str,
    index: int | None,
    *,
    phase: _Phase | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "event",
        "sequence": transaction.sequence + 1,
        "previous_sha256": transaction.entry_sha256,
        "token": transaction.token,
        "event": event,
        "index": index,
        "phase": None if phase is None else phase.value,
    }
    entry, digest = _journal_entry(payload)
    if transaction.valid_bytes + len(entry) > _MAX_JOURNAL_BYTES:
        raise GeneratorTransactionError(
            "journal_limit",
            "Generator transaction exhausted its bounded journal.",
        )
    journal = transaction.directory / _JOURNAL_FILE
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        with _pinned_parent(
            transaction.directory,
            expected=transaction.directory_identity,
        ) as parent:
            descriptor = os.open(
                journal if parent.descriptor is None else _JOURNAL_FILE,
                flags,
                **({} if parent.descriptor is None else {"dir_fd": parent.descriptor}),
            )
            try:
                current = os.fstat(descriptor)
                if (
                    _capture_stable_identity(current, descriptor=descriptor)
                    != transaction.journal_identity
                ):
                    raise _invalid_record("generator journal identity changed before append")
                os.ftruncate(descriptor, transaction.valid_bytes)
                os.lseek(descriptor, transaction.valid_bytes, os.SEEK_SET)
                _write_all(descriptor, entry)
                os.fsync(descriptor)
            finally:
                _close_descriptor(descriptor)
            parent.assert_unchanged()
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc
    _fault(
        "journal_appended:"
        f"{event}:"
        f"{phase.value if phase is not None else 'none' if index is None else index}"
    )
    transaction.sequence += 1
    transaction.entry_sha256 = digest
    transaction.valid_bytes += len(entry)


def _parse_journal_line(
    line: bytes,
    *,
    previous: str | None,
    sequence: int,
) -> tuple[dict[str, object], str]:
    try:
        value = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise _invalid_record("generator journal contains invalid JSON") from exc
    if not isinstance(value, dict) or "entry_sha256" not in value:
        raise _invalid_record("generator journal entry is invalid")
    supplied = value.pop("entry_sha256")
    if not _is_sha256(supplied) or supplied != _sha256(_canonical_json(value)):
        raise _invalid_record("generator journal entry digest does not match")
    if type(value.get("sequence")) is not int or value["sequence"] != sequence:
        raise _invalid_record("generator journal sequence is not contiguous")
    if value.get("previous_sha256") != previous:
        raise _invalid_record("generator journal hash chain is not contiguous")
    return cast("dict[str, object]", value), cast("str", supplied)


def _validate_journal_events(
    events: list[dict[str, object]],
    *,
    edit_count: int,
    root_count: int,
) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "sequence",
        "previous_sha256",
        "token",
        "event",
        "index",
        "phase",
    }
    edit_events = {
        "original_moved",
        "new_published",
        "new_removed",
        "original_restored",
    }
    root_events = {"created_root_published", "created_root_restored"}
    commit_events = {"original_moved", "new_published", "created_root_published"}
    rollback_events = {
        "new_removed",
        "original_restored",
        "created_root_restored",
    }
    current_phase = _Phase.PREPARED
    observed: set[tuple[str, int]] = set()
    for value in events:
        if set(value) != expected_fields or (
            type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION
        ):
            raise _invalid_record("generator journal event has unexpected fields")
        event = value["event"]
        index = value["index"]
        event_phase = value["phase"]
        if event == "phase":
            if index is not None or not isinstance(event_phase, str):
                raise _invalid_record("generator phase event is invalid")
            try:
                next_phase = _Phase(value["phase"])
            except (TypeError, ValueError) as exc:
                raise _invalid_record("generator phase event is unknown") from exc
            _require_phase_successor(current_phase, next_phase)
            current_phase = next_phase
            continue
        if event_phase is not None or not isinstance(event, str):
            raise _invalid_record("generator mutation event is invalid")
        if event in edit_events:
            limit = edit_count
        elif event in root_events:
            limit = root_count
        else:
            raise _invalid_record("generator journal event type is unknown")
        if type(index) is not int or not 0 <= index < limit:
            raise _invalid_record("generator journal event index is invalid")
        event_key = (event, index)
        if event_key in observed:
            raise _invalid_record("generator journal repeats a mutation event")
        observed.add(event_key)
        if (event in commit_events and current_phase is not _Phase.COMMITTING) or (
            event in rollback_events and current_phase is not _Phase.ROLLING_BACK
        ):
            raise _invalid_record("generator mutation event occurs in an invalid phase")


def _journal_entry(payload: dict[str, object]) -> tuple[bytes, str]:
    canonical = _canonical_json(payload)
    digest = _sha256(canonical)
    return _canonical_json({**payload, "entry_sha256": digest}) + b"\n", digest


def _require_phase_successor(previous: _Phase, current: _Phase) -> None:
    allowed = {
        _Phase.PREPARED: {_Phase.COMMITTING},
        _Phase.COMMITTING: {_Phase.ROLLING_BACK, _Phase.COMMITTED},
        _Phase.ROLLING_BACK: {_Phase.ROLLED_BACK},
        _Phase.COMMITTED: set(),
        _Phase.ROLLED_BACK: set(),
    }
    if current not in allowed[previous]:
        raise _invalid_record("generator transaction phase transition is invalid")


def _validate_request(request: GeneratorTransactionRequest) -> None:
    if type(request) is not GeneratorTransactionRequest:
        raise GeneratorTransactionError(
            "invalid_request",
            "Generator transaction request has an unsupported type.",
        )
    scalar_limits = (
        (request.schema_version, 32),
        (request.slice_name, 256),
        (request.tool_name, 256),
        (request.effect, 64),
        (request.authoring_state, 256),
    )
    if any(
        not isinstance(value, str) or not _utf8_within(value, limit)
        for value, limit in scalar_limits
    ):
        raise GeneratorTransactionError(
            "invalid_request",
            "Generator transaction identity fields exceed their bounds.",
        )
    if request.schema_version != APP_MANIFEST_SCHEMA_VERSION:
        raise GeneratorTransactionError(
            "unsupported_plan_schema",
            "Generator transaction request uses an unsupported plan schema.",
        )
    if (
        type(request.edits) is not tuple
        or type(request.preconditions) is not tuple
        or type(request.verification_commands) is not tuple
        or len(request.verification_commands) > 64
        or any(
            not isinstance(command, str) or not _utf8_within(command, 4096)
            for command in request.verification_commands
        )
        or sum(len(command.encode("utf-8")) for command in request.verification_commands)
        > 64 * 1024
    ):
        raise GeneratorTransactionError(
            "invalid_request",
            "Generator transaction collections exceed their bounds.",
        )
    validate_generator_transaction_collection_bounds(
        edit_count=len(request.edits),
        precondition_count=len(request.preconditions),
    )
    total = 0
    edit_paths: dict[str, str] = {}
    all_paths: dict[str, str] = {}
    for edit in request.edits:
        if type(edit) is not GeneratorTransactionEdit or type(edit.content) is not bytes:
            raise GeneratorTransactionError(
                "invalid_edit",
                "Generator transaction edit has an unsupported type.",
            )
        _validate_relative_path(edit.path)
        key = _normalized_path(edit.path)
        previous = edit_paths.get(key)
        if previous is not None:
            raise GeneratorTransactionError(
                "duplicate_path",
                "Generator plan contains duplicate or aliased edit paths.",
                paths=(previous, edit.path),
            )
        edit_paths[key] = edit.path
        all_paths[key] = edit.path
        if edit.operation not in {"create", "update_region"}:
            raise GeneratorTransactionError(
                "invalid_operation", "Generator edit operation is invalid."
            )
        total += len(edit.content)
        if total > _MAX_STAGED_BYTES:
            raise GeneratorTransactionError(
                "staged_content_limit",
                "Generator plan exceeds the aggregate staged-content limit.",
            )
        if not _is_sha256(edit.content_sha256) or _sha256(edit.content) != edit.content_sha256:
            raise GeneratorTransactionError(
                "content_digest_mismatch",
                "Generator edit content does not match its digest.",
                paths=(edit.path,),
            )
        if edit.operation == "create" and edit.preimage_sha256 is not None:
            raise GeneratorTransactionError(
                "unexpected_preimage",
                "Generator create edit has an unexpected preimage.",
                paths=(edit.path,),
            )
        if edit.operation == "update_region" and not _is_sha256(edit.preimage_sha256):
            raise GeneratorTransactionError(
                "missing_preimage",
                "Generator update edit is missing a valid preimage digest.",
                paths=(edit.path,),
            )
    ordered_keys = sorted(edit_paths)
    for index, key in enumerate(ordered_keys):
        prefix = f"{key}/"
        for later in ordered_keys[index + 1 :]:
            if later.startswith(prefix):
                raise GeneratorTransactionError(
                    "path_topology_conflict",
                    "Generator edit paths have an impossible ancestor relationship.",
                    paths=(edit_paths[key], edit_paths[later]),
                )
    seen_preconditions: set[str] = set()
    edits = {edit.path: edit for edit in request.edits}
    for precondition in request.preconditions:
        if type(precondition) is not GeneratorTransactionPrecondition:
            raise GeneratorTransactionError(
                "invalid_precondition",
                "Generator transaction precondition has an unsupported type.",
            )
        _validate_relative_path(precondition.path)
        key = _normalized_path(precondition.path)
        if key in seen_preconditions:
            raise GeneratorTransactionError(
                "duplicate_precondition",
                "Generator plan contains duplicate or aliased preconditions.",
                paths=(precondition.path,),
            )
        seen_preconditions.add(key)
        previous = all_paths.get(key)
        if previous is not None and previous != precondition.path:
            raise GeneratorTransactionError(
                "path_alias",
                "Generator plan contains conflicting path spellings.",
                paths=(previous, precondition.path),
            )
        all_paths[key] = precondition.path
        if not _is_sha256(precondition.content_sha256):
            raise GeneratorTransactionError(
                "invalid_precondition_digest",
                "Generator precondition digest is invalid.",
                paths=(precondition.path,),
            )
        edit = edits.get(precondition.path)
        if edit is not None and edit.preimage_sha256 != precondition.content_sha256:
            raise GeneratorTransactionError(
                "precondition_conflict",
                "Generator edit and precondition digests conflict.",
                paths=(precondition.path,),
            )
    # Prove the private record is bounded before allocating transaction state.
    identity_document = {
        "digest": request.digest,
        "edits": [
            [edit.path, edit.operation, edit.content_sha256, edit.preimage_sha256]
            for edit in request.edits
        ],
        "preconditions": [[item.path, item.content_sha256] for item in request.preconditions],
    }
    if len(_canonical_json(identity_document)) > _MAX_RECEIPT_BYTES:
        raise GeneratorTransactionError(
            "record_limit",
            "Generator plan exceeds the encoded transaction-record limit.",
        )


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise GeneratorTransactionError("invalid_path", "Generator path must be non-empty.")
    if not _utf8_within(value, _MAX_PATH_BYTES):
        raise GeneratorTransactionError(
            "invalid_path",
            "Generator path is outside the supported project-relative bounds.",
            paths=(value,),
        )
    pure = PurePosixPath(value)
    if (
        len(pure.parts) > _MAX_PATH_DEPTH
        or pure.is_absolute()
        or not pure.parts
        or value != pure.as_posix()
        or _normalized_path(pure.parts[0]) == ".cayu"
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(
            len(part.encode("utf-8")) > 255
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or (os.name == "nt" and _is_unsafe_windows_component(part))
            for part in pure.parts
        )
    ):
        raise GeneratorTransactionError(
            "invalid_path",
            "Generator path is outside the supported project-relative bounds.",
            paths=(value,),
        )


def _target_path(root: Path, relative: str) -> Path:
    _validate_relative_path(relative)
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts:
        current = current / part
        try:
            value = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(value.st_mode):
            raise GeneratorTransactionError(
                "unsafe_path",
                f"generated path contains a symbolic link: {current.relative_to(root).as_posix()}",
            )
        if _is_windows_reparse_point(value):
            raise GeneratorTransactionError(
                "unsafe_path",
                f"generated path contains a reparse point: {current.relative_to(root).as_posix()}",
            )
    try:
        current.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise GeneratorTransactionError(
            "invalid_path",
            "Generator path escapes the project root.",
            paths=(relative,),
        ) from exc
    return current


def _capture_parent_authority(
    root: Path,
    parent: Path,
    *,
    path: str,
) -> tuple[_Identity | None, str | None]:
    try:
        relative_parts = parent.relative_to(root).parts
    except ValueError as exc:
        raise GeneratorTransactionError(
            "invalid_parent", "Generator parent escapes the root."
        ) from exc
    current = root
    for index, part in enumerate(relative_parts):
        _reject_name_alias(current, part)
        candidate = current / part
        try:
            value = candidate.stat(follow_symlinks=False)
        except FileNotFoundError:
            missing = PurePosixPath(*relative_parts[: index + 1]).as_posix()
            return None, missing
        except OSError as exc:
            raise GeneratorTransactionError(
                "path_inspection_failed",
                "Could not inspect a generator parent.",
                paths=(path,),
            ) from exc
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_windows_reparse_point(value)
        ):
            raise GeneratorTransactionError(
                "unsafe_parent",
                "Generator target parent is not an ordinary directory.",
                paths=(path,),
            )
        current = candidate
    return _capture_directory_identity(parent, label=f"parent of {path}"), None


def _capture_existing_parent_identity(root: Path, parent: Path, *, path: str) -> _Identity:
    identity, missing = _capture_parent_authority(root, parent, path=path)
    if identity is None or missing is not None:
        raise GeneratorTransactionError(
            "parent_missing",
            "Generator target parent is missing.",
            paths=(path,),
        )
    return identity


def _require_existing_parent(
    root: Path,
    parent: Path,
    *,
    path: str,
    expected: _Identity,
) -> None:
    current = _capture_existing_parent_identity(root, parent, path=path)
    if current != expected:
        raise _conflict(
            "A generator target parent changed while the transaction was active.",
            paths=(path,),
        )


def _reject_component_aliases(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        _reject_name_alias(current, part)
        current = current / part


def _reject_name_alias(parent: Path, name: str) -> None:
    aliases: list[str] = []
    with os.scandir(parent) as entries:
        for count, entry in enumerate(entries, start=1):
            if count > _MAX_STATE_ENTRIES:
                raise GeneratorTransactionError(
                    "directory_census_limit",
                    "Generator path inspection exceeds its bounded directory census.",
                )
            if entry.name != name and _normalized_path(entry.name) == _normalized_path(name):
                aliases.append(entry.name)
    if aliases:
        raise GeneratorTransactionError(
            "path_alias",
            "Generator path has a conflicting case or Unicode alias.",
            paths=tuple(aliases),
        )


def _snapshot_optional(path: Path, *, label: str) -> _FileSnapshot | None:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GeneratorTransactionError(
            "path_inspection_failed",
            "Could not inspect a generator path.",
            paths=(label,),
        ) from exc
    if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
        raise GeneratorTransactionError(
            "unsafe_path",
            "Generator path is a link or reparse point.",
            paths=(label,),
        )
    if not stat.S_ISREG(value.st_mode):
        raise GeneratorTransactionError(
            "unsupported_entry",
            "Generator path is not an ordinary file.",
            paths=(label,),
        )
    return _snapshot_regular(path, label=label)


def _snapshot_regular(path: Path, *, label: str) -> _FileSnapshot:
    content, identity = _read_regular_bytes(path, limit=_MAX_STAGED_BYTES, label=label)
    value = path.stat(follow_symlinks=False)
    if _capture_stable_identity(value, path=path) != identity:
        raise GeneratorTransactionError(
            "path_changed",
            "Generator file changed while it was inspected.",
            paths=(label,),
        )
    return _FileSnapshot(
        identity=identity,
        content_sha256=_sha256(content),
        mode=stat.S_IMODE(value.st_mode),
        size=len(content),
    )


def _snapshot_tree_optional(path: Path, *, label: str) -> _TreeSnapshot | None:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GeneratorTransactionError(
            "path_inspection_failed",
            "Could not inspect a generator directory.",
            paths=(label,),
        ) from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_windows_reparse_point(value)
    ):
        raise GeneratorTransactionError(
            "unsupported_entry",
            "Generator-created root is not an ordinary directory.",
            paths=(label,),
        )
    return _snapshot_tree(path, label=label)


def _snapshot_tree(path: Path, *, label: str) -> _TreeSnapshot:
    root_identity = _capture_directory_identity(path, label=label)
    entries: list[dict[str, object]] = []
    pending: list[tuple[Path, PurePosixPath]] = [(path, PurePosixPath("."))]
    discovered = 1
    total_bytes = 0
    while pending:
        directory, relative = pending.pop()
        directory_identity = _capture_directory_identity(directory, label=label)
        directory_value = directory.stat(follow_symlinks=False)
        if _capture_stable_identity(directory_value, path=directory) != directory_identity:
            raise _conflict(
                "Generator-created directory changed during inspection.",
                paths=(label,),
            )
        entries.append(
            {
                "path": relative.as_posix(),
                "kind": "directory",
                "identity": directory_identity.as_json(),
                "mode": stat.S_IMODE(directory_value.st_mode),
            }
        )
        children: list[os.DirEntry[str]] = []
        try:
            with os.scandir(directory) as iterator:
                for child in iterator:
                    discovered += 1
                    if discovered > _MAX_CREATED_TREE_ENTRIES:
                        raise GeneratorTransactionError(
                            "tree_entry_limit",
                            "Generator-created directory exceeds its bounded entry limit.",
                            paths=(label,),
                        )
                    children.append(child)
        except OSError as exc:
            raise GeneratorTransactionError(
                "path_inspection_failed",
                "Could not inspect a generator-created directory.",
                paths=(label,),
            ) from exc
        for child in reversed(sorted(children, key=lambda item: item.name)):
            child_path = directory / child.name
            child_relative = (
                PurePosixPath(child.name)
                if relative == PurePosixPath(".")
                else relative / child.name
            )
            value = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
                raise _conflict(
                    "Generator-created directory contains a link or reparse point.",
                    paths=(child_relative.as_posix(),),
                )
            if stat.S_ISDIR(value.st_mode):
                pending.append((child_path, child_relative))
                continue
            if not stat.S_ISREG(value.st_mode):
                raise _conflict(
                    "Generator-created directory contains an unsupported entry.",
                    paths=(child_relative.as_posix(),),
                )
            snapshot = _snapshot_regular(child_path, label=child_relative.as_posix())
            total_bytes += snapshot.size
            if total_bytes > _MAX_STAGED_BYTES:
                raise GeneratorTransactionError(
                    "tree_byte_limit",
                    "Generator-created directory exceeds its bounded byte limit.",
                    paths=(label,),
                )
            entries.append(
                {
                    "path": child_relative.as_posix(),
                    "kind": "file",
                    "snapshot": snapshot.payload(),
                }
            )
    if _capture_directory_identity(path, label=label) != root_identity:
        raise _conflict(
            "Generator-created directory changed during inspection.",
            paths=(label,),
        )
    entries.sort(key=lambda item: cast("str", item["path"]))
    return _TreeSnapshot(
        identity=root_identity,
        fingerprint=_sha256(_canonical_json(entries)),
        entries=len(entries),
    )


def _tree_snapshot_matches(
    current: _TreeSnapshot | None,
    expected: _TreeSnapshot,
) -> bool:
    return current == expected


def _require_tree_snapshot(path: Path, expected: _TreeSnapshot, *, label: str) -> None:
    if not _tree_snapshot_matches(_snapshot_tree_optional(path, label=label), expected):
        raise _conflict(
            "Generator-created directory changed while the transaction was active.",
            paths=(label,),
        )


def _read_regular_bytes(path: Path, *, limit: int, label: str) -> tuple[bytes, _Identity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GeneratorTransactionError(
            "path_open_failed",
            "Could not open a generator-owned regular file.",
            paths=(label,),
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GeneratorTransactionError(
                "unsupported_entry",
                "Generator path is not an ordinary file.",
                paths=(label,),
            )
        identity = _capture_stable_identity(before, descriptor=descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise GeneratorTransactionError(
                    "file_size_limit",
                    "Generator file exceeds its bounded inspection limit.",
                    paths=(label,),
                )
        after = os.fstat(descriptor)
        if _capture_stable_identity(after, descriptor=descriptor) != identity:
            raise GeneratorTransactionError(
                "path_changed",
                "Generator file changed while it was inspected.",
                paths=(label,),
            )
        return b"".join(chunks), identity
    finally:
        _close_descriptor(descriptor)


def _create_owned_directory(
    path: Path,
    *,
    expected_parent: _Identity,
    mode: int,
    label: str,
) -> _Identity:
    try:
        with _pinned_parent(path.parent, expected=expected_parent) as parent:
            if parent.entry_stat(path.name) is not None:
                raise _conflict(
                    f"{label.capitalize()} destination is no longer absent.",
                    paths=(path.name,),
                )
            windows_security_error: OSError | None = None
            if os.name == "nt":
                windows_security_error = _create_private_windows_directory(path)
            elif parent.descriptor is None:
                path.mkdir(mode=mode)
            else:
                os.mkdir(path.name, mode=mode, dir_fd=parent.descriptor)
            current = parent.entry_stat(path.name)
            if (
                current is None
                or not stat.S_ISDIR(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or _is_windows_reparse_point(current)
            ):
                raise _conflict(
                    f"{label.capitalize()} was not created as an ordinary directory.",
                    paths=(path.name,),
                )
            identity = parent.entry_identity(path.name, value=current)
            if os.name == "nt":
                _assert_windows_directory_dacl_is_protected(path)
            parent.assert_unchanged()
            if windows_security_error is not None:
                raise windows_security_error
            return identity
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _write_private_file(
    path: Path,
    content: bytes,
    *,
    expected_parent: _Identity,
) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        with _pinned_parent(path.parent, expected=expected_parent) as parent:
            if parent.entry_stat(path.name) is not None:
                raise _conflict(
                    "Generator private file destination is no longer absent.",
                    paths=(path.name,),
                )
            descriptor = os.open(
                path if parent.descriptor is None else path.name,
                flags,
                0o600,
                **({} if parent.descriptor is None else {"dir_fd": parent.descriptor}),
            )
            try:
                _write_all(descriptor, content)
                os.fsync(descriptor)
                written = os.fstat(descriptor)
                written_identity = _capture_stable_identity(
                    written,
                    path=path if parent.descriptor is None else None,
                    descriptor=None if parent.descriptor is None else descriptor,
                )
            finally:
                _close_descriptor(descriptor)
            current = parent.entry_stat(path.name)
            if (
                current is None
                or not stat.S_ISREG(current.st_mode)
                or parent.entry_identity(path.name, value=current) != written_identity
            ):
                raise _conflict(
                    "Generator private file changed during creation.",
                    paths=(path.name,),
                )
            parent.assert_unchanged()
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write")
        view = view[written:]


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    expected: _Identity,
    expected_source_parent: _Identity,
    expected_destination_parent: _Identity,
    label: str,
    unexpected_policy: _UnexpectedRenamePolicy = _UnexpectedRenamePolicy.RESTORE_SOURCE,
) -> None:
    if (
        _capture_directory_identity(source.parent, label=f"parent of {label}")
        != expected_source_parent
        or _capture_directory_identity(
            destination.parent,
            label=f"destination parent of {label}",
        )
        != expected_destination_parent
    ):
        raise _conflict(f"{label.capitalize()} parent changed before its namespace transition.")
    try:
        with _pinned_parent(
            source.parent,
            expected=expected_source_parent,
        ) as source_parent:
            destination_context = (
                nullcontext(source_parent)
                if destination.parent == source.parent
                else _pinned_parent(
                    destination.parent,
                    expected=expected_destination_parent,
                )
            )
            with destination_context as destination_parent:
                current = source_parent.entry_stat(source.name)
                if (
                    current is None
                    or source_parent.entry_identity(source.name, value=current) != expected
                    or stat.S_ISLNK(current.st_mode)
                    or _is_windows_reparse_point(current)
                    or not (stat.S_ISREG(current.st_mode) or stat.S_ISDIR(current.st_mode))
                ):
                    raise _conflict(
                        f"{label.capitalize()} changed before its namespace transition."
                    )
                if destination_parent.entry_stat(destination.name) is not None:
                    raise _conflict(
                        f"{label.capitalize()} destination is no longer absent.",
                        paths=(destination.name,),
                    )
                pinned_descriptor: int | None = None
                if source_parent.descriptor is not None:
                    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                    if stat.S_ISDIR(current.st_mode):
                        flags |= getattr(os, "O_DIRECTORY", 0)
                    pinned_descriptor = os.open(
                        source.name,
                        flags,
                        dir_fd=source_parent.descriptor,
                    )
                try:
                    if pinned_descriptor is not None and (
                        _capture_stable_identity(
                            os.fstat(pinned_descriptor),
                            descriptor=pinned_descriptor,
                        )
                        != expected
                    ):
                        raise _conflict(
                            f"{label.capitalize()} changed while its source was pinned."
                        )
                    _rename_names_no_replace(
                        source_parent_descriptor=source_parent.descriptor,
                        source_parent_path=source_parent.path,
                        source_name=source.name,
                        destination_parent_descriptor=destination_parent.descriptor,
                        destination_parent_path=destination_parent.path,
                        destination_name=destination.name,
                    )
                    moved = destination_parent.entry_stat(destination.name)
                    pinned_matches = pinned_descriptor is None or (
                        _capture_stable_identity(
                            os.fstat(pinned_descriptor),
                            descriptor=pinned_descriptor,
                        )
                        == expected
                    )
                    if (
                        moved is None
                        or destination_parent.entry_identity(
                            destination.name,
                            value=moved,
                        )
                        != expected
                        or not pinned_matches
                    ):
                        conflict = _conflict(
                            f"{label.capitalize()} changed during its namespace transition.",
                            paths=(source.name, destination.name),
                        )
                        if unexpected_policy is _UnexpectedRenamePolicy.RESTORE_SOURCE:
                            _restore_unexpected_rename(
                                source_parent=source_parent,
                                destination_parent=destination_parent,
                                source_name=source.name,
                                destination_name=destination.name,
                                moved=moved,
                                error=conflict,
                            )
                        else:
                            _sync_preserved_unexpected_rename(
                                source_parent=source_parent,
                                destination_parent=destination_parent,
                                error=conflict,
                            )
                        raise conflict
                    source_parent.sync()
                    if destination_parent is not source_parent:
                        destination_parent.sync()
                finally:
                    if pinned_descriptor is not None:
                        _close_descriptor(pinned_descriptor)
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _restore_unexpected_rename(
    *,
    source_parent: _Parent,
    destination_parent: _Parent,
    source_name: str,
    destination_name: str,
    moved: os.stat_result | None,
    error: GeneratorTransactionError,
) -> None:
    """Restore an unexpected moved object without replacing newer namespace state."""

    if moved is None or source_parent.entry_stat(source_name) is not None:
        return
    moved_identity = destination_parent.entry_identity(destination_name, value=moved)
    try:
        _rename_names_no_replace(
            source_parent_descriptor=destination_parent.descriptor,
            source_parent_path=destination_parent.path,
            source_name=destination_name,
            destination_parent_descriptor=source_parent.descriptor,
            destination_parent_path=source_parent.path,
            destination_name=source_name,
        )
        restored = source_parent.entry_stat(source_name)
        remaining = destination_parent.entry_stat(destination_name)
        if (
            restored is None
            or source_parent.entry_identity(source_name, value=restored) != moved_identity
            or remaining is not None
        ):
            raise _conflict(
                "An unexpected generator rename changed before it could be restored.",
                paths=(source_name, destination_name),
            )
        source_parent.sync()
        if destination_parent is not source_parent:
            destination_parent.sync()
    except BaseException as restoration_error:
        if restoration_error.__cause__ is None and restoration_error.__context__ is error:
            restoration_error.__context__ = None
        _raise_primary(error, [restoration_error])


def _sync_preserved_unexpected_rename(
    *,
    source_parent: _Parent,
    destination_parent: _Parent,
    error: GeneratorTransactionError,
) -> None:
    """Durably preserve an unauthenticated object found at a public destination."""

    settlement_errors: list[BaseException] = []
    for parent in dict.fromkeys((source_parent, destination_parent)):
        try:
            parent.sync()
        except BaseException as settlement_error:
            settlement_errors.append(settlement_error)
    if settlement_errors:
        _raise_primary(error, settlement_errors)


def _rename_names_no_replace(
    *,
    source_parent_descriptor: int | None,
    source_parent_path: Path,
    source_name: str,
    destination_parent_descriptor: int | None,
    destination_parent_path: Path,
    destination_name: str,
) -> None:
    if os.name == "nt":
        os.rename(
            source_parent_path / source_name,
            destination_parent_path / destination_name,
        )
        return
    if source_parent_descriptor is None or destination_parent_descriptor is None:
        raise AssertionError("POSIX generator rename parents are not pinned")
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise GeneratorTransactionError(
                "no_replace_unavailable",
                "This platform lacks atomic no-replace generator publication.",
            )
        result = renameat2(
            source_parent_descriptor,
            os.fsencode(source_name),
            destination_parent_descriptor,
            os.fsencode(destination_name),
            1,
        )
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise GeneratorTransactionError(
                "no_replace_unavailable",
                "This platform lacks atomic no-replace generator publication.",
            )
        result = renameatx_np(
            source_parent_descriptor,
            os.fsencode(source_name),
            destination_parent_descriptor,
            os.fsencode(destination_name),
            0x00000004,
        )
    else:
        raise GeneratorTransactionError(
            "no_replace_unavailable",
            "This platform lacks atomic no-replace generator publication.",
        )
    if result != 0:
        error_code = ctypes.get_errno()
        raise OSError(error_code, os.strerror(error_code), destination_name)


def _unlink_owned_file(
    path: Path,
    *,
    expected: _Identity,
    expected_parent: _Identity,
    label: str,
) -> None:
    try:
        with _pinned_parent(path.parent, expected=expected_parent) as parent:
            current = parent.entry_stat(path.name)
            if (
                current is None
                or not stat.S_ISREG(current.st_mode)
                or parent.entry_identity(path.name, value=current) != expected
            ):
                raise _conflict(
                    f"{label.capitalize()} changed before cleanup.",
                    paths=(path.name,),
                )
            if parent.descriptor is None:
                path.unlink()
            else:
                os.unlink(path.name, dir_fd=parent.descriptor)
            if parent.entry_stat(path.name) is not None:
                raise _conflict(
                    f"{label.capitalize()} remained after cleanup.",
                    paths=(path.name,),
                )
            parent.sync()
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _sync_file(path: Path, *, expected: _Identity) -> None:
    if _capture_file_identity(path, label="generator file") != expected:
        raise _conflict("Generator file changed before synchronization.", paths=(path.name,))
    if os.name == "nt":
        _sync_windows_path(path, directory=False)
    else:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if _capture_stable_identity(os.fstat(descriptor), descriptor=descriptor) != expected:
                raise _conflict(
                    "Generator file changed before synchronization.",
                    paths=(path.name,),
                )
            os.fsync(descriptor)
        finally:
            _close_descriptor(descriptor)
    if _capture_file_identity(path, label="generator file") != expected:
        raise _conflict("Generator file changed during synchronization.", paths=(path.name,))


def _finalize_windows_published_path(path: Path, *, expected: _Identity) -> None:
    if os.name != "nt":
        return
    try:
        with _windows_directory_namespace_fence(path):
            before = path.stat(follow_symlinks=False)
            if (
                stat.S_ISLNK(before.st_mode)
                or _is_windows_reparse_point(before)
                or _capture_stable_identity(before, path=path) != expected
            ):
                raise _conflict(
                    "Generator after-image changed before Windows permissions were restored.",
                    paths=(path.name,),
                )
            _restore_windows_directory_inheritance(path)
            after = path.stat(follow_symlinks=False)
            if _capture_stable_identity(after, path=path) != expected:
                raise _conflict(
                    "Generator after-image changed while Windows permissions were restored.",
                    paths=(path.name,),
                )
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _sync_directory(path: Path) -> None:
    identity = _capture_directory_identity(path, label="generator directory")
    if os.name == "nt":
        _sync_windows_path(path, directory=True)
    else:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        try:
            if _capture_stable_identity(os.fstat(descriptor), descriptor=descriptor) != identity:
                raise _conflict("Generator directory changed before synchronization.")
            os.fsync(descriptor)
        finally:
            _close_descriptor(descriptor)


def _sync_tree(path: Path) -> None:
    directories: list[Path] = []
    pending = [path]
    count = 0
    while pending:
        directory = pending.pop()
        directories.append(directory)
        children: list[os.DirEntry[str]] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > _MAX_CREATED_TREE_ENTRIES:
                    raise GeneratorTransactionError(
                        "tree_entry_limit",
                        "Generator-created directory exceeds its bounded entry limit.",
                        paths=(path.name,),
                    )
                children.append(entry)
        for entry in children:
            candidate = directory / entry.name
            value = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
                raise _conflict(
                    "Generator-created directory contains a link or reparse point.",
                    paths=(entry.name,),
                )
            if stat.S_ISREG(value.st_mode):
                _sync_file(
                    candidate,
                    expected=_capture_file_identity(candidate, label=entry.name),
                )
            elif stat.S_ISDIR(value.st_mode):
                pending.append(candidate)
            else:
                raise _conflict(
                    "Generator-created directory contains an unsupported entry.",
                    paths=(entry.name,),
                )
    for directory in reversed(directories):
        _sync_directory(directory)


def _state_directory(root: Path, *, create: bool) -> Path | None:
    root_identity = _capture_directory_identity(root, label="project root")
    cayu = root / ".cayu"
    if not _entry_exists(cayu):
        if not create:
            return None
        cayu_identity = _create_owned_directory(
            cayu,
            expected_parent=root_identity,
            mode=0o700,
            label=".cayu state root",
        )
        _sync_directory(root)
    else:
        cayu_identity = _capture_directory_identity(cayu, label=".cayu state root")
    _require_root(root, root_identity)
    state = cayu / _STATE_DIRECTORY
    if not _entry_exists(state):
        if not create:
            return None
        state_identity = _create_owned_directory(
            state,
            expected_parent=cayu_identity,
            mode=0o700,
            label="generator transaction state",
        )
        _sync_directory(cayu)
    else:
        state_identity = _capture_directory_identity(
            state,
            label="generator transaction state",
        )
    _require_directory_identity(cayu, cayu_identity)
    _require_root(root, root_identity)
    if state_identity.device != root_identity.device:
        raise GeneratorTransactionError(
            "cross_device_state",
            "Generator transaction state must share the project filesystem.",
        )
    return state


def _require_state_census(state: Path) -> None:
    seen: dict[str, str] = {}
    with os.scandir(state) as entries:
        for count, entry in enumerate(entries, start=1):
            if count > _MAX_STATE_ENTRIES:
                raise GeneratorTransactionError(
                    "state_census_limit",
                    "Generator transaction state exceeds its bounded entry census.",
                )
            key = _normalized_path(entry.name)
            previous = seen.get(key)
            if previous is not None and previous != entry.name:
                raise _conflict(
                    "Generator transaction state contains aliased entries.",
                    paths=(previous, entry.name),
                )
            seen[key] = entry.name
            allowed = (
                entry.name in {_ACTIVE_DIRECTORY, _RECEIPT_FILE}
                or _PREPARE_PATTERN.fullmatch(entry.name) is not None
                or _PREPARATION_OWNER_PATTERN.fullmatch(entry.name) is not None
                or _CLEANUP_PATTERN.fullmatch(entry.name) is not None
                or _CLEANUP_CLAIM_PATTERN.fullmatch(entry.name) is not None
                or _CLEANUP_OWNER_PATTERN.fullmatch(entry.name) is not None
            )
            if not allowed:
                raise _conflict(
                    "Generator transaction state contains an unknown entry.",
                    paths=(entry.name,),
                )
            value = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
                raise _conflict(
                    "Generator transaction state contains a link or reparse point.",
                    paths=(entry.name,),
                )
            if (
                entry.name == _RECEIPT_FILE
                or _PREPARATION_OWNER_PATTERN.fullmatch(entry.name)
                or _CLEANUP_CLAIM_PATTERN.fullmatch(entry.name)
                or _CLEANUP_OWNER_PATTERN.fullmatch(entry.name)
            ):
                if not stat.S_ISREG(value.st_mode):
                    raise _conflict(
                        "Generator transaction receipt is not a regular file.",
                        paths=(entry.name,),
                    )
            elif not stat.S_ISDIR(value.st_mode):
                raise _conflict(
                    "Generator transaction owner is not a directory.",
                    paths=(entry.name,),
                )


def _remove_owned_directory(
    path: Path,
    *,
    expected: _Identity,
    expected_parent: _Identity,
    authority: tuple[_CleanupEntry, ...],
) -> None:
    authority_by_path = {entry.path: entry for entry in authority}
    if len(authority_by_path) != len(authority):
        raise _invalid_record("generator cleanup authority contains duplicate paths")
    try:
        with _pinned_parent(path.parent, expected=expected_parent) as parent:
            current = parent.entry_stat(path.name)
            if (
                current is None
                or not stat.S_ISDIR(current.st_mode)
                or parent.entry_identity(path.name, value=current) != expected
            ):
                raise _conflict(
                    "Generator cleanup directory changed before deletion.",
                    paths=(path.name,),
                )
            if os.name == "nt":
                _delete_windows_entry_by_handle(
                    path,
                    expected=expected,
                    authority=authority_by_path,
                )
                _fault("cleanup_owner_removed")
                parent.assert_unchanged()
                parent.sync()
                return
            if parent.descriptor is None:
                raise AssertionError("POSIX generator cleanup parent is not pinned")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path.name, flags, dir_fd=parent.descriptor)
            try:
                opened = os.fstat(descriptor)
                if _capture_stable_identity(opened, descriptor=descriptor) != expected:
                    raise _conflict(
                        "Generator cleanup directory changed while it was pinned.",
                        paths=(path.name,),
                    )
                _remove_directory_contents_from_fd(
                    descriptor,
                    path=path,
                    flags=flags,
                    authority=authority_by_path,
                )
                if (
                    _capture_stable_identity(os.fstat(descriptor), descriptor=descriptor)
                    != expected
                ):
                    raise _conflict(
                        "Generator cleanup directory changed during deletion.",
                        paths=(path.name,),
                    )
            finally:
                _close_descriptor(descriptor)
            _fault("cleanup_owner_removed")
            final = parent.entry_stat(path.name)
            if (
                final is None
                or parent.entry_identity(path.name, value=final) != expected
                or not stat.S_ISDIR(final.st_mode)
            ):
                raise _conflict(
                    "Generator cleanup directory changed before final deletion.",
                    paths=(path.name,),
                )
            os.rmdir(path.name, dir_fd=parent.descriptor)
            parent.sync()
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _canonical_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise GeneratorTransactionError(
            "root_unavailable",
            "Generator project root is unavailable.",
        ) from exc
    _capture_directory_identity(resolved, label="project root")
    return resolved


def _capture_directory_identity(path: Path, *, label: str) -> _Identity:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise GeneratorTransactionError(
            "directory_unavailable",
            f"{label.capitalize()} is unavailable.",
            paths=(path.name,),
        ) from exc
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or _is_windows_reparse_point(value)
    ):
        raise GeneratorTransactionError(
            "unsafe_directory",
            f"{label.capitalize()} must be an ordinary directory.",
            paths=(path.name,),
        )
    try:
        return _capture_stable_identity(value, path=path)
    except GuardedTreePublicationError as exc:
        raise GeneratorTransactionError(exc.code, str(exc), paths=exc.paths) from exc


def _capture_file_identity(path: Path, *, label: str) -> _Identity:
    snapshot = _snapshot_regular(path, label=label)
    return snapshot.identity


def _require_directory_identity(path: Path, expected: _Identity) -> None:
    if _capture_directory_identity(path, label="generator transaction") != expected:
        raise _conflict("Generator transaction directory identity changed.", paths=(path.name,))


def _require_root(root: Path, expected: _Identity) -> None:
    if _capture_directory_identity(root, label="project root") != expected:
        raise _conflict("Generator project root changed while the transaction was active.")


def _entry_exists(path: Path) -> bool:
    try:
        path.stat(follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _snapshot_matches(current: _FileSnapshot | None, expected: _FileSnapshot | None) -> bool:
    return current == expected


def _identity_from_json(
    value: object,
    *,
    field: str,
    expected_kind: int | None = None,
) -> _Identity:
    if not isinstance(value, list) or len(value) != 4:
        raise _invalid_record(f"invalid {field} identity")
    identity = _Identity(
        device=cast("int", value[0]),
        inode=cast("int", value[1]),
        kind=cast("int", value[2]),
        incarnation=cast("int", value[3]),
    )
    try:
        _require_identity_record_bound(identity)
    except GeneratorTransactionError as exc:
        raise _invalid_record(f"invalid {field} identity") from exc
    if expected_kind is not None and identity.kind != expected_kind:
        raise _invalid_record(f"invalid {field} identity kind")
    return identity


def _parse_record_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise _invalid_record(f"invalid {field}")
    try:
        _validate_relative_path(value)
    except GeneratorTransactionError as exc:
        raise _invalid_record(f"invalid {field}") from exc
    return value


def _read_json_file(path: Path, *, limit: int, label: str) -> object:
    content, _identity = _read_regular_bytes(path, limit=limit, label=label)
    try:
        return json.loads(content)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _invalid_record(f"{label} contains invalid JSON") from exc


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _normalized_path(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


def _utf8_within(value: str, limit: int) -> bool:
    if len(value) > limit:
        return False
    total = 0
    try:
        for offset in range(0, len(value), 256):
            total += len(value[offset : offset + 256].encode("utf-8"))
            if total > limit:
                return False
        return True
    except UnicodeEncodeError:
        return False


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _invalid_record(message: str) -> GeneratorTransactionError:
    return GeneratorTransactionError("invalid_transaction_record", message)


def _conflict(message: str, *, paths: tuple[str, ...] = ()) -> GeneratorTransactionError:
    return GeneratorTransactionError("transaction_conflict", message, paths=paths)


def _partition_process_control(
    error: BaseException,
) -> tuple[BaseException | None, BaseException | None]:
    if isinstance(error, _PROCESS_CONTROL_SIGNALS):
        return error, None
    if isinstance(error, BaseExceptionGroup):
        signal: BaseException | None = None
        residuals: list[BaseException] = []
        for child in error.exceptions:
            if signal is None:
                child_signal, child_residual = _partition_process_control(child)
                if child_signal is not None:
                    signal = child_signal
                    if child_residual is not None:
                        residuals.append(child_residual)
                    continue
            residuals.append(child)
        residual = None if not residuals else error.derive(residuals)
        return signal, residual
    return None, error


def _raise_primary(primary: BaseException, settlement_errors: list[BaseException]) -> NoReturn:
    signal, residual_primary = _partition_process_control(primary)
    evidence: list[BaseException] = []
    if signal is None:
        evidence.append(primary)
    elif residual_primary is not None:
        evidence.append(residual_primary)
    for settlement_error in settlement_errors:
        if signal is None:
            settlement_signal, residual = _partition_process_control(settlement_error)
            if settlement_signal is not None:
                signal = settlement_signal
                if residual is not None:
                    evidence.append(residual)
                continue
        evidence.append(settlement_error)
    if signal is None:
        if not settlement_errors:
            raise primary
        prior = primary.__cause__
        if prior is None and primary.__context__ is not None and not primary.__suppress_context__:
            prior = primary.__context__
        ordinary_evidence = [*(() if prior is None else (prior,)), *settlement_errors]
        cause: BaseException = (
            ordinary_evidence[0]
            if len(ordinary_evidence) == 1
            else BaseExceptionGroup(
                "Generator transaction settlement failures.",
                ordinary_evidence,
            )
        )
        raise primary from cause
    signal_prior = signal.__cause__
    if signal_prior is None and signal.__context__ is not None and not signal.__suppress_context__:
        signal_prior = signal.__context__
    if signal_prior is not None and signal_prior not in evidence:
        evidence.append(signal_prior)
    if not evidence:
        raise signal
    cause: BaseException = (
        evidence[0]
        if len(evidence) == 1
        else BaseExceptionGroup(
            "Generator transaction settlement failures.",
            evidence,
        )
    )
    raise signal from cause

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank
from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult


class WorkspaceRevisionObservationLimits(BaseModel):
    """Hard collection limits for workspace revision evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_paths: StrictInt = Field(default=4096, ge=1, le=100_000)
    max_path_bytes: StrictInt = Field(default=4096, ge=1, le=65_536)
    max_file_bytes: StrictInt = Field(default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_total_file_bytes: StrictInt = Field(
        default=64 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
    )
    max_manifest_bytes: StrictInt = Field(
        default=1024 * 1024,
        ge=1024,
        le=16 * 1024 * 1024,
    )


class WorkspaceRevisionObservationStatus(StrEnum):
    """Outcome of one optional workspace revision observation."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    TRUNCATED = "truncated"


class WorkspaceRevisionDeltaStatus(StrEnum):
    """Outcome of comparing two workspace observations."""

    CHANGED = "changed"
    NO_CHANGE = "no_change"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    TRUNCATED = "truncated"


class WorkspaceIdentity(BaseModel):
    """Stable workspace identity plus the adapter that observed it."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    workspace_id: str
    observer: str

    @field_validator("workspace_id", "observer")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class WorkspacePathRevision(BaseModel):
    """Content-free Git/workspace state for one bounded relative path."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    staged: str | None = None
    working_tree: str | None = None
    untracked: StrictBool = False
    ignored: StrictBool = False
    present: StrictBool | None = None
    tracked: StrictBool | None = None
    kind: Literal["file", "symlink", "submodule", "unknown"] = "unknown"
    content_sha256: str | None = None
    index_object_id: str | None = None
    index_mode: str | None = None
    renamed_from: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = require_durable_clean_nonblank(value, "path")
        if not _safe_relative_path(path):
            raise ValueError("Workspace revision path must be relative and traversal-free.")
        if str(PurePosixPath(path)) != path:
            raise ValueError("Workspace revision path must use canonical POSIX spelling.")
        return path

    @field_validator(
        "staged",
        "working_tree",
        "content_sha256",
        "index_object_id",
        "index_mode",
        "renamed_from",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        text = require_durable_clean_nonblank(value, info.field_name)
        if info.field_name == "renamed_from" and not _safe_relative_path(text):
            raise ValueError("Workspace revision rename source must be traversal-free.")
        if info.field_name == "renamed_from" and str(PurePosixPath(text)) != text:
            raise ValueError("Workspace revision rename source must use canonical POSIX spelling.")
        if info.field_name == "index_mode" and (
            len(text) != 6 or any(char not in "01234567" for char in text)
        ):
            raise ValueError("Workspace revision index mode must be a six-digit octal mode.")
        return text


class WorkspaceRevisionObservation(BaseModel):
    """Bounded evidence for one observed workspace state.

    ``revision`` is an observation digest, not an optimistic file token and not
    a restorable filesystem snapshot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: WorkspaceIdentity
    status: WorkspaceRevisionObservationStatus
    revision: str | None = None
    head_revision: str | None = None
    branch: str | None = None
    path_scope: Literal["complete", "changed"] = "complete"
    paths: tuple[WorkspacePathRevision, ...] = ()
    total_paths: StrictInt = Field(default=0, ge=0)
    detail_code: str | None = None

    @field_validator("revision", "head_revision", "branch", "detail_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_observation_shape(self) -> WorkspaceRevisionObservation:
        if len({entry.path for entry in self.paths}) != len(self.paths):
            raise ValueError("Workspace observation paths must be unique.")
        if self.total_paths < len(self.paths):
            raise ValueError("Workspace observation total_paths cannot be smaller than paths.")
        if self.status is WorkspaceRevisionObservationStatus.SUPPORTED:
            if self.revision is None:
                raise ValueError("A supported workspace observation requires a revision.")
            if self.total_paths != len(self.paths):
                raise ValueError("A supported workspace observation must include every path.")
        elif self.revision is not None or self.paths:
            raise ValueError("An incomplete workspace observation cannot carry revision evidence.")
        return self


class WorkspacePathRevisionDelta(BaseModel):
    """One content-free path change between two observations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    change: Literal["added", "modified", "deleted", "renamed"]
    renamed_from: str | None = None

    @field_validator("path", "renamed_from")
    @classmethod
    def validate_path_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        path = require_durable_clean_nonblank(value, info.field_name)
        if not _safe_relative_path(path):
            raise ValueError("Workspace revision delta path must be traversal-free.")
        return path


class WorkspaceRevisionDelta(BaseModel):
    """Bounded attributable change between two observed revisions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: WorkspaceIdentity
    status: WorkspaceRevisionDeltaStatus
    before_revision: str | None = None
    after_revision: str | None = None
    paths: tuple[WorkspacePathRevisionDelta, ...] = ()
    total_paths: StrictInt = Field(default=0, ge=0)
    head_changed: StrictBool = False
    branch_changed: StrictBool = False
    detail_code: str | None = None

    @field_validator("before_revision", "after_revision", "detail_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_delta_shape(self) -> WorkspaceRevisionDelta:
        if self.total_paths < len(self.paths):
            raise ValueError("Workspace delta total_paths cannot be smaller than paths.")
        if self.status in {
            WorkspaceRevisionDeltaStatus.CHANGED,
            WorkspaceRevisionDeltaStatus.NO_CHANGE,
        }:
            if self.before_revision is None or self.after_revision is None:
                raise ValueError("A complete workspace delta requires both revisions.")
            if self.total_paths != len(self.paths):
                raise ValueError("A complete workspace delta must include every path.")
        elif self.paths:
            raise ValueError("An incomplete workspace delta cannot carry path evidence.")
        if self.status is WorkspaceRevisionDeltaStatus.NO_CHANGE and (
            self.before_revision != self.after_revision
            or self.paths
            or self.head_changed
            or self.branch_changed
        ):
            raise ValueError("A no-change workspace delta cannot contain change evidence.")
        return self


def unsupported_workspace_revision(
    *,
    workspace_id: str,
    observer: str,
) -> WorkspaceRevisionObservation:
    return WorkspaceRevisionObservation(
        identity=WorkspaceIdentity(workspace_id=workspace_id, observer=observer),
        status=WorkspaceRevisionObservationStatus.UNSUPPORTED,
        detail_code="revision_observation_unsupported",
    )


async def observe_deterministic_workspace(
    workspace: Workspace,
    *,
    observer: str,
    limits: WorkspaceRevisionObservationLimits,
) -> WorkspaceRevisionObservation:
    """Observe one backend-neutral workspace through its bounded public API."""

    identity = WorkspaceIdentity(workspace_id=workspace.id, observer=observer)
    try:
        raw_listed = await workspace.list("**/*", limit=limits.max_paths + 1)
        if type(raw_listed) is not WorkspaceListResult:
            raise TypeError("Workspace list returned an invalid result.")
        listed = WorkspaceListResult(
            paths=raw_listed.paths,
            total_count=raw_listed.total_count,
            truncated=raw_listed.truncated,
        )
    except Exception:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="workspace_list_failed",
        )
    if listed.truncated or len(listed.paths) > limits.max_paths:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.TRUNCATED,
            total_paths=listed.total_count or len(listed.paths),
            detail_code="path_count_limit_exceeded",
        )

    revisions: list[WorkspacePathRevision] = []
    manifest: list[dict[str, object]] = []
    observed_file_bytes = 0
    for path in sorted(listed.paths):
        if not _safe_relative_path(path):
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.FAILED,
                total_paths=len(listed.paths),
                detail_code="unsafe_workspace_path",
            )
        if len(path.encode("utf-8")) > limits.max_path_bytes:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.TRUNCATED,
                total_paths=len(listed.paths),
                detail_code="path_byte_limit_exceeded",
            )
        remaining_file_bytes = limits.max_total_file_bytes - observed_file_bytes
        try:
            raw_read = await workspace.read_bytes(
                path,
                # Workspace reads require a positive limit. Probe one byte when
                # the aggregate is exactly exhausted so trailing empty files
                # remain valid while any additional content still fails closed.
                max_bytes=min(limits.max_file_bytes, max(1, remaining_file_bytes)),
            )
            if type(raw_read) is not WorkspaceReadResult:
                raise TypeError("Workspace read returned an invalid result.")
            read = WorkspaceReadResult(
                content=raw_read.content,
                total_bytes=raw_read.total_bytes,
                truncated=raw_read.truncated,
                offset=raw_read.offset,
                revision=raw_read.revision,
                sha256=raw_read.sha256,
                source_bytes_read=raw_read.source_bytes_read,
                redaction_truncated=raw_read.redaction_truncated,
            )
        except Exception:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.INCOMPLETE,
                total_paths=len(listed.paths),
                detail_code="workspace_file_read_failed",
            )
        if read.truncated:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.TRUNCATED,
                total_paths=len(listed.paths),
                detail_code=(
                    "total_file_byte_limit_exceeded"
                    if remaining_file_bytes < limits.max_file_bytes
                    else "file_byte_limit_exceeded"
                ),
            )
        observed_file_bytes += read.total_bytes
        if observed_file_bytes > limits.max_total_file_bytes:
            return WorkspaceRevisionObservation(
                identity=identity,
                status=WorkspaceRevisionObservationStatus.TRUNCATED,
                total_paths=len(listed.paths),
                detail_code="total_file_byte_limit_exceeded",
            )
        digest = hashlib.sha256(read.content).hexdigest()
        revision = WorkspacePathRevision(
            path=path,
            working_tree="present",
            kind="file",
            content_sha256=digest,
            present=True,
        )
        revisions.append(revision)
        manifest.append({"path": path, "sha256": digest, "bytes": read.total_bytes})

    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_manifest_bytes:
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.TRUNCATED,
            total_paths=len(revisions),
            detail_code="manifest_byte_limit_exceeded",
        )
    return WorkspaceRevisionObservation(
        identity=identity,
        status=WorkspaceRevisionObservationStatus.SUPPORTED,
        revision="sha256:" + hashlib.sha256(encoded).hexdigest(),
        paths=tuple(revisions),
        total_paths=len(revisions),
    )


def compare_workspace_revisions(
    before: WorkspaceRevisionObservation,
    after: WorkspaceRevisionObservation,
) -> WorkspaceRevisionDelta:
    """Compare two complete observations without interpreting revisions as snapshots."""

    if (
        type(before) is not WorkspaceRevisionObservation
        or type(after) is not WorkspaceRevisionObservation
    ):
        raise TypeError("Workspace revision comparison requires observation instances.")
    if before.identity != after.identity:
        raise ValueError("Workspace revision comparison requires the same workspace identity.")
    non_success = _comparison_failure_status(before.status, after.status)
    if non_success is not None:
        return WorkspaceRevisionDelta(
            identity=before.identity,
            status=non_success,
            before_revision=before.revision,
            after_revision=after.revision,
            detail_code="observation_not_complete",
        )

    before_by_path = _workspace_paths_by_path(before.paths)
    after_by_path = _workspace_paths_by_path(after.paths)
    if before.path_scope != after.path_scope:
        return WorkspaceRevisionDelta(
            identity=before.identity,
            status=WorkspaceRevisionDeltaStatus.FAILED,
            before_revision=before.revision,
            after_revision=after.revision,
            detail_code="observation_path_scope_mismatch",
        )
    sparse_paths = before.path_scope == "changed"
    after_rename_sources = {
        entry.renamed_from for entry in after.paths if entry.renamed_from is not None
    }
    inferred_renames = _infer_exact_git_renames(before_by_path, after_by_path)
    inferred_rename_sources = set(inferred_renames.values())
    head_changed = before.head_revision != after.head_revision
    branch_changed = before.branch != after.branch
    changes: list[WorkspacePathRevisionDelta] = []
    for path in sorted(before_by_path.keys() | after_by_path.keys()):
        previous = before_by_path.get(path)
        current = after_by_path.get(path)
        if previous is None:
            if current is None:  # pragma: no cover - set union makes this impossible
                raise AssertionError("Workspace path disappeared during comparison.")
            changes.append(
                WorkspacePathRevisionDelta(
                    path=path,
                    change=("renamed" if path in inferred_renames else _new_path_change(current)),
                    renamed_from=current.renamed_from or inferred_renames.get(path),
                )
            )
        elif current is None:
            if path in after_rename_sources or path in inferred_rename_sources:
                continue
            changes.append(
                WorkspacePathRevisionDelta(
                    path=path,
                    change=(
                        "deleted"
                        if not sparse_paths
                        or (
                            previous.present is True
                            and previous.tracked is False
                            and not head_changed
                        )
                        else "modified"
                    ),
                )
            )
        elif previous != current:
            changes.append(
                WorkspacePathRevisionDelta(
                    path=path,
                    change=(
                        "deleted"
                        if current.present is False
                        else (
                            "renamed"
                            if current.renamed_from is not None
                            and current.renamed_from != previous.renamed_from
                            else "modified"
                        )
                    ),
                    renamed_from=current.renamed_from,
                )
            )
    changed = before.revision != after.revision or bool(changes) or head_changed or branch_changed
    return WorkspaceRevisionDelta(
        identity=before.identity,
        status=(
            WorkspaceRevisionDeltaStatus.CHANGED
            if changed
            else WorkspaceRevisionDeltaStatus.NO_CHANGE
        ),
        before_revision=before.revision,
        after_revision=after.revision,
        paths=tuple(changes),
        total_paths=len(changes),
        head_changed=head_changed,
        branch_changed=branch_changed,
    )


def _infer_exact_git_renames(
    before_by_path: dict[str, WorkspacePathRevision],
    after_by_path: dict[str, WorkspacePathRevision],
) -> dict[str, str]:
    """Infer unambiguous committed renames from stable Git index object ids."""

    explicit_sources = {
        entry.renamed_from for entry in after_by_path.values() if entry.renamed_from is not None
    }
    deleted_by_object: dict[tuple[str, str], list[str]] = {}
    added_by_object: dict[tuple[str, str], list[str]] = {}
    for path, entry in before_by_path.items():
        if path in after_by_path or path in explicit_sources or entry.index_object_id is None:
            continue
        deleted_by_object.setdefault((entry.kind, entry.index_object_id), []).append(path)
    for path, entry in after_by_path.items():
        if (
            path in before_by_path
            or entry.index_object_id is None
            or entry.renamed_from is not None
        ):
            continue
        added_by_object.setdefault((entry.kind, entry.index_object_id), []).append(path)
    return {
        added_paths[0]: deleted_by_object[key][0]
        for key, added_paths in added_by_object.items()
        if len(added_paths) == 1 and len(deleted_by_object.get(key, ())) == 1
    }


def _new_path_change(
    current: WorkspacePathRevision,
) -> Literal["added", "modified", "deleted", "renamed"]:
    if current.renamed_from is not None:
        return "renamed"
    if current.present is False:
        return "deleted"
    if current.untracked or current.staged == "A" or current.working_tree == "present":
        return "added"
    if current.staged == "D" or current.working_tree == "D":
        return "deleted"
    return "modified"


def _comparison_failure_status(
    before: WorkspaceRevisionObservationStatus,
    after: WorkspaceRevisionObservationStatus,
) -> WorkspaceRevisionDeltaStatus | None:
    statuses = {before, after}
    for observation, delta in (
        (WorkspaceRevisionObservationStatus.FAILED, WorkspaceRevisionDeltaStatus.FAILED),
        (WorkspaceRevisionObservationStatus.TRUNCATED, WorkspaceRevisionDeltaStatus.TRUNCATED),
        (WorkspaceRevisionObservationStatus.INCOMPLETE, WorkspaceRevisionDeltaStatus.INCOMPLETE),
        (WorkspaceRevisionObservationStatus.UNSUPPORTED, WorkspaceRevisionDeltaStatus.UNSUPPORTED),
    ):
        if observation in statuses:
            return delta
    return None


def _safe_relative_path(path: str) -> bool:
    if type(path) is not str or not path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and str(candidate) == path


def _workspace_paths_by_path(
    paths: tuple[WorkspacePathRevision, ...],
) -> dict[str, WorkspacePathRevision]:
    """Index validated observation paths without silently discarding duplicates."""

    indexed: dict[str, WorkspacePathRevision] = {}
    for entry in paths:
        if entry.path in indexed:
            raise ValueError("Workspace observation paths must be unique.")
        indexed[entry.path] = entry
    return indexed

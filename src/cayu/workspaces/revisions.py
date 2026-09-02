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

_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS = 256


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


class WorkspaceMutationAttributionConfidence(StrEnum):
    """How confidently an observed workspace change can be assigned."""

    EXCLUSIVE_TOOL = "exclusive_tool"
    CONCURRENT_AMBIGUITY = "concurrent_ambiguity"
    EXTERNAL_OR_UNKNOWN = "external_or_unknown"
    UNATTRIBUTED_FINALIZATION_CHANGE = "unattributed_finalization_change"


class WorkspaceWriterIsolationStatus(StrEnum):
    """Adapter evidence about writers outside one Cayu mutation window."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    UNKNOWN = "unknown"


class WorkspaceForkLineageStatus(StrEnum):
    """Relationship between source and child workspaces at session fork."""

    DERIVED = "derived"
    SHARED_OR_AMBIGUOUS = "shared_or_ambiguous"
    UNPROVEN = "unproven"


class WorkspaceForkLineage(BaseModel):
    """Bounded workspace lineage evidence for a transcript/session fork."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: WorkspaceForkLineageStatus
    source_workspace_revision: str | None = None
    detail_code: str = Field(max_length=_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS)

    @field_validator("source_workspace_revision")
    @classmethod
    def validate_optional_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "source_workspace_revision")

    @field_validator("detail_code")
    @classmethod
    def validate_lineage_detail(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "detail_code")

    @model_validator(mode="after")
    def validate_lineage_shape(self) -> WorkspaceForkLineage:
        derived = self.status is WorkspaceForkLineageStatus.DERIVED
        if derived != (self.source_workspace_revision is not None):
            raise ValueError(
                "Only demonstrably derived workspace lineage may carry a source revision."
            )
        return self


class WorkspaceWriterIsolationEvidence(BaseModel):
    """Bounded adapter evidence for workspace-writer isolation.

    ``EXCLUSIVE`` is an adapter contract, not an inference from serialized tool
    execution.  The same non-secret generation must be observed at both ends
    of a mutation window before the runtime may use it for exact attribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: WorkspaceWriterIsolationStatus = WorkspaceWriterIsolationStatus.UNKNOWN
    mechanism: str | None = Field(
        default=None,
        max_length=_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS,
    )
    generation: str | None = Field(
        default=None,
        max_length=_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS,
    )
    detail_code: str | None = Field(
        default="writer_isolation_unavailable",
        max_length=_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS,
    )

    @field_validator("mechanism", "generation", "detail_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> WorkspaceWriterIsolationEvidence:
        if self.status is WorkspaceWriterIsolationStatus.EXCLUSIVE:
            if self.mechanism is None or self.generation is None:
                raise ValueError(
                    "Exclusive workspace-writer isolation requires a mechanism and generation."
                )
            if self.detail_code is not None:
                raise ValueError(
                    "Exclusive workspace-writer isolation cannot carry a failure detail."
                )
        elif self.generation is not None:
            raise ValueError("Non-exclusive workspace-writer isolation cannot carry a generation.")
        return self


class WorkspaceDirectMutationReconciliation(StrEnum):
    """Agreement between direct workspace operations and observed revisions."""

    NOT_OBSERVED = "not_observed"
    CONSISTENT = "consistent"
    INCOMPLETE = "incomplete"
    CONTRADICTORY = "contradictory"
    TRUNCATED = "truncated"


class WorkspaceMutationAttribution(BaseModel):
    """Conservative attribution attached to an observed workspace delta."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    confidence: WorkspaceMutationAttributionConfidence
    writer_isolation: WorkspaceWriterIsolationStatus
    overlap_detected: StrictBool = False
    direct_reconciliation: WorkspaceDirectMutationReconciliation = (
        WorkspaceDirectMutationReconciliation.NOT_OBSERVED
    )
    detail_code: str = Field(max_length=_WORKSPACE_ATTRIBUTION_TEXT_MAX_CHARS)

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "detail_code")

    @model_validator(mode="after")
    def validate_attribution_shape(self) -> WorkspaceMutationAttribution:
        if (
            self.confidence is WorkspaceMutationAttributionConfidence.EXCLUSIVE_TOOL
            and self.writer_isolation is not WorkspaceWriterIsolationStatus.EXCLUSIVE
        ):
            raise ValueError("Exclusive tool attribution requires exclusive writer isolation.")
        if (
            self.confidence is WorkspaceMutationAttributionConfidence.EXCLUSIVE_TOOL
            and self.overlap_detected
        ):
            raise ValueError("An overlapping mutation window cannot have exclusive attribution.")
        if (
            self.confidence is WorkspaceMutationAttributionConfidence.EXCLUSIVE_TOOL
            and self.direct_reconciliation is WorkspaceDirectMutationReconciliation.CONTRADICTORY
        ):
            raise ValueError("Contradictory direct evidence cannot have exclusive attribution.")
        if (
            self.confidence
            is WorkspaceMutationAttributionConfidence.UNATTRIBUTED_FINALIZATION_CHANGE
            and self.writer_isolation is WorkspaceWriterIsolationStatus.EXCLUSIVE
        ):
            raise ValueError("Finalization-only change cannot claim tool-writer isolation.")
        return self


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
    worktree_mode: str | None = None
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
        "worktree_mode",
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
        if info.field_name in {"index_mode", "worktree_mode"} and (
            len(text) != 6 or any(char not in "01234567" for char in text)
        ):
            raise ValueError(
                f"Workspace revision {info.field_name.replace('_', ' ')} must be a "
                "six-digit octal mode."
            )
        return text


_WORKSPACE_PATH_REVISION_FIELDS = frozenset(WorkspacePathRevision.model_fields)
_WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS = frozenset(
    {
        "content_sha256",
        "index_mode",
        "worktree_mode",
        "index_object_id",
        "kind",
        "staged",
        "working_tree",
    }
)


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


class WorkspaceRevisionObservationLimitExceeded(ValueError):
    """A detached observation exceeds the runtime's bounded evidence contract."""


def copy_bounded_workspace_revision_observation(
    observed: object,
    *,
    expected_identity: WorkspaceIdentity,
    limits: WorkspaceRevisionObservationLimits,
    max_total_paths: int = (1 << 63) - 1,
) -> WorkspaceRevisionObservation:
    """Detach extension-owned revision evidence without invoking its serializers."""

    if type(observed) is not WorkspaceRevisionObservation:
        raise TypeError("Workspace observer must return WorkspaceRevisionObservation.")
    if type(expected_identity) is not WorkspaceIdentity:
        raise TypeError("expected_identity must be a WorkspaceIdentity.")
    if type(limits) is not WorkspaceRevisionObservationLimits:
        raise TypeError("limits must be WorkspaceRevisionObservationLimits.")
    if type(max_total_paths) is not int or max_total_paths < 0:
        raise ValueError("max_total_paths must be a non-negative integer.")
    if type(observed.identity) is not WorkspaceIdentity:
        raise TypeError("Workspace observer returned an invalid identity.")
    if (
        type(observed.identity.workspace_id) is not str
        or type(observed.identity.observer) is not str
        or observed.identity.workspace_id != expected_identity.workspace_id
        or observed.identity.observer != expected_identity.observer
    ):
        raise ValueError("Workspace observer returned an unexpected identity.")
    if type(observed.status) is not WorkspaceRevisionObservationStatus:
        raise TypeError("Workspace observation status is invalid.")
    if type(observed.path_scope) is not str or observed.path_scope not in {
        "complete",
        "changed",
    }:
        raise TypeError("Workspace observation path scope is invalid.")
    if type(observed.paths) is not tuple:
        raise TypeError("Workspace observation paths must be a tuple.")
    if len(observed.paths) > limits.max_paths:
        raise WorkspaceRevisionObservationLimitExceeded
    if (
        type(observed.total_paths) is not int
        or observed.total_paths < 0
        or observed.total_paths > max_total_paths
    ):
        raise WorkspaceRevisionObservationLimitExceeded

    serialized_size = 256
    for value in (
        observed.identity.workspace_id,
        observed.identity.observer,
        observed.revision,
        observed.head_revision,
        observed.branch,
        observed.detail_code,
    ):
        serialized_size += _bounded_observation_text_size(
            value,
            max_bytes=limits.max_path_bytes,
        )
    detached_paths: list[WorkspacePathRevision] = []
    detached_path_names: set[str] = set()
    for path in observed.paths:
        if type(path) is not WorkspacePathRevision:
            raise TypeError("Workspace observation contains an invalid path entry.")
        if type(path.path) is not str or path.path in detached_path_names:
            raise ValueError("Workspace observation paths must be unique strings.")
        detached_path_names.add(path.path)
        for value in (path.untracked, path.ignored):
            if type(value) is not bool:
                raise TypeError("Workspace observation path flags must be booleans.")
        for value in (path.present, path.tracked):
            if value is not None and type(value) is not bool:
                raise TypeError("Workspace observation path flags must be booleans or None.")
        if type(path.kind) is not str or path.kind not in {
            "file",
            "symlink",
            "submodule",
            "unknown",
        }:
            raise TypeError("Workspace observation path kind is invalid.")
        for value in (
            path.path,
            path.staged,
            path.working_tree,
            path.content_sha256,
            path.index_object_id,
            path.index_mode,
            path.worktree_mode,
            path.renamed_from,
        ):
            serialized_size += _bounded_observation_text_size(
                value,
                max_bytes=limits.max_path_bytes,
            )
        serialized_size += 192
        if serialized_size > limits.max_manifest_bytes:
            raise WorkspaceRevisionObservationLimitExceeded
        detached_paths.append(
            WorkspacePathRevision(
                path=path.path,
                staged=path.staged,
                working_tree=path.working_tree,
                untracked=path.untracked,
                ignored=path.ignored,
                present=path.present,
                tracked=path.tracked,
                kind=path.kind,
                content_sha256=path.content_sha256,
                index_object_id=path.index_object_id,
                index_mode=path.index_mode,
                worktree_mode=path.worktree_mode,
                renamed_from=path.renamed_from,
            )
        )

    detached = WorkspaceRevisionObservation(
        identity=WorkspaceIdentity(
            workspace_id=observed.identity.workspace_id,
            observer=observed.identity.observer,
        ),
        status=observed.status,
        revision=observed.revision,
        head_revision=observed.head_revision,
        branch=observed.branch,
        path_scope=observed.path_scope,
        paths=tuple(detached_paths),
        total_paths=observed.total_paths,
        detail_code=observed.detail_code,
    )
    encoded = detached.model_dump_json().encode("utf-8")
    if len(encoded) > limits.max_manifest_bytes:
        raise WorkspaceRevisionObservationLimitExceeded
    return WorkspaceRevisionObservation.model_validate_json(encoded)


def _bounded_observation_text_size(value: str | None, *, max_bytes: int) -> int:
    if value is None:
        return 0
    if type(value) is not str:
        raise TypeError("Workspace observation text fields must be strings.")
    if len(value) > max_bytes:
        raise WorkspaceRevisionObservationLimitExceeded
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise WorkspaceRevisionObservationLimitExceeded
    return len(encoded)


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


_WORKSPACE_PATH_REVISION_DELTA_FIELDS = frozenset(WorkspacePathRevisionDelta.model_fields)
_WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS = frozenset({"change"})


class WorkspaceRevisionDelta(BaseModel):
    """Bounded observed change between two revisions; it does not imply causation."""

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
                git_mode=raw_read.git_mode,
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
            worktree_mode=read.git_mode,
            present=True,
        )
        revisions.append(revision)
        manifest.append(
            {
                "path": path,
                "sha256": digest,
                "bytes": read.total_bytes,
                "git_mode": read.git_mode,
            }
        )

    encoded = _deterministic_workspace_manifest_bytes(manifest)
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
        revision=_deterministic_workspace_manifest_revision(encoded),
        paths=tuple(revisions),
        total_paths=len(revisions),
    )


def _deterministic_workspace_manifest_bytes(
    manifest: list[dict[str, object]],
) -> bytes:
    """Encode the one canonical non-Git workspace revision manifest."""

    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _deterministic_workspace_manifest_revision(encoded: bytes) -> str:
    """Return the #909 revision identity for canonical manifest bytes."""

    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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

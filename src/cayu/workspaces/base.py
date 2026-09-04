from __future__ import annotations

import posixpath
import re
from abc import ABC, abstractmethod
from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, BinaryIO, Literal

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runners.base import Runner

if TYPE_CHECKING:
    from cayu.workspaces.branches import (
        WorkspaceBranchCapabilities,
        WorkspaceBranchCreationResult,
        WorkspaceBranchLifecycleSummary,
        WorkspaceBranchRequest,
    )

WorkspaceGitMode = Literal["100644", "100755"]
WorkspaceGitEntryMode = Literal["100644", "100755", "120000"]


@dataclass(frozen=True)
class WorkspaceReadResult:
    content: bytes
    total_bytes: int
    truncated: bool = False
    offset: int = 0
    revision: str | None = None
    sha256: str | None = None
    git_mode: WorkspaceGitMode | None = None
    source_bytes_read: int | None = None
    redaction_truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("WorkspaceReadResult content must be bytes.")
        if type(self.total_bytes) is not int:
            raise TypeError("WorkspaceReadResult total_bytes must be an integer.")
        if self.total_bytes < 0:
            raise ValueError("WorkspaceReadResult total_bytes must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceReadResult truncated must be a bool.")
        if type(self.offset) is not int:
            raise TypeError("WorkspaceReadResult offset must be an integer.")
        if self.offset < 0:
            raise ValueError("WorkspaceReadResult offset must be non-negative.")
        if self.offset > self.total_bytes:
            raise ValueError("WorkspaceReadResult offset cannot exceed total_bytes.")
        source_bytes_read = (
            len(self.content) if self.source_bytes_read is None else self.source_bytes_read
        )
        if type(source_bytes_read) is not int:
            raise TypeError("WorkspaceReadResult source_bytes_read must be an integer or None.")
        if source_bytes_read < 0:
            raise ValueError("WorkspaceReadResult source_bytes_read must be non-negative.")
        if source_bytes_read < len(self.content):
            raise ValueError("WorkspaceReadResult source progress cannot be smaller than content.")
        if self.total_bytes < self.offset + source_bytes_read:
            raise ValueError(
                "WorkspaceReadResult total_bytes cannot be smaller than content "
                "or source-page progress."
            )
        expected_truncated = self.offset + source_bytes_read < self.total_bytes
        if self.truncated != expected_truncated:
            raise ValueError(
                "WorkspaceReadResult truncated must match source progress and total_bytes."
            )
        if self.truncated and source_bytes_read == 0:
            raise ValueError("WorkspaceReadResult truncated pages must make forward progress.")
        if type(self.redaction_truncated) is not bool:
            raise TypeError("WorkspaceReadResult redaction_truncated must be a bool.")
        if self.revision is not None and (
            type(self.revision) is not str or not self.revision.strip()
        ):
            raise ValueError("WorkspaceReadResult revision must be a nonblank string or None.")
        if self.sha256 is not None and (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError(
                "WorkspaceReadResult sha256 must be a lowercase SHA-256 digest or None."
            )
        if self.git_mode is not None and self.git_mode not in {"100644", "100755"}:
            raise ValueError("WorkspaceReadResult git_mode must be 100644, 100755, or None.")
        if (self.revision is not None or self.sha256 is not None or self.git_mode is not None) and (
            self.offset != 0 or self.truncated
        ):
            raise ValueError(
                "WorkspaceReadResult identity metadata requires a complete offset-zero snapshot."
            )
        object.__setattr__(self, "source_bytes_read", source_bytes_read)

    @property
    def next_offset(self) -> int | None:
        return (
            self.offset + self.source_bytes_read
            if self.truncated and self.source_bytes_read
            else None
        )


@dataclass(frozen=True)
class WorkspaceGitEntry:
    """One Git-significant workspace entry observed without following symlinks."""

    path: str
    git_mode: WorkspaceGitEntryMode
    symlink_target_sha256: str | None = None
    symlink_target_bytes: int | None = None

    def __post_init__(self) -> None:
        path = _validate_workspace_relative_path(self.path)
        if self.git_mode not in {"100644", "100755", "120000"}:
            raise ValueError("WorkspaceGitEntry git_mode is invalid.")
        if self.git_mode == "120000":
            if (
                type(self.symlink_target_sha256) is not str
                or len(self.symlink_target_sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in self.symlink_target_sha256
                )
            ):
                raise ValueError(
                    "WorkspaceGitEntry symlink_target_sha256 must be a lowercase SHA-256 digest."
                )
            if type(self.symlink_target_bytes) is not int or self.symlink_target_bytes < 0:
                raise ValueError(
                    "WorkspaceGitEntry symlink_target_bytes must be non-negative for symlinks."
                )
        elif self.symlink_target_sha256 is not None or self.symlink_target_bytes is not None:
            raise ValueError("Regular WorkspaceGitEntry values cannot define a symlink target.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class WorkspaceGitEntryListResult:
    """Bounded complete-or-truncated Git-significant entry observation."""

    entries: tuple[WorkspaceGitEntry, ...]
    total_count: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.entries, str | bytes):
            raise TypeError("WorkspaceGitEntryListResult entries must be an iterable.")
        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise TypeError("WorkspaceGitEntryListResult entries must be an iterable.") from exc
        if any(type(entry) is not WorkspaceGitEntry for entry in entries):
            raise TypeError(
                "WorkspaceGitEntryListResult entries must be exact WorkspaceGitEntry values."
            )
        if tuple(sorted(entries, key=lambda entry: entry.path)) != entries:
            raise ValueError("WorkspaceGitEntryListResult entries must be sorted by path.")
        paths = tuple(entry.path for entry in entries)
        if len(paths) != len(set(paths)):
            raise ValueError("WorkspaceGitEntryListResult entries must have unique paths.")
        if type(self.total_count) is not int or self.total_count < 0:
            raise ValueError("WorkspaceGitEntryListResult total_count must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceGitEntryListResult truncated must be a bool.")
        if self.total_count < len(entries):
            raise ValueError(
                "WorkspaceGitEntryListResult total_count cannot be smaller than entries."
            )
        if not self.truncated and self.total_count != len(entries):
            raise ValueError(
                "WorkspaceGitEntryListResult total_count must equal entries when complete."
            )
        object.__setattr__(self, "entries", entries)


class WorkspaceGitEntryObservationUnsupportedError(RuntimeError):
    """A workspace cannot enumerate every Git-significant entry without following links."""


class WorkspaceReadOffsetError(ValueError):
    """A workspace read requested a byte offset beyond the current file size."""

    def __init__(self, offset: int, total_bytes: int) -> None:
        self.offset = offset
        self.total_bytes = total_bytes
        super().__init__(f"Workspace read offset {offset} cannot exceed file size {total_bytes}.")


WorkspaceMutationOperation = Literal["create", "replace", "delete"]
WorkspaceMoveFidelity = Literal["atomic_rename", "link_unlink"]


@dataclass(frozen=True)
class WorkspaceMutationResult:
    operation: WorkspaceMutationOperation
    before_revision: str | None
    after_revision: str | None
    before_sha256: str | None
    after_sha256: str | None
    before_bytes: int | None
    after_bytes: int | None

    def __post_init__(self) -> None:
        if self.operation not in {"create", "replace", "delete"}:
            raise ValueError("WorkspaceMutationResult operation is invalid.")
        for field_name in ("before_revision", "after_revision"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"WorkspaceMutationResult {field_name} must be nonblank or None.")
        for field_name in ("before_sha256", "after_sha256"):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"WorkspaceMutationResult {field_name} must be a lowercase SHA-256 digest or None."
                )
        for field_name in ("before_bytes", "after_bytes"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"WorkspaceMutationResult {field_name} must be non-negative or None."
                )
        if self.operation == "create" and any(
            value is not None
            for value in (self.before_revision, self.before_sha256, self.before_bytes)
        ):
            raise ValueError("WorkspaceMutationResult create cannot define before metadata.")
        if self.operation == "delete" and any(
            value is not None
            for value in (self.after_revision, self.after_sha256, self.after_bytes)
        ):
            raise ValueError("WorkspaceMutationResult delete cannot define after metadata.")
        before = (self.before_revision, self.before_sha256, self.before_bytes)
        after = (self.after_revision, self.after_sha256, self.after_bytes)
        if self.operation in {"replace", "delete"} and any(value is None for value in before):
            raise ValueError(
                f"WorkspaceMutationResult {self.operation} requires complete before metadata."
            )
        if self.operation in {"create", "replace"} and any(value is None for value in after):
            raise ValueError(
                f"WorkspaceMutationResult {self.operation} requires complete after metadata."
            )


@dataclass(frozen=True)
class WorkspaceMoveResult:
    """Evidence for one conditional, no-overwrite workspace file move.

    ``link_unlink`` is intentionally distinct from ``atomic_rename``: the
    destination link is created conditionally before the source name is
    removed, so a backend using that portable implementation must not claim a
    single atomic rename.
    """

    source_before_revision: str
    source_after_revision: None
    destination_before_revision: None
    destination_after_revision: str
    source_before_sha256: str
    source_after_sha256: None
    destination_before_sha256: None
    destination_after_sha256: str
    source_before_bytes: int
    source_after_bytes: None
    destination_before_bytes: None
    destination_after_bytes: int
    fidelity: WorkspaceMoveFidelity

    def __post_init__(self) -> None:
        for field_name in ("source_before_revision", "destination_after_revision"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"WorkspaceMoveResult {field_name} must be nonblank.")
        for field_name in ("source_before_sha256", "destination_after_sha256"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"WorkspaceMoveResult {field_name} must be a lowercase SHA-256 digest."
                )
        for field_name in ("source_before_bytes", "destination_after_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"WorkspaceMoveResult {field_name} must be non-negative.")
        if self.fidelity not in {"atomic_rename", "link_unlink"}:
            raise ValueError("WorkspaceMoveResult fidelity is invalid.")
        if (
            self.source_before_revision != self.destination_after_revision
            or self.source_before_sha256 != self.destination_after_sha256
            or self.source_before_bytes != self.destination_after_bytes
        ):
            raise ValueError("WorkspaceMoveResult must preserve the moved content identity.")


class WorkspacePreconditionUnsupportedError(RuntimeError):
    """A workspace cannot authoritatively evaluate a required precondition."""


class WorkspaceMoveUnsupportedError(RuntimeError):
    """A workspace cannot provide a conditional same-workspace regular-file move."""


class WorkspaceMoveAmbiguousError(RuntimeError):
    """A move may have created its destination before settlement became uncertain."""

    def __init__(self, result: WorkspaceMoveResult | None = None) -> None:
        self.result = result
        super().__init__("Workspace move settlement is ambiguous; observe both paths freshly.")


class WorkspaceRevisionMismatchError(RuntimeError):
    """A conditional workspace mutation observed a different current revision."""

    def __init__(self, expected_revision: str, actual_revision: str) -> None:
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            "Workspace file revision changed: "
            f"expected {expected_revision}, found {actual_revision}."
        )


class WorkspaceGitModeMismatchError(RuntimeError):
    """A conditional workspace mutation observed a different Git file mode."""

    def __init__(
        self, expected_git_mode: WorkspaceGitMode, actual_git_mode: WorkspaceGitMode
    ) -> None:
        self.expected_git_mode = expected_git_mode
        self.actual_git_mode = actual_git_mode
        super().__init__(
            "Workspace file Git mode changed: "
            f"expected {expected_git_mode}, found {actual_git_mode}."
        )


@dataclass(frozen=True)
class WorkspaceListResult:
    paths: tuple[str, ...]
    total_count: int | None
    truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.paths, str | bytes):
            raise TypeError("WorkspaceListResult paths must be an iterable of strings.")
        try:
            paths = tuple(self.paths)
        except TypeError as exc:
            raise TypeError("WorkspaceListResult paths must be an iterable of strings.") from exc
        for path in paths:
            if type(path) is not str:
                raise TypeError("WorkspaceListResult paths entries must be strings.")
        if self.total_count is not None:
            if type(self.total_count) is not int:
                raise TypeError("WorkspaceListResult total_count must be an integer.")
            if self.total_count < 0:
                raise ValueError("WorkspaceListResult total_count must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceListResult truncated must be a bool.")
        if not self.truncated and self.total_count is None:
            raise ValueError("WorkspaceListResult total_count is required when not truncated.")
        if not self.truncated and self.total_count is not None and self.total_count != len(paths):
            raise ValueError("WorkspaceListResult total_count must equal paths when not truncated.")
        if self.total_count is not None and self.total_count < len(paths):
            raise ValueError("WorkspaceListResult total_count cannot be smaller than paths.")
        object.__setattr__(self, "paths", paths)


class _WorkspaceListCollector:
    """Collect a deterministic sorted prefix without retaining every matched path."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._paths: list[str] = []
        self._total_count = 0

    def add(self, path: str) -> None:
        self._total_count += 1
        if self._limit is None:
            self._paths.append(path)
            return
        insort(self._paths, path)
        if len(self._paths) > self._limit:
            self._paths.pop()

    def result(self, *, exact_total_when_truncated: bool = True) -> WorkspaceListResult:
        paths = tuple(sorted(self._paths))
        truncated = self._total_count > len(paths)
        total_count = self._total_count if exact_total_when_truncated or not truncated else None
        return WorkspaceListResult(
            paths=paths,
            total_count=total_count,
            truncated=truncated,
        )


def validate_list_pattern(pattern: str) -> str:
    """Validate a Workspace ``list()`` pattern shared by every backend."""
    value = require_nonblank(pattern, "pattern")
    if posixpath.isabs(value):
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    parts = tuple(part for part in value.split("/") if part)
    if ".." in parts:
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    return value


def translate_list_pattern(pattern: str) -> str:
    """Translate a Workspace ``list()`` pattern into an anchored regular expression.

    This defines the one normative matching semantics shared by every
    Workspace backend, applied to a file's full workspace-relative POSIX path:

    - The pattern is anchored at both ends (``*.txt`` does NOT match
      ``nested/a.txt``).
    - ``*`` matches any run of characters within one path segment, ``?``
      matches one character within a segment, and ``[...]``/``[!...]``
      character classes match one character within a segment; none of them
      cross ``/``.
    - ``**`` as a whole segment matches zero or more directories when more
      pattern follows (``**/*.txt`` matches ``a.txt`` and ``d/a.txt``), and
      matches any remaining path when it is the final segment.
    - Empty segments and ``.`` segments in the pattern are ignored.
    """
    segments = [segment for segment in pattern.split("/") if segment not in {"", "."}]
    if not segments:
        return r"(?!)"
    parts: list[str] = []
    last_index = len(segments) - 1
    for index, segment in enumerate(segments):
        is_last = index == last_index
        if segment == "**":
            if is_last:
                parts.append(r"[^/]+(?:/[^/]+)*")
            else:
                parts.append(r"(?:[^/]+/)*")
            continue
        parts.append(_segment_regex(segment) + ("" if is_last else "/"))
    return "".join(parts)


def matches_list_pattern(path: str, pattern: str) -> bool:
    """Report whether a workspace-relative POSIX file path matches a list pattern."""
    return _compiled_list_pattern(pattern).fullmatch(path) is not None


@lru_cache(maxsize=256)
def _compiled_list_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(translate_list_pattern(pattern))


def _segment_regex(segment: str) -> str:
    parts: list[str] = []
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if char == "*":
            parts.append(r"[^/]*")
        elif char == "?":
            parts.append(r"[^/]")
        elif char == "[":
            closing = index + 1
            if closing < length and segment[closing] in "!^":
                closing += 1
            if closing < length and segment[closing] == "]":
                closing += 1
            while closing < length and segment[closing] != "]":
                closing += 1
            if closing >= length:
                parts.append(r"\[")
            else:
                inner = segment[index + 1 : closing]
                inner = (
                    "^" + _literal_regex_class(inner[1:])
                    if inner.startswith("!")
                    else _literal_regex_class(inner)
                )
                parts.append(f"[{inner}]")
                index = closing
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


def _literal_regex_class(value: str) -> str:
    """Escape glob class content where every character is a literal option."""
    escaped = value.replace("\\", "\\\\")
    if escaped.startswith("^"):
        escaped = "\\" + escaped
    return escaped


class Workspace(ABC):
    """Filesystem/artifact area an agent can work in."""

    id: str

    @abstractmethod
    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        """Read a byte page from the workspace.

        Complete offset-zero snapshots expose an opaque revision suitable for a
        later conditional mutation. Backends must not expose a revision for a
        partial page. An offset equal to the current file size returns an empty
        page; an offset beyond it raises ``WorkspaceReadOffsetError``.
        """

    @abstractmethod
    def bounded_read_limit(self, max_bytes: int) -> int:
        """Resolve a hard read ceiling without loosening backend defaults.

        Return a positive integer no greater than ``max_bytes``. Backends with
        their own finite default read limit must return the smaller value;
        backends without one return ``max_bytes``. Callers use this before
        ``read_bytes`` when a safety bound must compose with backend policy.
        """

    @abstractmethod
    async def write_bytes(self, path: str, content: bytes) -> None:
        """Write a file into the workspace."""

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete a file from the workspace if it exists."""

    @abstractmethod
    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        """Create a missing file without overwriting an existing path."""

    @abstractmethod
    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        """Replace a file only when its current opaque revision matches."""

    @abstractmethod
    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        """Delete a file only when its current opaque revision matches."""

    async def require_absent(self, path: str) -> None:
        """Authoritatively preflight that one workspace path is absent.

        This is an optional compatibility capability. Implementations used by
        the structured patch tool must override it; the later create or move
        still retains its own conditional absence precondition because this
        preflight is not a lock.
        """

        del path
        raise WorkspacePreconditionUnsupportedError(
            f"{type(self).__name__} does not support authoritative absence preconditions."
        )

    async def move_if_revision(
        self,
        source_path: str,
        destination_path: str,
        *,
        expected_source_revision: str,
        require_destination_absent: bool = True,
    ) -> WorkspaceMoveResult:
        """Conditionally move one regular file within this workspace.

        Implementations must never overwrite the destination and must report
        their actual move fidelity. Unsupported adapters fail explicitly; they
        never fall back to shell execution or an unlabelled copy/delete pair.
        """

        del source_path, destination_path, expected_source_revision
        if type(require_destination_absent) is not bool:
            raise TypeError("Workspace require_destination_absent must be a bool.")
        if not require_destination_absent:
            raise ValueError("Workspace moves must require an absent destination.")
        raise WorkspaceMoveUnsupportedError(
            f"{type(self).__name__} does not support conditional file moves."
        )

    @abstractmethod
    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        """List files in the workspace.

        Every backend must match ``pattern`` against workspace-relative POSIX
        file paths with the normative semantics of ``matches_list_pattern``.
        """

    async def list_git_entries(self, *, limit: int) -> WorkspaceGitEntryListResult:
        """List regular files and symlinks without following either entry type."""

        del limit
        raise WorkspaceGitEntryObservationUnsupportedError(
            f"{type(self).__name__} does not support Git-significant entry observation."
        )

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        """Stable, hashable token identifying the underlying resource (filesystem/sandbox area)
        this workspace reads and writes, so callers can tell whether two ``Workspace`` objects
        point at the SAME place.

        Returns ``None`` when identity cannot be determined. ``SyncBinding`` then refuses to bind
        rather than risk clearing a target that is actually the source; override this in a custom
        ``Workspace`` to return a stable identity token and enable that safety check.
        """
        return None

    def tar_copy_policy_identity(self) -> tuple[object, ...] | None:
        """Return an exact, content-free identity for tar copy security policy.

        ``SyncBinding`` only shares a sealed archive when both adapters opt in
        with stable identities. Implementations must change the identity whenever
        path exclusions, archive parsing, or extraction policy changes. ``None``
        conservatively disables sharing for this adapter.
        """

        return None

    def bounded_tar_stream_reader(self) -> BoundedTarStreamReader | None:
        """Return an explicit bounded streaming-tar reader capability, if any."""

        return None

    def tar_stream_writer(self) -> TarStreamWriter | None:
        """Return an explicit streaming-tar writer capability, if any."""

        return None

    def branch_capabilities(self) -> WorkspaceBranchCapabilities:
        """Return explicit workspace-branch guarantees for this adapter.

        Unsupported is the compatibility default. Capability must be declared
        by the exact adapter instance and is never inferred from adjacent
        filesystem, snapshot, runner, sync, reconnect, or Git behavior.
        """

        from cayu.workspaces.branches import WorkspaceBranchCapabilities

        return WorkspaceBranchCapabilities()

    def branch_lifecycle_summary(self) -> WorkspaceBranchLifecycleSummary:
        """Return bounded lifecycle states for branch handles attached here."""

        from cayu.workspaces.branches import WorkspaceBranchLifecycleSummary

        return WorkspaceBranchLifecycleSummary(attached_count=0, statuses=(), truncated=False)

    async def create_branch(
        self,
        request: WorkspaceBranchRequest,
    ) -> WorkspaceBranchCreationResult:
        """Create an isolated bounded workspace branch when supported.

        Branching is an optional capability rather than an abstract workspace
        requirement. Backends that do not implement the complete isolation and
        publication contract return a typed unsupported result.
        """

        from cayu.workspaces.branches import (
            WorkspaceBranchCreationResult,
            WorkspaceBranchOutcomeStatus,
            _bounded_workspace_branch_evidence,
            _copy_workspace_branch_request_envelope,
        )

        copied = _copy_workspace_branch_request_envelope(request)
        return WorkspaceBranchCreationResult(
            status=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            branch=None,
            evidence=_bounded_workspace_branch_evidence(
                source=copied.source,
                baseline_revision=copied.baseline_revision,
                outcome=WorkspaceBranchOutcomeStatus.UNSUPPORTED,
                max_bytes=copied.limits.max_evidence_bytes,
                detail_code="workspace_branching_unsupported",
                hash_fixed_identity_on_overflow=True,
            ),
        )


class WorkspaceDirectoryPruner(ABC):
    """Remove only directory trees with no files, links, or excluded paths."""

    @abstractmethod
    async def prune_empty_directories(self, path: str, *, max_directories: int) -> None:
        """Remove an obstructing empty tree, bounded by ``max_directories``.

        Missing paths and regular files are unchanged. Reject links, special
        files, excluded descendants, and trees exceeding the bound. Callers
        must hold exclusive writer isolation throughout restoration.
        """


class WorkspaceGitModeMutator(ABC):
    """Nominal capability for exact Git executable-mode mutations."""

    @abstractmethod
    async def write_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        git_mode: WorkspaceGitMode,
    ) -> None:
        """Write exact bytes and normalize the file to one Git regular-file mode."""

    @abstractmethod
    async def create_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        git_mode: WorkspaceGitMode,
    ) -> WorkspaceMutationResult:
        """Create a missing file with an exact Git regular-file mode."""

    @abstractmethod
    async def replace_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
        expected_git_mode: WorkspaceGitMode,
        git_mode: WorkspaceGitMode,
    ) -> WorkspaceMutationResult:
        """Conditionally replace bytes and mode against both prior authorities."""


class RunnerBoundWorkspace(Workspace):
    """Workspace whose operations target one declared runner-owned resource.

    Native bindings use this nominal contract instead of reflecting on an
    incidental ``runner`` attribute. Implementations can prove that they target
    the lifecycle-owning runner without exposing that runner through the
    workspace's public surface.
    """

    @abstractmethod
    def is_bound_to_runner(self, runner: Runner) -> bool:
        """Return whether this workspace targets ``runner`` exactly."""

    @abstractmethod
    def _control_plane_runner(self) -> Runner:
        """Return the private runner used by Cayu-owned workspace bindings."""

    @property
    @abstractmethod
    def runner_cwd(self) -> str:
        """Absolute runner path that this workspace exposes."""

    @property
    @abstractmethod
    def bound_runner_resource_key(self) -> tuple[object, ...] | None:
        """Stable runner resource identity used by workspace operations."""


class BoundedTarReader(ABC):
    """Nominal capability for preflight-bounded bulk tar reads.

    Implementations must validate the requested files' combined logical size
    and the conservative raw archive size before allocating or materializing
    the archive. Merely accepting the limit keywords or rejecting after
    construction does not satisfy this contract.
    """

    @abstractmethod
    async def read_tar_bytes(
        self,
        paths: Sequence[str],
        *,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_archive_bytes: int | None = None,
    ) -> bytes:
        """Return an uncompressed tar after preflighting every configured limit."""


@dataclass(frozen=True, slots=True)
class TarStreamReadResult:
    """Content-free accounting for one completed streamed tar export."""

    archive_bytes: int

    def __post_init__(self) -> None:
        if type(self.archive_bytes) is not int:
            raise TypeError("TarStreamReadResult archive_bytes must be an integer.")
        if self.archive_bytes < 0:
            raise ValueError("TarStreamReadResult archive_bytes must be non-negative.")


class BoundedTarStreamReader(ABC):
    """Nominal capability for bounded tar production into a caller-owned stream.

    The destination is a binary stream, never a pathname. Implementations must
    preflight logical and archive limits before emitting archive bytes and must
    settle delegated reads before returning.
    """

    @abstractmethod
    async def read_tar_stream(
        self,
        paths: Sequence[str],
        destination: BinaryIO,
        *,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_archive_bytes: int | None = None,
    ) -> TarStreamReadResult:
        """Write one uncompressed tar to ``destination`` and report its size."""


class TarWriter(ABC):
    """Nominal capability for writing a caller-validated bulk tar archive."""

    @abstractmethod
    async def write_tar_bytes(self, data: bytes) -> None:
        """Extract caller-validated uncompressed tar data into this workspace."""


class TarStreamWriter(ABC):
    """Nominal capability for extracting a caller-validated tar byte stream."""

    @abstractmethod
    async def write_tar_stream(self, source: BinaryIO, *, archive_bytes: int) -> None:
        """Consume exactly ``archive_bytes`` bytes without receiving a host path."""


def _validate_absolute_guest_root(path: str, *, owner: str) -> str:
    root = require_clean_nonblank(path, "root")
    if not posixpath.isabs(root):
        raise ValueError(f"{owner} root must be an absolute guest path.")
    return posixpath.normpath(root)


def _validate_workspace_relative_path(path: str) -> str:
    value = require_nonblank(path, "path")
    if posixpath.isabs(value):
        raise ValueError("Workspace paths must be relative.")
    parts = tuple(part for part in value.split("/") if part)
    if ".." in parts:
        raise ValueError("Workspace path escapes the workspace root via parent traversal.")
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ValueError("Workspace paths must reference a file.")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("Workspace path escapes the workspace root.")
    return normalized


def _validate_workspace_positive_limit(value: int, field_name: str, *, owner: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{owner} {field_name} must be greater than zero.")
    return value


def _validate_workspace_offset(value: int, *, owner: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} offset must be an integer.")
    if value < 0:
        raise ValueError(f"{owner} offset must be non-negative.")
    return value


def _validate_workspace_revision(value: str, *, owner: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{owner} expected_revision must be a nonblank string.")
    return value


def _local_resource_key(path: object) -> tuple[object, ...]:
    """Canonical identity for a host-filesystem directory, shared by every host-backed workspace view."""
    return ("local", str(path))


def _runner_resource_key(runner: object) -> tuple[object, ...] | None:
    """Stable identity for a runner, or ``None`` when the runner exposes no stable identifier.

    Returning ``None`` for an indeterminate runner lets runner-backed workspaces fail closed rather
    than treating Python object identity as proof that two runners are distinct resources.
    """
    if runner is None:
        return None
    declared_key = getattr(runner, "resource_key", None)
    if declared_key is not None:
        if type(declared_key) is not tuple or not declared_key:
            raise TypeError("Runner resource_key must be a non-empty tuple or None.")
        try:
            hash(declared_key)
        except TypeError as exc:
            raise TypeError("Runner resource_key must be hashable.") from exc
        return declared_key
    for attr in (
        "sandbox_id",
        "microvm_id",
        "name",
        "container_name",
        "sandbox_name",
        "root",
    ):
        value = getattr(runner, attr, None)
        if value is not None:
            return (type(runner), attr, str(value))
    return None


def _runner_workspace_resource_key(runner: object, path: str) -> tuple[object, ...] | None:
    """Compose a runner-backed workspace key, or ``None`` when the runner identity is indeterminate."""
    runner_key = _runner_resource_key(runner)
    if runner_key is None:
        return None
    return ("runner", runner_key, path)

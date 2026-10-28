from __future__ import annotations

import asyncio
import errno
import os
import stat
from bisect import insort
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from hashlib import sha256
from os import PathLike
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.workspaces._local_guard import (
    _SEARCH_BASE_FLAGS,
    _require_descriptor_guard_support,
    create_regular,
    delete_regular,
    delete_regular_if_revision,
    move_regular_if_revision,
    open_regular_for_read,
    prune_empty_directories,
    replace_regular_if_revision,
    require_absent_regular,
    restore_regular,
    write_regular,
)
from cayu.workspaces._mutations import (
    content_identity,
    mutation_result,
    mutation_result_from_identities,
    workspace_path_lock,
    workspace_path_locks,
    workspace_source_lock,
)
from cayu.workspaces.base import (
    Workspace,
    WorkspaceDirectoryPruner,
    WorkspaceGitEntry,
    WorkspaceGitEntryListResult,
    WorkspaceGitEntryObservationUnsupportedError,
    WorkspaceGitMode,
    WorkspaceGitModeMutator,
    WorkspaceListResult,
    WorkspaceMoveResult,
    WorkspaceMutationResult,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
    _local_resource_key,
    _validate_workspace_relative_path,
    _WorkspaceListCollector,
    matches_list_pattern,
    validate_list_pattern,
)
from cayu.workspaces.branches import (
    WorkspaceBranchBindingAuthorityClaimScope,
    WorkspaceBranchBindingAuthorityProvider,
    WorkspaceBranchCapabilities,
    WorkspaceBranchCreationResult,
    WorkspaceBranchLifecycleInspection,
    WorkspaceBranchLifecycleSummary,
    WorkspaceBranchPublicationStrength,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRecoveryResult,
    WorkspaceBranchRecoveryStrength,
    WorkspaceBranchRequest,
    WorkspaceBranchRetentionStrength,
    WorkspaceBranchStore,
    WorkspaceBranchStoreDurability,
    _WorkspaceBranchLifecycleRegistry,
)


class LocalWorkspace(Workspace, WorkspaceGitModeMutator, WorkspaceDirectoryPruner):
    """Filesystem workspace rooted at one local directory.

    ``excluded_directory_names`` and ``excluded_path_patterns`` remove matching
    paths from listing and reject every direct read, resolution, or mutation.
    This opt-in projected view does not advertise workspace branching because
    a branch must not regain access to excluded source authority.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        workspace_id: str | None = None,
        excluded_directory_names: Iterable[str] = (),
        excluded_path_patterns: Iterable[str] = (),
        branch_store: WorkspaceBranchStore | None = None,
        branch_authority_resolver: WorkspaceBranchBindingAuthorityProvider | None = None,
    ) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalWorkspace root must be a string or Path.")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {root_path}")

        if workspace_id is None:
            self.id = str(root_path)
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")
        self.workspace_id_explicit = workspace_id is not None
        self.root = root_path
        self.excluded_directory_names = _validate_excluded_directory_names(excluded_directory_names)
        self._excluded_directory_keys = frozenset(
            _directory_name_key(value) for value in self.excluded_directory_names
        )
        self.excluded_path_patterns = _validate_excluded_path_patterns(excluded_path_patterns)
        self._excluded_path_pattern_keys = tuple(
            _normalized_exclusion_path(pattern) for pattern in self.excluded_path_patterns
        )
        if (self.excluded_directory_names or self.excluded_path_patterns) and (
            branch_store is not None or branch_authority_resolver is not None
        ):
            raise ValueError(
                "LocalWorkspace path exclusions cannot be combined with "
                "workspace branch persistence."
            )
        if branch_store is not None and not isinstance(branch_store, WorkspaceBranchStore):
            raise TypeError("LocalWorkspace branch_store must implement WorkspaceBranchStore.")
        self._branch_store = branch_store
        if branch_store is None:
            self._branch_store_durability = None
        else:
            try:
                self._branch_store_durability = WorkspaceBranchStoreDurability(
                    branch_store.durability
                )
            except Exception:
                raise TypeError(
                    "LocalWorkspace branch_store must declare its durability."
                ) from None
        if branch_authority_resolver is not None and not isinstance(
            branch_authority_resolver,
            WorkspaceBranchBindingAuthorityProvider,
        ):
            raise TypeError(
                "LocalWorkspace branch_authority_resolver must own binding-generation claims."
            )
        self._branch_authority_resolver = branch_authority_resolver
        if branch_authority_resolver is None:
            self._branch_claim_scope = None
        else:
            try:
                self._branch_claim_scope = WorkspaceBranchBindingAuthorityClaimScope(
                    branch_authority_resolver.claim_scope
                )
            except Exception:
                raise TypeError(
                    "LocalWorkspace branch_authority_resolver must declare its claim scope."
                ) from None
        self._branch_lifecycle_registry = _WorkspaceBranchLifecycleRegistry()

    @property
    def resource_key(self) -> tuple[object, ...]:
        return _local_resource_key(self.root)

    def tar_copy_policy_identity(self) -> tuple[object, ...]:
        return (
            "cayu-local-workspace-tar-v2",
            tuple(sorted(self._excluded_directory_keys)),
            tuple(sorted(self._excluded_path_pattern_keys)),
        )

    @staticmethod
    def require_path_operations_supported() -> None:
        """Require the primitives used by secure path-addressed operations."""

        _require_descriptor_guard_support()

    def bounded_read_limit(self, max_bytes: int) -> int:
        validated = _validate_limit(max_bytes, "max_bytes")
        if validated is None:
            raise TypeError("Workspace max_bytes must be an integer.")
        return validated

    def branch_capabilities(self) -> WorkspaceBranchCapabilities:
        if type(self) is not LocalWorkspace or self._has_path_exclusions:
            return WorkspaceBranchCapabilities()
        durable = (
            self._branch_store is not None
            and self._branch_store_durability is WorkspaceBranchStoreDurability.DURABLE
            and self._branch_authority_resolver is not None
            and self._branch_claim_scope is WorkspaceBranchBindingAuthorityClaimScope.DURABLE
        )
        return WorkspaceBranchCapabilities(
            isolation=True,
            net_changes=True,
            publication=WorkspaceBranchPublicationStrength.COOPERATIVE_ATOMIC,
            recovery=(
                WorkspaceBranchRecoveryStrength.DURABLE
                if durable
                else WorkspaceBranchRecoveryStrength.PROCESS_LOCAL
            ),
            retention=(
                WorkspaceBranchRetentionStrength.DURABLE
                if durable
                else WorkspaceBranchRetentionStrength.PROCESS_LOCAL
            ),
            lifecycle_inspection=(
                WorkspaceBranchLifecycleInspection.RECOVERABLE_BY_ID
                if durable
                else WorkspaceBranchLifecycleInspection.ATTACHED
            ),
            detail_code=(
                "durable_local_workspace_branches"
                if durable
                else "process_local_workspace_branches"
            ),
        )

    def branch_lifecycle_summary(self) -> WorkspaceBranchLifecycleSummary:
        if type(self) is not LocalWorkspace:
            return WorkspaceBranchLifecycleSummary(
                attached_count=0,
                statuses=(),
                truncated=False,
            )
        return self._branch_lifecycle_registry.summary()

    async def create_branch(
        self,
        request: WorkspaceBranchRequest,
    ) -> WorkspaceBranchCreationResult:
        if self._has_path_exclusions:
            return await Workspace.create_branch(self, request)
        from cayu.workspaces._local_branch import create_local_workspace_branch

        result = await create_local_workspace_branch(self, request)
        self._branch_lifecycle_registry.attach(result.branch)
        return result

    async def recover_branch(
        self,
        request: WorkspaceBranchRecoveryRequest,
    ) -> WorkspaceBranchRecoveryResult:
        """Recover one durable local branch from store and filesystem evidence."""

        if self._has_path_exclusions:
            raise ValueError(
                "Local workspace branch recovery is unavailable when directories are excluded."
            )

        from cayu.workspaces._local_branch import recover_local_workspace_branch

        result = await recover_local_workspace_branch(self, request)
        self._branch_lifecycle_registry.attach(result.branch)
        return result

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        validated_offset = _validate_offset(offset)
        limit = _validate_limit(max_bytes, "max_bytes")
        return await asyncio.to_thread(
            _read_file_locked,
            self.root,
            relative_path,
            validated_offset,
            limit,
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        await asyncio.to_thread(_write_file, self.root, relative_path, content)

    async def write_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        git_mode: WorkspaceGitMode,
    ) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        mode = _git_mode_bits(git_mode)
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        await asyncio.to_thread(
            _write_file_with_mode,
            self.root,
            relative_path,
            content,
            mode,
        )

    async def delete(self, path: str) -> None:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        await asyncio.to_thread(_delete_file, self.root, relative_path)

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        return await asyncio.to_thread(_create_file, self.root, relative_path, content)

    async def create_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        git_mode: WorkspaceGitMode,
    ) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        mode = _git_mode_bits(git_mode)
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        return await asyncio.to_thread(
            _create_file,
            self.root,
            relative_path,
            content,
            mode,
        )

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace replace content must be bytes.")
        expected_revision = _validate_revision(expected_revision)
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        return await asyncio.to_thread(
            _replace_file,
            self.root,
            relative_path,
            content,
            expected_revision,
        )

    async def replace_bytes_with_git_mode(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
        expected_git_mode: WorkspaceGitMode,
        git_mode: WorkspaceGitMode,
    ) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace replace content must be bytes.")
        expected_revision = _validate_revision(expected_revision)
        expected_mode = _git_mode_bits(expected_git_mode)
        replacement_mode = _git_mode_bits(git_mode)
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        return await asyncio.to_thread(
            _replace_file,
            self.root,
            relative_path,
            content,
            expected_revision,
            expected_mode,
            replacement_mode,
        )

    async def prune_empty_directories(self, path: str, *, max_directories: int) -> None:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        if type(max_directories) is not int or max_directories < 1:
            raise ValueError("max_directories must be a positive integer.")

        def prune():
            with workspace_source_lock(self.root, exclusive=True):
                prune_empty_directories(
                    self.root, relative_path, max_directories, self._require_path_allowed
                )

        await asyncio.to_thread(prune)

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        expected_revision = _validate_revision(expected_revision)
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        return await asyncio.to_thread(
            _delete_file_if_revision,
            self.root,
            relative_path,
            expected_revision,
        )

    async def require_absent(self, path: str) -> None:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        await asyncio.to_thread(_require_file_absent, self.root, relative_path)

    async def move_if_revision(
        self,
        source_path: str,
        destination_path: str,
        *,
        expected_source_revision: str,
        require_destination_absent: bool = True,
    ) -> WorkspaceMoveResult:
        if type(require_destination_absent) is not bool:
            raise TypeError("Workspace require_destination_absent must be a bool.")
        if not require_destination_absent:
            raise ValueError("Workspace moves must require an absent destination.")
        expected_source_revision = _validate_revision(expected_source_revision)
        source = _validate_workspace_relative_path(source_path)
        destination = _validate_workspace_relative_path(destination_path)
        self._require_path_allowed(source)
        self._require_path_allowed(destination)
        return await asyncio.to_thread(
            _move_file_if_revision,
            self.root,
            source,
            destination,
            expected_source_revision,
        )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        validated_limit = _validate_limit(limit, "limit")

        return await asyncio.to_thread(
            _list_files,
            self.root,
            pattern,
            validated_limit,
            self._excluded_directory_keys,
            self._excluded_path_pattern_keys,
        )

    async def list_git_entries(self, *, limit: int) -> WorkspaceGitEntryListResult:
        validated_limit = _validate_limit(limit, "limit")
        if validated_limit is None:
            raise TypeError("Workspace Git entry limit must be an integer.")
        return await asyncio.to_thread(
            _list_git_entries,
            self.root,
            validated_limit,
            self._excluded_directory_keys,
            self._excluded_path_pattern_keys,
        )

    async def _observe_transfer_sizes(self, paths: tuple[str, ...]) -> tuple[int, int]:
        """Return the total and largest size of regular files, without reading them.

        SyncBinding sizes its staging reservation from this; the copy itself
        still enforces every limit.
        """

        def measure() -> tuple[int, int]:
            total_bytes = 0
            largest_file_bytes = 0
            for path in paths:
                info = self.resolve_no_symlinks(path).lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"Workspace path is not a regular file: {path}")
                total_bytes += info.st_size
                largest_file_bytes = max(largest_file_bytes, info.st_size)
            return total_bytes, largest_file_bytes

        return await asyncio.to_thread(measure)

    def resolve(self, path: str) -> Path:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        candidate = Path(relative_path)
        resolved = (self.root / candidate).resolve()
        self._ensure_inside_root(resolved)
        if resolved == self.root:
            raise ValueError("Workspace paths must reference a file.")
        self._require_path_allowed(resolved.relative_to(self.root).as_posix())
        return resolved

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Workspace path escapes the workspace root.") from exc

    def resolve_no_symlinks(self, path: str) -> Path:
        relative_path = _validate_workspace_relative_path(path)
        self._require_path_allowed(relative_path)
        candidate = Path(relative_path)
        target = self._resolve_without_symlinks(candidate)
        resolved = target.resolve(strict=False)
        self._ensure_inside_root(resolved)
        if resolved == self.root:
            raise ValueError("Workspace paths must reference a file.")
        return target

    def _resolve_without_symlinks(self, candidate: Path) -> Path:
        current = self.root
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                current = (current / part).resolve(strict=False)
                self._ensure_inside_root(current)
                continue
            current = current / part
            if current.is_symlink():
                raise ValueError("Workspace path escapes the workspace root.")
        return current

    def _require_path_allowed(self, relative_path: str) -> None:
        if _path_has_excluded_directory(relative_path, self._excluded_directory_keys):
            raise ValueError("Workspace path is inside an excluded directory.")
        if _path_matches_excluded_pattern(relative_path, self._excluded_path_pattern_keys):
            raise ValueError("Workspace path matches an excluded path pattern.")

    @property
    def _has_path_exclusions(self) -> bool:
        return bool(self.excluded_directory_names or self.excluded_path_patterns)


def _write_file(root: Path, relative_path: str, content: bytes) -> None:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        write_regular(root, relative_path, content)


def _write_file_with_mode(
    root: Path,
    relative_path: str,
    content: bytes,
    mode: int,
) -> None:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        restore_regular(root, relative_path, content, mode=mode)


def _delete_file(root: Path, relative_path: str) -> None:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        delete_regular(root, relative_path)


def _read_file_locked(
    root: Path,
    relative_path: str,
    offset: int,
    max_bytes: int | None,
) -> WorkspaceReadResult:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        return _read_file(root, relative_path, offset, max_bytes)


def _read_file(
    root: Path,
    relative_path: str,
    offset: int,
    max_bytes: int | None,
) -> WorkspaceReadResult:
    with open_regular_for_read(root, relative_path) as (file, total_bytes):
        if offset > total_bytes:
            raise WorkspaceReadOffsetError(offset, total_bytes)
        file.seek(offset)
        # A sized read allocates its whole size up front, so ask for no more
        # than the file holds; the extra byte still reveals a file that grew.
        content = (
            file.read()
            if max_bytes is None
            else file.read(min(max_bytes, total_bytes - offset + 1))
        )
        file_mode = os.fstat(file.fileno()).st_mode
    complete = offset == 0 and len(content) == total_bytes
    revision, digest = content_identity(content) if complete else (None, None)
    return WorkspaceReadResult(
        content=content,
        total_bytes=total_bytes,
        truncated=offset + len(content) < total_bytes,
        offset=offset,
        revision=revision,
        sha256=digest,
        git_mode=("100755" if file_mode & 0o111 else "100644") if complete else None,
    )


def _create_file(
    root: Path,
    relative_path: str,
    content: bytes,
    mode: int | None = None,
) -> WorkspaceMutationResult:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        create_regular(root, relative_path, content, mode=mode)
        return mutation_result("create", before=None, after=content)


def _replace_file(
    root: Path,
    relative_path: str,
    content: bytes,
    expected_revision: str,
    expected_git_mode: int | None = None,
    replacement_mode: int | None = None,
) -> WorkspaceMutationResult:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        before = replace_regular_if_revision(
            root,
            relative_path,
            content,
            expected_revision,
            expected_git_mode=expected_git_mode,
            replacement_mode=replacement_mode,
        )
        return mutation_result_from_identities("replace", before=before, after=content)


def _git_mode_bits(git_mode: WorkspaceGitMode) -> int:
    if git_mode == "100644":
        return 0o644
    if git_mode == "100755":
        return 0o755
    raise ValueError("Workspace git_mode must be 100644 or 100755.")


def _delete_file_if_revision(
    root: Path,
    relative_path: str,
    expected_revision: str,
) -> WorkspaceMutationResult:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        before = delete_regular_if_revision(root, relative_path, expected_revision)
        return mutation_result_from_identities("delete", before=before, after=None)


def _require_file_absent(root: Path, relative_path: str) -> None:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_lock(root, relative_path),
    ):
        require_absent_regular(root, relative_path)


def _move_file_if_revision(
    root: Path,
    source_path: str,
    destination_path: str,
    expected_source_revision: str,
) -> WorkspaceMoveResult:
    with (
        workspace_source_lock(root, exclusive=False),
        workspace_path_locks(root, source_path, destination_path),
    ):
        return move_regular_if_revision(
            root,
            source_path,
            destination_path,
            expected_source_revision,
        )


def _list_files(
    root: Path,
    pattern: str,
    limit: int | None,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> WorkspaceListResult:
    with workspace_source_lock(root, exclusive=False):
        collector = _WorkspaceListCollector(limit)
        matched = 0
        with closing(
            _iter_listable_files(root, excluded_directory_keys, excluded_path_pattern_keys)
        ) as candidates:
            for relative_path in candidates:
                if not matches_list_pattern(relative_path, pattern):
                    continue
                collector.add(relative_path)
                matched += 1
                # The walk yields paths in sorted order, so the first match past
                # the limit settles both the page and the truncation flag.
                if limit is not None and matched > limit:
                    break
        return collector.result(exact_total_when_truncated=False)


def _list_git_entries(
    root: Path,
    limit: int,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> WorkspaceGitEntryListResult:
    retained: list[tuple[str, WorkspaceGitEntry]] = []
    total_count = 0

    def add(path: Path, info: os.stat_result) -> None:
        nonlocal total_count
        relative_path = path.relative_to(root).as_posix()
        if _path_matches_excluded_pattern(relative_path, excluded_path_pattern_keys):
            return
        if stat.S_ISLNK(info.st_mode):
            target = os.fsencode(os.readlink(path))
            entry = WorkspaceGitEntry(
                path=relative_path,
                git_mode="120000",
                symlink_target_sha256=sha256(target).hexdigest(),
                symlink_target_bytes=len(target),
            )
        elif stat.S_ISREG(info.st_mode):
            entry = WorkspaceGitEntry(
                path=relative_path,
                git_mode="100755" if info.st_mode & 0o111 else "100644",
            )
        else:
            raise WorkspaceGitEntryObservationUnsupportedError(
                "Workspace contains a non-file, non-directory, non-symlink entry."
            )
        total_count += 1
        insort(retained, (relative_path, entry), key=lambda item: item[0])
        if len(retained) > limit:
            retained.pop()

    with workspace_source_lock(root, exclusive=False):
        for directory, child_directories, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=_raise_workspace_walk_error,
        ):
            parent = Path(directory)
            traversable: list[str] = []
            for name in child_directories:
                if _directory_name_key(name) in excluded_directory_keys:
                    continue
                path = parent / name
                relative_path = path.relative_to(root).as_posix()
                if _path_matches_excluded_pattern(
                    relative_path,
                    excluded_path_pattern_keys,
                ):
                    continue
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    add(path, info)
                elif stat.S_ISDIR(info.st_mode):
                    traversable.append(name)
                else:
                    raise WorkspaceGitEntryObservationUnsupportedError(
                        "Workspace directory traversal encountered an unsupported entry."
                    )
            child_directories[:] = traversable
            for name in filenames:
                path = parent / name
                relative_path = path.relative_to(root).as_posix()
                if _path_has_excluded_directory(relative_path, excluded_directory_keys):
                    continue
                if _path_matches_excluded_pattern(relative_path, excluded_path_pattern_keys):
                    continue
                add(path, path.lstat())
    return WorkspaceGitEntryListResult(
        entries=tuple(entry for _path, entry in retained),
        total_count=total_count,
        truncated=total_count > len(retained),
    )


class _ListEntryNotDirectoryError(Exception):
    """A directory entry stopped being a real directory before it was opened."""


_SUPPORTS_LIST_DIRECTORY_FD = os.scandir in os.supports_fd
_LIST_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_LIST_SEARCH_FLAGS = (
    _SEARCH_BASE_FLAGS | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
# Opening a symlink with O_NOFOLLOW fails with ELOOP (EMLINK on some BSDs), and
# some kernels report ENOTDIR for O_NOFOLLOW | O_DIRECTORY instead.
_LIST_NOT_DIRECTORY_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK, errno.ENOTDIR})
# Errors that mean one directory cannot be read. Anything else, such as running
# out of descriptors or an I/O error, fails the listing instead of shortening it.
_LIST_UNREADABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.ENOENT})
# Directories below the root deeper than this have their descriptors closed and
# are reopened from the nearest open ancestor if the walk needs them again.
_LIST_OPEN_DIRECTORY_LIMIT = 32

_ListEntries = Iterator[tuple[str, str, bool]]


class _ListLevel:
    __slots__ = ("fd", "name", "pending")

    def __init__(self, fd: int | None, name: str, pending: _ListEntries) -> None:
        self.fd = fd
        self.name = name
        self.pending = pending


def _iter_listable_files(
    root: Path,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> Generator[str, None, None]:
    """Yield the workspace's listable regular files as relative POSIX paths, sorted.

    The root is opened one component at a time from ``/`` and every directory
    below it relative to an opened parent, all with ``O_NOFOLLOW``; each is read
    once with ``os.scandir``. Entries are classified from the directory read
    itself (no per-entry ``stat`` where the filesystem reports entry types),
    and anything that is not a regular file or a real directory, including
    every symlink, is skipped. The walk therefore never leaves the root through
    a symlink, even one swapped in while it runs. At most
    ``_LIST_OPEN_DIRECTORY_LIMIT`` directories below the root are open at once.

    Siblings are visited by name, with ``/`` appended to directory names, which
    is exactly the order ``sorted()`` gives the full paths, so callers can stop
    after the first matches they need.
    """
    if not _SUPPORTS_LIST_DIRECTORY_FD:
        raise RuntimeError(
            "LocalWorkspace requires POSIX descriptor-relative filesystem primitives."
        )
    # Listings without exclusions used Path.rglob, which skips directories it
    # cannot read; listings with exclusions used os.walk, which raised. Both
    # behaviors are kept.
    skip_unreadable = not excluded_directory_keys and not excluded_path_pattern_keys
    try:
        root_fd = _open_list_root(root)
    except OSError as exc:
        if exc.errno in _LIST_NOT_DIRECTORY_ERRNOS:
            # The root was resolved when the workspace was created, so a
            # symlink on the way means it or an ancestor was replaced since.
            raise ValueError("Workspace path escapes the workspace root.") from exc
        if skip_unreadable and exc.errno in _LIST_UNREADABLE_ERRNOS:
            return
        raise
    root_entries = _read_list_directory(
        root_fd,
        "",
        excluded_directory_keys,
        excluded_path_pattern_keys,
        skip_unreadable,
    )
    if root_entries is None:
        return
    stack = [_ListLevel(root_fd, "", root_entries)]
    try:
        while stack:
            level = stack[-1]
            for name, relative_path, is_directory in level.pending:
                if not is_directory:
                    yield relative_path
                    continue
                if level.fd is None:
                    level.fd = _reopen_list_level(stack, skip_unreadable)
                    if level.fd is None:
                        stack.pop()
                        break
                try:
                    child = _open_list_directory(
                        level.fd,
                        name,
                        relative_path + "/",
                        excluded_directory_keys,
                        excluded_path_pattern_keys,
                        skip_unreadable,
                    )
                except _ListEntryNotDirectoryError:
                    # Replaced by a symlink or a file since it was read; the
                    # listing skips those either way.
                    continue
                if child is not None:
                    stack.append(_ListLevel(child[0], name, child[1]))
                    # The root stays open, so a reopen never starts above it.
                    if len(stack) > _LIST_OPEN_DIRECTORY_LIMIT + 1:
                        ancestor = stack[-_LIST_OPEN_DIRECTORY_LIMIT - 1]
                        if ancestor.fd is not None:
                            os.close(ancestor.fd)
                            ancestor.fd = None
                    break
            else:
                stack.pop()
                if level.fd is not None:
                    os.close(level.fd)
    finally:
        for level in stack:
            if level.fd is not None:
                os.close(level.fd)


def _open_list_root(root: Path) -> int:
    """Open the absolute root for reading without following any symlink on the way."""
    parts = root.parts
    if len(parts) == 1:
        return os.open(parts[0], _LIST_DIRECTORY_FLAGS)
    # Ancestors only need search permission, not read permission.
    current = os.open(parts[0], _LIST_SEARCH_FLAGS)
    try:
        for index, name in enumerate(parts[1:], start=2):
            flags = _LIST_DIRECTORY_FLAGS if index == len(parts) else _LIST_SEARCH_FLAGS
            descriptor = os.open(name, flags, dir_fd=current)
            os.close(current)
            current = descriptor
        opened, current = current, None
        return opened
    finally:
        if current is not None:
            os.close(current)


def _reopen_list_level(stack: list[_ListLevel], skip_unreadable: bool) -> int | None:
    """Reopen the top level's directory from its nearest open ancestor.

    Every component is opened with ``O_NOFOLLOW``, so a directory replaced by
    a symlink in the meantime is not followed; the rest of that directory is
    then skipped, as is one that became unreadable when ``skip_unreadable``.
    """
    base = len(stack) - 1
    while stack[base].fd is None:
        base -= 1
    parent = stack[base].fd
    opened: int | None = None
    try:
        for level in stack[base + 1 :]:
            try:
                descriptor = os.open(level.name, _LIST_DIRECTORY_FLAGS, dir_fd=parent)
            except OSError as exc:
                if exc.errno in _LIST_NOT_DIRECTORY_ERRNOS:
                    return None
                if skip_unreadable and exc.errno in _LIST_UNREADABLE_ERRNOS:
                    return None
                raise
            if opened is not None:
                os.close(opened)
            opened = parent = descriptor
        reopened, opened = opened, None
        return reopened
    finally:
        if opened is not None:
            os.close(opened)


def _open_list_directory(
    parent_fd: int,
    name: str,
    prefix: str,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
    skip_unreadable: bool,
) -> tuple[int, _ListEntries] | None:
    """Open and read one subdirectory; ``None`` when it is skipped as unreadable."""
    try:
        directory_fd = os.open(name, _LIST_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in _LIST_NOT_DIRECTORY_ERRNOS:
            raise _ListEntryNotDirectoryError(name) from exc
        if skip_unreadable and exc.errno in _LIST_UNREADABLE_ERRNOS:
            return None
        raise
    entries = _read_list_directory(
        directory_fd,
        prefix,
        excluded_directory_keys,
        excluded_path_pattern_keys,
        skip_unreadable,
    )
    if entries is None:
        return None
    return directory_fd, entries


def _read_list_directory(
    directory_fd: int,
    prefix: str,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
    skip_unreadable: bool,
) -> _ListEntries | None:
    """Return one opened directory's listable entries in walk order.

    Entries are ``(name, relative_path, is_directory)``. Excluded names and
    paths are dropped here, so excluded directories are never opened. The
    descriptor is closed when this returns ``None`` or raises.
    """
    keyed: list[tuple[str, str, str, bool]] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                entry_name = entry.name
                try:
                    if entry.is_dir(follow_symlinks=False):
                        is_directory = True
                    elif entry.is_file(follow_symlinks=False):
                        is_directory = False
                    else:
                        continue
                except OSError as exc:
                    # Filesystems without entry types need a stat here; an
                    # I/O error must not look like a missing file.
                    if not skip_unreadable or exc.errno not in _LIST_UNREADABLE_ERRNOS:
                        raise
                    continue
                if _path_has_excluded_directory(entry_name, excluded_directory_keys):
                    continue
                relative_path = prefix + entry_name
                if _path_matches_excluded_pattern(relative_path, excluded_path_pattern_keys):
                    continue
                sort_key = entry_name + "/" if is_directory else entry_name
                keyed.append((sort_key, entry_name, relative_path, is_directory))
    except BaseException as exc:
        os.close(directory_fd)
        if skip_unreadable and isinstance(exc, OSError) and exc.errno in _LIST_UNREADABLE_ERRNOS:
            return None
        raise
    keyed.sort()
    return ((entry_name, path, is_dir) for _key, entry_name, path, is_dir in keyed)


def _raise_workspace_walk_error(error: OSError) -> None:
    raise error


def _validate_excluded_directory_names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("LocalWorkspace excluded_directory_names must be an iterable of strings.")
    try:
        copied = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "LocalWorkspace excluded_directory_names must be an iterable of strings."
        ) from exc
    validated: list[str] = []
    seen: set[str] = set()
    for value in copied:
        if type(value) is not str:
            raise TypeError("LocalWorkspace excluded_directory_names entries must be strings.")
        value = require_unicode_scalar_text(
            require_clean_nonblank(value, "excluded directory name"),
            "excluded directory name",
        )
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(
                "LocalWorkspace excluded directory names must be single path segments."
            )
        key = _directory_name_key(value)
        if key in seen:
            raise ValueError("LocalWorkspace excluded directory names must be unique.")
        seen.add(key)
        validated.append(value)
    return tuple(validated)


def _validate_excluded_path_patterns(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("LocalWorkspace excluded_path_patterns must be an iterable of strings.")
    try:
        copied = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "LocalWorkspace excluded_path_patterns must be an iterable of strings."
        ) from exc
    validated: list[str] = []
    seen: set[str] = set()
    for value in copied:
        if type(value) is not str:
            raise TypeError("LocalWorkspace excluded_path_patterns entries must be strings.")
        value = require_unicode_scalar_text(
            require_clean_nonblank(value, "excluded path pattern"),
            "excluded path pattern",
        )
        if "\\" in value:
            raise ValueError("LocalWorkspace excluded path patterns must use POSIX separators.")
        value = validate_list_pattern(value)
        normalized = _normalized_exclusion_path(value)
        if normalized in seen:
            raise ValueError("LocalWorkspace excluded path patterns must be unique.")
        seen.add(normalized)
        validated.append(value)
    return tuple(validated)


def _directory_name_key(value: str) -> str:
    return value.rstrip(" .").casefold()


def _path_has_excluded_directory(
    relative_path: str,
    excluded_directory_keys: frozenset[str],
) -> bool:
    if not excluded_directory_keys:
        return False
    return any(
        _directory_name_key(part) in excluded_directory_keys
        for part in relative_path.replace("\\", "/").split("/")
        if part
    )


def _normalized_exclusion_path(value: str) -> str:
    return "/".join(
        _directory_name_key(part)
        for part in value.replace("\\", "/").split("/")
        if part not in {"", "."}
    )


def _path_matches_excluded_pattern(
    relative_path: str,
    excluded_path_pattern_keys: tuple[str, ...],
) -> bool:
    if not excluded_path_pattern_keys:
        return False
    normalized_parts = tuple(
        part for part in _normalized_exclusion_path(relative_path).split("/") if part
    )
    return any(
        matches_list_pattern("/".join(normalized_parts[:end]), pattern)
        for pattern in excluded_path_pattern_keys
        for end in range(1, len(normalized_parts) + 1)
    )


def _validate_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"Workspace {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Workspace {field_name} must be greater than zero.")
    return value


def _validate_offset(value: int) -> int:
    if type(value) is not int:
        raise TypeError("Workspace offset must be an integer.")
    if value < 0:
        raise ValueError("Workspace offset must be non-negative.")
    return value


def _validate_revision(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("Workspace expected_revision must be a nonblank string.")
    return value

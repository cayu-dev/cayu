from __future__ import annotations

import asyncio
import os
import stat
from bisect import insort
from collections.abc import Iterable
from hashlib import sha256
from os import PathLike
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.workspaces._local_guard import (
    _require_descriptor_guard_support,
    create_regular,
    delete_regular,
    delete_regular_if_revision,
    move_regular_if_revision,
    open_regular_for_read,
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


class LocalWorkspace(Workspace, WorkspaceGitModeMutator):
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
        content = file.read() if max_bytes is None else file.read(max_bytes)
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
        for path in _iter_list_file_candidates(
            root,
            excluded_directory_keys,
            excluded_path_pattern_keys,
        ):
            if _has_symlink_component(root, path):
                continue
            resolved = path.resolve()
            _ensure_inside_root(root, resolved)
            if resolved == root or not resolved.is_file():
                continue
            relative_path = resolved.relative_to(root).as_posix()
            if _path_has_excluded_directory(relative_path, excluded_directory_keys):
                continue
            if _path_matches_excluded_pattern(relative_path, excluded_path_pattern_keys):
                continue
            if not matches_list_pattern(relative_path, pattern):
                continue
            collector.add(relative_path)
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


def _iter_list_file_candidates(
    root: Path,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> Iterable[Path]:
    if not excluded_directory_keys and not excluded_path_pattern_keys:
        yield from root.rglob("*")
        return

    for directory, child_directories, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=_raise_workspace_walk_error,
    ):
        parent = Path(directory)
        child_directories[:] = [
            name
            for name in child_directories
            if _directory_name_key(name) not in excluded_directory_keys
            and not _path_matches_excluded_pattern(
                (parent / name).relative_to(root).as_posix(),
                excluded_path_pattern_keys,
            )
            and not (parent / name).is_symlink()
        ]
        for filename in filenames:
            yield parent / filename


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


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _ensure_inside_root(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Workspace path escapes the workspace root.") from exc


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

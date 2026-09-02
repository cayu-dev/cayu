from __future__ import annotations

import base64
import io
import json
import posixpath
from collections.abc import Iterable, Sequence
from typing import Any, BinaryIO

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    require_clean_nonblank,
    require_durable_text,
    require_nonblank,
)
from cayu.runners import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    LocalRunner,
    RemoteWorkspaceBranchCapability,
    Runner,
    RunnerBinaryStreamCapability,
)
from cayu.workspaces._guest_guard import (
    GUEST_DESCRIPTOR_GUARD_SOURCE,
    guard_create,
    guard_delete_if_revision,
    guard_move_if_revision,
    guard_read,
    guard_replace,
    guard_require_absent,
)
from cayu.workspaces._tar import tar_archive_size_bound
from cayu.workspaces.base import (
    BoundedTarReader,
    BoundedTarStreamReader,
    RunnerBoundWorkspace,
    TarStreamReadResult,
    TarStreamWriter,
    TarWriter,
    WorkspaceGitEntry,
    WorkspaceGitEntryListResult,
    WorkspaceListResult,
    WorkspaceMoveResult,
    WorkspaceMutationResult,
    WorkspaceReadResult,
    _local_resource_key,
    _runner_resource_key,
    _validate_workspace_offset,
    _validate_workspace_positive_limit,
    _validate_workspace_relative_path,
    _validate_workspace_revision,
    matches_list_pattern,
    translate_list_pattern,
    validate_list_pattern,
)
from cayu.workspaces.branches import (
    RemoteWorkspaceBranchAuthorityProvider,
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
    _WorkspaceBranchLifecycleRegistry,
)

DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES = 256 * 1024
DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT = 500
RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES = 4096
RUNNER_WORKSPACE_LIST_PAYLOAD_LIMIT_BYTES = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES

_RUNNER_WORKSPACE_PROGRAM = (
    GUEST_DESCRIPTOR_GUARD_SOURCE
    + r"""
import base64
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile

SYNC_GIT_MODE_TAR_OWNER = "cayu.git-mode.v1"

# Match RunnerWorkspace's historical pathlib creation behavior: missing
# directories and files start from the conventional permissive modes, then the
# guest process's umask determines their effective permissions.
GUARDED_DIRECTORY_CREATE_MODE = 0o777
GUARDED_FILE_CREATE_MODE = 0o666

# CAYU_TEST_BOUNDED_TAR_MEMBER_READS


_BINARY_STDOUT = False


def fail(error_type, message):
    print(
        json.dumps({"ok": False, "error_type": error_type, "message": message}),
        file=sys.stderr if _BINARY_STDOUT else sys.stdout,
    )
    sys.exit(1)


def close_fd(fd):
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def path_matches_excluded_pattern(path, excluded_path_regexes):
    normalized_parts = tuple(
        part.rstrip(" .").casefold()
        for part in path.replace("\\", "/").split("/")
        if part not in ("", ".")
    )
    return any(
        pattern.fullmatch("/".join(normalized_parts[:end]))
        for pattern in excluded_path_regexes
        for end in range(1, len(normalized_parts) + 1)
    )


def directory_name_key(value):
    return value.rstrip(" .").casefold()


def fail_guard(exc, path=None):
    if exc.status == "escape":
        fail("invalid_path", "Workspace path escapes the workspace root.")
    if exc.status == "hardlink":
        fail("invalid_path", f"Workspace file has multiple hard links: {path}")
    if exc.status == "unsupported":
        fail(
            "workspace_error",
            "Runner workspace requires POSIX descriptor-relative filesystem primitives.",
        )
    if exc.status in ("enoent", "notdir"):
        fail("not_found", f"Workspace file not found: {path}")
    if exc.status in ("notfile", "isdir"):
        fail("not_file", f"Workspace path is not a file: {path}")
    fail("workspace_error", f"Unexpected workspace guard status: {exc.status}")


def open_path(root_fd, rel_path, create=False):
    parts = guarded_parts(rel_path)
    return open_guarded_parent(root_fd, parts, create)


def read_operation(root_fd):
    rel_path = sys.argv[2]
    limit = int(sys.argv[3])
    parent_fd = None
    leaf_fd = None
    try:
        parent_fd, leaf_name = open_path(root_fd, rel_path)
        leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
        chunks = []
        remaining = limit
        while remaining > 0:
            chunk = os.read(leaf_fd, min(remaining, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        print(json.dumps({
            "ok": True,
            "content_base64": base64.b64encode(content).decode("ascii"),
            "total_bytes": max(info.st_size, len(content)),
        }))
    except GuardPathError as exc:
        if exc.status == "notfile":
            fail("not_found", f"Workspace file not found: {rel_path}")
        fail_guard(exc, rel_path)
    finally:
        close_fd(leaf_fd)
        close_fd(parent_fd)


def write_operation(root_fd):
    payload = json.loads(sys.stdin.read())
    rel_path = payload["path"]
    content = base64.b64decode(payload["content_base64"], validate=True)
    parent_fd = None
    try:
        with workspace_path_lock(root_fd, rel_path):
            parent_fd, leaf_name = open_path(root_fd, rel_path, create=True)
            write_guarded_regular(leaf_name, parent_fd, (content,))
            print(json.dumps({"ok": True, "bytes": len(content)}))
    except GuardPathError as exc:
        fail_guard(exc, rel_path)
    finally:
        close_fd(parent_fd)


def delete_operation(root_fd):
    rel_path = sys.argv[2]
    parent_fd = None
    try:
        with workspace_path_lock(root_fd, rel_path):
            parent_fd, leaf_name = open_path(root_fd, rel_path)
            deleted = delete_guarded_regular(leaf_name, parent_fd)
            print(json.dumps({"ok": True, "deleted": deleted}))
    except GuardPathError as exc:
        if exc.status in ("enoent", "notdir"):
            print(json.dumps({"ok": True, "deleted": False}))
            return
        fail_guard(exc, rel_path)
    finally:
        close_fd(parent_fd)


def collect_files(
    dir_fd,
    prefix,
    pattern_regex,
    matches,
    ancestor_directories,
    excluded_directory_names,
    excluded_path_regexes,
):
    if os.listdir not in getattr(os, "supports_fd", ()):
        raise GuardPathError("unsupported")
    directory_info = os.fstat(dir_fd)
    identity = (directory_info.st_dev, directory_info.st_ino)
    # CAYU_TEST_DIRECTORY_IDENTITY_ALIAS
    if identity in ancestor_directories:
        return
    ancestor_directories.add(identity)
    try:
        for name in os.listdir(dir_fd):
            rel_path = name if not prefix else prefix + "/" + name
            if directory_name_key(name) in excluded_directory_names:
                continue
            if path_matches_excluded_pattern(rel_path, excluded_path_regexes):
                continue
            try:
                entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except OSError as exc:
                if exc.errno in MISSING_ERRNOS:
                    continue
                raise
            if stat.S_ISLNK(entry.st_mode):
                continue
            if stat.S_ISDIR(entry.st_mode):
                try:
                    child_fd = open_guarded_directory(name, dir_fd, False, True)
                except GuardPathError as exc:
                    if exc.status in ("enoent", "escape"):
                        continue
                    raise
                try:
                    collect_files(
                        child_fd,
                        rel_path,
                        pattern_regex,
                        matches,
                        ancestor_directories,
                        excluded_directory_names,
                        excluded_path_regexes,
                    )
                finally:
                    close_fd(child_fd)
                continue
            if not stat.S_ISREG(entry.st_mode):
                continue
            # Listing is metadata-only: this directory-relative, no-follow
            # stat is the atomic observation of the leaf. Opening the file
            # would unnecessarily require content-read permission.
            if pattern_regex.fullmatch(rel_path):
                matches.append(rel_path)
    finally:
        ancestor_directories.remove(identity)


def list_operation(root_fd):
    pattern_regex = re.compile(sys.argv[2])
    limit = int(sys.argv[3])
    payload_limit = int(sys.argv[4])
    excluded_directory_names = frozenset(json.loads(sys.argv[5]))
    excluded_path_regexes = tuple(
        re.compile(pattern) for pattern in json.loads(sys.argv[6])
    )
    matches = []
    collect_files(
        root_fd,
        "",
        pattern_regex,
        matches,
        set(),
        excluded_directory_names,
        excluded_path_regexes,
    )
    sorted_matches = sorted(matches)
    paths = sorted_matches[:limit]
    total_count = len(matches)

    def serialize(path_count):
        return json.dumps(
            {"ok": True, "paths": paths[:path_count], "total_count": total_count},
            separators=(",", ":"),
        )

    low = 0
    high = len(paths)
    while low < high:
        candidate = (low + high + 1) // 2
        encoded = (serialize(candidate) + "\n").encode("utf-8")
        if len(encoded) <= payload_limit:
            low = candidate
        else:
            high = candidate - 1
    payload = serialize(low) + "\n"
    if len(payload.encode("utf-8")) > payload_limit:
        fail("workspace_error", "Workspace list result exceeds its transfer limit.")
    sys.stdout.write(payload)


def collect_git_entries(
    dir_fd,
    prefix,
    entries,
    ancestor_directories,
    excluded_directory_names,
    excluded_path_regexes,
    limit,
):
    if os.listdir not in getattr(os, "supports_fd", ()):
        raise GuardPathError("unsupported")
    if os.readlink not in getattr(os, "supports_dir_fd", ()):
        raise GuardPathError("unsupported")
    directory_info = os.fstat(dir_fd)
    identity = (directory_info.st_dev, directory_info.st_ino)
    if identity in ancestor_directories:
        raise GuardPathError("escape")
    ancestor_directories.add(identity)
    try:
        for name in sorted(os.listdir(dir_fd)):
            rel_path = name if not prefix else prefix + "/" + name
            if directory_name_key(name) in excluded_directory_names:
                continue
            if path_matches_excluded_pattern(rel_path, excluded_path_regexes):
                continue
            try:
                entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except OSError as exc:
                if exc.errno in MISSING_ERRNOS:
                    fail("workspace_error", "Workspace changed during Git entry observation.")
                raise
            if stat.S_ISLNK(entry.st_mode):
                target = os.fsencode(os.readlink(name, dir_fd=dir_fd))
                entries.append({
                    "path": rel_path,
                    "git_mode": "120000",
                    "symlink_target_sha256": hashlib.sha256(target).hexdigest(),
                    "symlink_target_bytes": len(target),
                })
            elif stat.S_ISDIR(entry.st_mode):
                try:
                    child_fd = open_guarded_directory(name, dir_fd, False, True)
                except GuardPathError as exc:
                    if exc.status in ("enoent", "escape"):
                        fail("workspace_error", "Workspace changed during Git entry observation.")
                    raise
                try:
                    if collect_git_entries(
                        child_fd,
                        rel_path,
                        entries,
                        ancestor_directories,
                        excluded_directory_names,
                        excluded_path_regexes,
                        limit,
                    ):
                        return True
                finally:
                    close_fd(child_fd)
                continue
            elif stat.S_ISREG(entry.st_mode):
                entries.append({
                    "path": rel_path,
                    "git_mode": "100755" if entry.st_mode & 0o111 else "100644",
                    "symlink_target_sha256": None,
                    "symlink_target_bytes": None,
                })
            else:
                fail(
                    "workspace_error",
                    "Workspace contains an unsupported Git-significant entry type.",
                )
            if len(entries) > limit:
                return True
        return False
    finally:
        ancestor_directories.remove(identity)


def git_entries_operation(root_fd):
    limit = int(sys.argv[2])
    payload_limit = int(sys.argv[3])
    excluded_directory_names = frozenset(json.loads(sys.argv[4]))
    excluded_path_regexes = tuple(
        re.compile(pattern) for pattern in json.loads(sys.argv[5])
    )
    entries = []
    truncated = collect_git_entries(
        root_fd,
        "",
        entries,
        set(),
        excluded_directory_names,
        excluded_path_regexes,
        limit,
    )
    entries.sort(key=lambda item: item["path"])
    total_count = len(entries)

    def serialize(entry_count):
        return json.dumps(
            {
                "ok": True,
                "entries": entries[:entry_count],
                "total_count": total_count,
                "truncated": truncated or entry_count < total_count,
            },
            separators=(",", ":"),
        )

    low = 0
    high = min(limit, len(entries))
    while low < high:
        candidate = (low + high + 1) // 2
        encoded = (serialize(candidate) + "\n").encode("utf-8")
        if len(encoded) <= payload_limit:
            low = candidate
        else:
            high = candidate - 1
    payload = serialize(low) + "\n"
    if len(payload.encode("utf-8")) > payload_limit:
        fail("workspace_error", "Workspace Git entry result exceeds its transfer limit.")
    sys.stdout.write(payload)


def read_tar_preflight(root_fd):
    payload = json.loads(sys.stdin.read())
    rel_paths = payload["paths"]
    max_file_bytes = payload["max_file_bytes"]
    max_total_bytes = payload["max_total_bytes"]
    max_archive_bytes = payload["max_archive_bytes"]
    archive_overhead_bytes = payload["archive_overhead_bytes"]
    preflight_files = []
    total_bytes = 0
    for rel_path in rel_paths:
        parent_fd = None
        leaf_fd = None
        try:
            parent_fd, leaf_name = open_path(root_fd, rel_path)
            leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
        except GuardPathError as exc:
            if exc.status == "notfile":
                fail("not_found", f"Workspace file not found: {rel_path}")
            fail_guard(exc, rel_path)
        finally:
            close_fd(leaf_fd)
            close_fd(parent_fd)
        size = info.st_size
        if max_file_bytes is not None and size > max_file_bytes:
            fail(
                "workspace_error",
                f"Workspace file exceeds max_file_bytes={max_file_bytes}: {rel_path}",
            )
        total_bytes += size
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            fail(
                "workspace_error",
                f"Workspace files exceed max_total_bytes={max_total_bytes}.",
            )
        git_mode = 0o755 if info.st_mode & 0o111 else 0o644
        preflight_files.append((rel_path, info.st_dev, info.st_ino, size, git_mode))
    archive_size_bound = total_bytes + archive_overhead_bytes
    if max_archive_bytes is not None and archive_size_bound > max_archive_bytes:
        fail(
            "workspace_error",
            f"Workspace tar exceeds max_archive_bytes={max_archive_bytes}.",
        )

    return rel_paths, preflight_files, total_bytes


def write_tar_from_workspace(root_fd, preflight_files, fileobj, mode):
    with tarfile.open(fileobj=fileobj, mode=mode) as archive:
        for rel_path, expected_dev, expected_ino, size, git_mode in preflight_files:
            parent_fd = None
            leaf_fd = None
            try:
                parent_fd, leaf_name = open_path(root_fd, rel_path)
                leaf_fd, current = open_guarded_regular(leaf_name, parent_fd)
                if (
                    current.st_dev != expected_dev
                    or current.st_ino != expected_ino
                    or current.st_size != size
                    or (0o755 if current.st_mode & 0o111 else 0o644) != git_mode
                ):
                    fail(
                        "workspace_error",
                        f"Workspace file changed during archive preflight: {rel_path}",
                    )
                info = tarfile.TarInfo(name=rel_path)
                info.size = size
                info.mode = git_mode
                info.uname = SYNC_GIT_MODE_TAR_OWNER
                with os.fdopen(leaf_fd, "rb") as file:
                    leaf_fd = None
                    archive.addfile(info, file)
            except GuardPathError as exc:
                if exc.status == "notfile":
                    fail("not_found", f"Workspace file not found: {rel_path}")
                fail_guard(exc, rel_path)
            finally:
                close_fd(leaf_fd)
                close_fd(parent_fd)


def read_tar_operation(root_fd):
    rel_paths, preflight_files, total_bytes = read_tar_preflight(root_fd)
    buffer = io.BytesIO()
    write_tar_from_workspace(root_fd, preflight_files, buffer, "w")
    print(json.dumps({
        "ok": True,
        "tar_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "file_count": len(rel_paths),
        "total_bytes": total_bytes,
    }))


def read_tar_stream_operation(root_fd):
    _, preflight_files, _ = read_tar_preflight(root_fd)
    write_tar_from_workspace(root_fd, preflight_files, sys.stdout.buffer, "w|")


def member_chunks(extracted):
    while True:
        chunk = extracted.read(1 << 16)
        if not chunk:
            return
        yield chunk


def validate_tar_member(member, member_paths, member_parent_paths):
    if not member.isreg():
        fail(
            "invalid_path",
            f"Workspace tar member must be a regular file: {member.name}",
        )
    try:
        parts = tuple(guarded_parts(member.name))
    except GuardPathError:
        fail("invalid_path", "Workspace paths must stay inside the workspace.")
    has_file_ancestor = any(
        parts[:index] in member_paths for index in range(1, len(parts))
    )
    if parts in member_parent_paths or has_file_ancestor:
        fail(
            "invalid_path",
            f"Workspace tar members have conflicting paths: {member.name}",
        )
    member_paths.add(parts)
    member_parent_paths.update(parts[:index] for index in range(1, len(parts)))


def tar_member_git_mode(member):
    if member.uname != SYNC_GIT_MODE_TAR_OWNER:
        return None
    if member.mode not in (0o644, 0o755):
        fail(
            "invalid_path",
            f"Workspace tar member has invalid Git mode authority: {member.name}",
        )
    return member.mode


def preflight_tar_destination(root_fd, rel_path):
    parent_fd = None
    try:
        parent_fd, leaf_name = open_path(root_fd, rel_path)
        inspect_guarded_write_target(leaf_name, parent_fd)
    except GuardPathError as exc:
        if exc.status == "enoent":
            return
        fail_guard(exc, rel_path)
    finally:
        close_fd(parent_fd)


def write_tar_file(
    root_fd,
    fileobj,
    excluded_directory_names,
    excluded_path_regexes,
):
    member_paths = set()
    member_parent_paths = set()
    fileobj.seek(0)
    with tarfile.open(fileobj=fileobj, mode="r") as archive:
        for member in archive:
            validate_tar_member(member, member_paths, member_parent_paths)
            tar_member_git_mode(member)
            if any(
                directory_name_key(part) in excluded_directory_names
                for part in member.name.split("/")
            ):
                fail(
                    "invalid_path",
                    f"Workspace tar member is inside an excluded directory: {member.name}",
                )
            if path_matches_excluded_pattern(member.name, excluded_path_regexes):
                fail(
                    "invalid_path",
                    f"Workspace tar member matches an excluded path pattern: {member.name}",
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                fail(
                    "workspace_error",
                    f"Workspace tar member could not be read: {member.name}",
                )
            try:
                for _ in member_chunks(extracted):
                    pass
            finally:
                extracted.close()
            preflight_tar_destination(root_fd, member.name)

    written_bytes = 0
    written_files = 0
    fileobj.seek(0)
    with tarfile.open(fileobj=fileobj, mode="r") as archive:
        for member in archive:
            extracted = archive.extractfile(member)
            if extracted is None:
                fail(
                    "workspace_error",
                    f"Workspace tar member could not be read: {member.name}",
                )
            parent_fd = None
            try:
                parent_fd, leaf_name = open_path(root_fd, member.name, create=True)
                write_guarded_regular(
                    leaf_name,
                    parent_fd,
                    member_chunks(extracted),
                    mode=tar_member_git_mode(member),
                )
            except GuardPathError as exc:
                fail_guard(exc, member.name)
            finally:
                extracted.close()
                close_fd(parent_fd)
            written_files += 1
            written_bytes += member.size
    print(json.dumps({"ok": True, "files": written_files, "bytes": written_bytes}))


def write_tar_operation(root_fd):
    payload = json.loads(sys.stdin.read())
    data = base64.b64decode(payload["tar_base64"], validate=True)
    excluded_directory_names = frozenset(json.loads(sys.argv[2]))
    excluded_path_regexes = tuple(
        re.compile(pattern) for pattern in json.loads(sys.argv[3])
    )
    write_tar_file(
        root_fd,
        io.BytesIO(data),
        excluded_directory_names,
        excluded_path_regexes,
    )


def write_tar_stream_operation(root_fd):
    excluded_directory_names = frozenset(json.loads(sys.argv[2]))
    excluded_path_regexes = tuple(
        re.compile(pattern) for pattern in json.loads(sys.argv[3])
    )
    expected_bytes = int(sys.argv[4])
    if expected_bytes < 0:
        fail("workspace_error", "Workspace tar byte count must be non-negative.")
    with tempfile.TemporaryFile(mode="w+b", prefix="cayu-workspace-tar-") as staged:
        remaining = expected_bytes
        while remaining:
            chunk = sys.stdin.buffer.read(min(remaining, 1 << 16))
            if not chunk:
                fail("workspace_error", "Workspace tar stream ended before its declared size.")
            staged.write(chunk)
            remaining -= len(chunk)
        if sys.stdin.buffer.read(1):
            fail("workspace_error", "Workspace tar stream exceeded its declared size.")
        write_tar_file(
            root_fd,
            staged,
            excluded_directory_names,
            excluded_path_regexes,
        )


def main():
    global _BINARY_STDOUT
    operation = sys.argv[1]
    _BINARY_STDOUT = operation == "read_tar_stream"
    root_fd = None
    try:
        root_fd = open_guard_root(".", operation in ("list", "git_entries"))
        with workspace_source_lock(root_fd, False):
            if operation == "read":
                read_operation(root_fd)
            elif operation == "write":
                write_operation(root_fd)
            elif operation == "delete":
                delete_operation(root_fd)
            elif operation == "list":
                list_operation(root_fd)
            elif operation == "git_entries":
                git_entries_operation(root_fd)
            elif operation == "read_tar":
                read_tar_operation(root_fd)
            elif operation == "read_tar_stream":
                read_tar_stream_operation(root_fd)
            elif operation == "write_tar":
                write_tar_operation(root_fd)
            elif operation == "write_tar_stream":
                write_tar_stream_operation(root_fd)
            else:
                raise ValueError("Unknown runner workspace operation: " + operation)
    except GuardPathError as exc:
        fail_guard(exc)
    except Exception as exc:
        fail("workspace_error", str(exc))
    finally:
        close_fd(root_fd)


main()
"""
)


class RunnerWorkspace(
    RunnerBoundWorkspace,
    BoundedTarReader,
    TarWriter,
    BoundedTarStreamReader,
    TarStreamWriter,
):
    """Workspace whose descriptor-guarded file operations execute through a runner.

    The configured ``cwd`` is trusted operator input. Each operation pins that
    root and opens every guest-controlled component relative to the preceding
    directory descriptor with ``O_NOFOLLOW``. Within one traversal, an opened
    descriptor authorizes that inode even if another guest process later
    relocates it. Multi-pass operations start each later traversal from the
    pinned root and reject replacement symlinks. Guests without the required
    POSIX descriptor-relative primitives fail closed.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        cwd: str | None = None,
        workspace_id: str | None = None,
        python_executable: str = "python3",
        default_read_limit_bytes: int = DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES,
        default_list_limit: int = DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT,
        excluded_directory_names: Iterable[str] = (),
        excluded_path_patterns: Iterable[str] = (),
        enable_workspace_branches: bool = False,
        branch_operation_timeout_s: int = 300,
        branch_authority_resolver: WorkspaceBranchBindingAuthorityProvider | None = None,
    ) -> None:
        if not isinstance(runner, Runner):
            raise TypeError("RunnerWorkspace runner must be a Runner.")
        self._runner = runner
        self.cwd = _validate_optional_cwd(cwd)
        self.python_executable = require_clean_nonblank(python_executable, "python_executable")
        self.default_read_limit_bytes = _validate_required_limit(
            default_read_limit_bytes,
            "default_read_limit_bytes",
        )
        self.default_list_limit = _validate_required_limit(default_list_limit, "default_list_limit")
        self.excluded_directory_names = _validate_excluded_directory_names(excluded_directory_names)
        self._excluded_directory_keys = frozenset(
            _directory_name_key(value) for value in self.excluded_directory_names
        )
        self.excluded_path_patterns = _validate_excluded_path_patterns(excluded_path_patterns)
        self._excluded_path_pattern_keys = tuple(
            _normalized_exclusion_path(pattern) for pattern in self.excluded_path_patterns
        )
        self._excluded_path_regexes = tuple(
            translate_list_pattern(pattern) for pattern in self._excluded_path_pattern_keys
        )
        if (
            self.excluded_directory_names or self.excluded_path_patterns
        ) and enable_workspace_branches:
            raise ValueError(
                "RunnerWorkspace path exclusions cannot be combined with workspace branches."
            )
        if type(enable_workspace_branches) is not bool:
            raise TypeError("RunnerWorkspace enable_workspace_branches must be a bool.")
        self.branch_operation_timeout_s = _validate_required_limit(
            branch_operation_timeout_s,
            "branch_operation_timeout_s",
        )
        self._branch_capability = (
            runner.workspace_capability(RemoteWorkspaceBranchCapability)
            if enable_workspace_branches
            else None
        )
        if branch_authority_resolver is not None and not isinstance(
            branch_authority_resolver,
            WorkspaceBranchBindingAuthorityProvider,
        ):
            raise TypeError(
                "RunnerWorkspace branch_authority_resolver must own binding-generation claims."
            )
        self._branch_authority_resolver = branch_authority_resolver
        self._branch_lifecycle_registry = _WorkspaceBranchLifecycleRegistry()
        if branch_authority_resolver is None:
            self._branch_claim_scope = None
        else:
            try:
                self._branch_claim_scope = WorkspaceBranchBindingAuthorityClaimScope(
                    branch_authority_resolver.claim_scope
                )
            except Exception:
                raise TypeError(
                    "RunnerWorkspace branch_authority_resolver must declare its claim scope."
                ) from None
        if workspace_id is None:
            self.id = f"runner:{getattr(runner, 'isolation', 'unknown')}:{self.cwd or '.'}"
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        runner = self._runner
        # LocalRunner reads/writes the host filesystem directly, so a RunnerWorkspace over it aliases
        # the same directory a LocalWorkspace addresses. Emit the canonical host-fs key so SyncBinding
        # detects that alias and refuses to clear one view while the other is the source.
        if isinstance(runner, LocalRunner):
            host_dir = runner.root if self.cwd is None else (runner.root / self.cwd).resolve()
            return _local_resource_key(host_dir)
        # Check identity first so an indeterminate runner fails closed (None) without calling resolve_cwd.
        runner_key = _runner_resource_key(runner)
        if runner_key is None:
            return None
        # Key by the runner's resolved absolute working directory so this RunnerWorkspace and the native
        # wrapper (E2BWorkspace / MicrosandboxWorkspace) over the same sandbox directory produce equal keys.
        return ("runner", runner_key, runner.resolve_cwd(self.cwd))

    def is_bound_to_runner(self, runner: Runner) -> bool:
        return self._runner is runner

    def _control_plane_runner(self) -> Runner:
        """Return the runner for Cayu-owned bindings without publishing it to tools."""

        return self._runner

    @property
    def runner_cwd(self) -> str:
        return self._runner.resolve_cwd(self.cwd)

    @property
    def bound_runner_resource_key(self) -> tuple[object, ...] | None:
        return _runner_resource_key(self._runner)

    def bounded_read_limit(self, max_bytes: int) -> int:
        return min(
            self.default_read_limit_bytes,
            _validate_required_limit(max_bytes, "max_bytes"),
        )

    def branch_capabilities(self) -> WorkspaceBranchCapabilities:
        if type(self) is not RunnerWorkspace or self._branch_capability is None:
            return WorkspaceBranchCapabilities()
        durable = (
            isinstance(
                self._branch_authority_resolver,
                RemoteWorkspaceBranchAuthorityProvider,
            )
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
                "durable_remote_runner_workspace_branches"
                if durable
                else "process_local_remote_runner_workspace_branches"
            ),
        )

    def branch_lifecycle_summary(self) -> WorkspaceBranchLifecycleSummary:
        if type(self) is not RunnerWorkspace:
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
        from cayu.workspaces._runner_branch import create_runner_workspace_branch

        result = await create_runner_workspace_branch(self, request)
        self._branch_lifecycle_registry.attach(result.branch)
        return result

    async def recover_branch(
        self,
        request: WorkspaceBranchRecoveryRequest,
    ) -> WorkspaceBranchRecoveryResult:
        from cayu.workspaces._runner_branch import recover_runner_workspace_branch

        result = await recover_runner_workspace_branch(self, request)
        self._branch_lifecycle_registry.attach(result.branch)
        return result

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        offset = _validate_workspace_offset(offset, owner="RunnerWorkspace")
        limit = (
            self.default_read_limit_bytes
            if max_bytes is None
            else _validate_required_limit(max_bytes, "max_bytes")
        )
        content, total_bytes, revision, digest, git_mode = await guard_read(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            rel_path=path,
            offset=offset,
            limit=limit,
            original_path=path,
            backend="Runner",
            python_executable=self.python_executable,
        )
        return WorkspaceReadResult(
            content=content,
            total_bytes=max(total_bytes, offset + len(content)),
            truncated=total_bytes > offset + len(content),
            offset=offset,
            revision=revision,
            sha256=digest,
            git_mode=git_mode,
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        payload = {
            "path": path,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        await self._run_json_operation(
            "write",
            stdin=json.dumps(payload),
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )

    async def delete(self, path: str) -> None:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        await self._run_json_operation(
            "delete",
            path,
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        return await guard_create(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            rel_path=path,
            content=content,
            original_path=path,
            backend="Runner",
            python_executable=self.python_executable,
        )

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        if type(content) is not bytes:
            raise TypeError("Workspace replace content must be bytes.")
        return await guard_replace(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            rel_path=path,
            content=content,
            expected_revision=_validate_workspace_revision(
                expected_revision, owner="RunnerWorkspace"
            ),
            original_path=path,
            backend="Runner",
            python_executable=self.python_executable,
        )

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        return await guard_delete_if_revision(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            rel_path=path,
            expected_revision=_validate_workspace_revision(
                expected_revision, owner="RunnerWorkspace"
            ),
            original_path=path,
            backend="Runner",
            python_executable=self.python_executable,
        )

    async def require_absent(self, path: str) -> None:
        path = _validate_relative_path(path)
        self._require_path_allowed(path)
        await guard_require_absent(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            rel_path=path,
            original_path=path,
            backend="Runner",
            python_executable=self.python_executable,
        )

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
        source = _validate_relative_path(source_path)
        destination = _validate_relative_path(destination_path)
        self._require_path_allowed(source)
        self._require_path_allowed(destination)
        return await guard_move_if_revision(
            self._runner,
            root=self._runner.resolve_cwd(self.cwd),
            source_rel_path=source,
            destination_rel_path=destination,
            expected_revision=_validate_workspace_revision(
                expected_source_revision,
                owner="RunnerWorkspace",
            ),
            original_source_path=source,
            original_destination_path=destination,
            backend="Runner",
            python_executable=self.python_executable,
        )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        effective_limit = (
            self.default_list_limit if limit is None else _validate_required_limit(limit, "limit")
        )
        result = await self._run_json_operation(
            "list",
            translate_list_pattern(pattern),
            str(effective_limit),
            str(RUNNER_WORKSPACE_LIST_PAYLOAD_LIMIT_BYTES),
            json.dumps(tuple(sorted(self._excluded_directory_keys))),
            json.dumps(self._excluded_path_regexes),
            output_limit_bytes=_json_list_output_limit(),
        )
        validated = _validate_workspace_list_result(
            result,
            pattern=pattern,
            effective_limit=effective_limit,
            excluded_directory_keys=self._excluded_directory_keys,
            excluded_path_pattern_keys=self._excluded_path_pattern_keys,
        )
        del result
        if isinstance(validated, Exception):
            raise validated from None
        return validated

    async def list_git_entries(self, *, limit: int) -> WorkspaceGitEntryListResult:
        effective_limit = _validate_required_limit(limit, "limit")
        result = await self._run_json_operation(
            "git_entries",
            str(effective_limit),
            str(RUNNER_WORKSPACE_LIST_PAYLOAD_LIMIT_BYTES),
            json.dumps(tuple(sorted(self._excluded_directory_keys))),
            json.dumps(self._excluded_path_regexes),
            output_limit_bytes=_json_list_output_limit(),
        )
        validated = _validate_workspace_git_entry_result(
            result,
            effective_limit=effective_limit,
            excluded_directory_keys=self._excluded_directory_keys,
            excluded_path_pattern_keys=self._excluded_path_pattern_keys,
        )
        del result
        if isinstance(validated, Exception):
            raise validated from None
        return validated

    def _require_path_allowed(self, path: str) -> None:
        if _path_has_excluded_directory(path, self._excluded_directory_keys):
            raise ValueError("Workspace path is inside an excluded directory.")
        if _path_matches_excluded_pattern(path, self._excluded_path_pattern_keys):
            raise ValueError("Workspace path matches an excluded path pattern.")

    def tar_copy_policy_identity(self) -> tuple[object, ...]:
        return (
            "cayu-runner-workspace-tar-v2",
            tuple(sorted(self._excluded_directory_keys)),
            tuple(sorted(self._excluded_path_pattern_keys)),
        )

    def bounded_tar_stream_reader(self) -> BoundedTarStreamReader | None:
        return self if isinstance(self._runner, RunnerBinaryStreamCapability) else None

    def tar_stream_writer(self) -> TarStreamWriter | None:
        return self if isinstance(self._runner, RunnerBinaryStreamCapability) else None

    async def read_tar_stream(
        self,
        paths: Sequence[str],
        destination: BinaryIO,
        *,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_archive_bytes: int | None = None,
    ) -> TarStreamReadResult:
        """Stream one preflight-bounded uncompressed tar into a private sink."""

        runner = self._runner
        if not isinstance(runner, RunnerBinaryStreamCapability):
            raise RuntimeError("RunnerWorkspace runner does not support binary streams.")
        if isinstance(destination, (str, bytes, bytearray, memoryview)) or not callable(
            getattr(destination, "write", None)
        ):
            raise TypeError("RunnerWorkspace tar destination must be a binary stream.")
        validated_paths = _validate_tar_paths(paths)
        for path in validated_paths:
            self._require_path_allowed(path)
        per_file_limit = (
            self.default_read_limit_bytes
            if max_file_bytes is None
            else _validate_required_limit(max_file_bytes, "max_file_bytes")
        )
        total_limit = _validate_optional_limit(max_total_bytes, "max_total_bytes")
        archive_limit = _validate_optional_limit(max_archive_bytes, "max_archive_bytes")
        archive_overhead_bytes = tar_archive_size_bound(0, validated_paths)
        logical_size_bound = per_file_limit * len(validated_paths)
        if total_limit is not None:
            logical_size_bound = min(logical_size_bound, total_limit)
        raw_size_bound = tar_archive_size_bound(logical_size_bound, validated_paths)
        if archive_limit is not None:
            raw_size_bound = min(raw_size_bound, archive_limit)
        payload = {
            "paths": list(validated_paths),
            "max_file_bytes": per_file_limit,
            "max_total_bytes": total_limit,
            "max_archive_bytes": archive_limit,
            "archive_overhead_bytes": archive_overhead_bytes,
        }
        result = await runner.exec_stream(
            ExecCommand.process(
                self.python_executable,
                "-c",
                _RUNNER_WORKSPACE_PROGRAM,
                "read_tar_stream",
            ),
            cwd=self.cwd,
            stdin=io.BytesIO(json.dumps(payload).encode("utf-8")),
            stdout=destination,
            stdout_limit_bytes=raw_size_bound,
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )
        if result.stdout_truncated:
            raise RuntimeError("Runner workspace tar exceeded its transfer limit.")
        if result.exit_code != 0:
            try:
                error_payload = _parse_json_object(result.stderr)
            except (RuntimeError, TypeError):
                raise RuntimeError(
                    f"Runner workspace operation failed with exit code {result.exit_code}: "
                    f"{result.stderr.strip()}"
                ) from None
            _raise_workspace_error(error_payload)
        archive_bytes = result.stdout_bytes
        if type(archive_bytes) is not int or archive_bytes < 0:
            raise RuntimeError("Runner workspace stream returned invalid byte accounting.")
        return TarStreamReadResult(archive_bytes=archive_bytes)

    async def read_tar_bytes(
        self,
        paths: Sequence[str],
        *,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_archive_bytes: int | None = None,
    ) -> bytes:
        """Read many workspace files in one runner exec as an uncompressed tar.

        This is the bulk-transfer fast path used by SyncBinding: one guest
        process archives every requested file instead of one exec per file.
        Each file is capped at ``max_file_bytes`` (``default_read_limit_bytes``
        when omitted), and their combined logical size is capped at
        ``max_total_bytes`` when provided. An oversized transfer fails before
        the guest allocates the tar buffer. ``max_archive_bytes`` independently
        caps the conservative size of the raw tar, including framing and path
        metadata.
        """

        validated_paths = _validate_tar_paths(paths)
        for path in validated_paths:
            self._require_path_allowed(path)
        per_file_limit = (
            self.default_read_limit_bytes
            if max_file_bytes is None
            else _validate_required_limit(max_file_bytes, "max_file_bytes")
        )
        total_limit = _validate_optional_limit(max_total_bytes, "max_total_bytes")
        archive_limit = _validate_optional_limit(max_archive_bytes, "max_archive_bytes")
        archive_overhead_bytes = tar_archive_size_bound(0, validated_paths)
        payload = {
            "paths": list(validated_paths),
            "max_file_bytes": per_file_limit,
            "max_total_bytes": total_limit,
            "max_archive_bytes": archive_limit,
            "archive_overhead_bytes": archive_overhead_bytes,
        }
        logical_size_bound = per_file_limit * len(validated_paths)
        if total_limit is not None:
            logical_size_bound = min(logical_size_bound, total_limit)
        raw_size_bound = (
            archive_limit
            if archive_limit is not None
            else tar_archive_size_bound(logical_size_bound, validated_paths)
        )
        result = await self._run_json_operation(
            "read_tar",
            stdin=json.dumps(payload),
            output_limit_bytes=_json_read_output_limit(raw_size_bound),
        )
        data = _decode_base64(result["tar_base64"], "tar_base64")
        if archive_limit is not None and len(data) > archive_limit:
            raise RuntimeError(f"Runner workspace tar exceeds max_archive_bytes={archive_limit}.")
        return data

    async def write_tar_bytes(self, data: bytes) -> None:
        """Write many workspace files in one runner exec from an uncompressed tar.

        Members must be regular files with workspace-relative paths; symlink,
        absolute, and ``..`` members are rejected inside the guest. Every
        member and its content is validated before the first file mutation.
        """

        if type(data) is not bytes:
            raise TypeError("Workspace tar content must be bytes.")
        payload = {"tar_base64": base64.b64encode(data).decode("ascii")}
        await self._run_json_operation(
            "write_tar",
            json.dumps(tuple(sorted(self._excluded_directory_keys))),
            json.dumps(self._excluded_path_regexes),
            stdin=json.dumps(payload),
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )

    async def write_tar_stream(self, source: BinaryIO, *, archive_bytes: int) -> None:
        """Stream a tar through runner stdin and spool it privately in the guest."""

        runner = self._runner
        if not isinstance(runner, RunnerBinaryStreamCapability):
            raise RuntimeError("RunnerWorkspace runner does not support binary streams.")
        if isinstance(source, (str, bytes, bytearray, memoryview)) or not callable(
            getattr(source, "read", None)
        ):
            raise TypeError("RunnerWorkspace tar source must be a binary stream.")
        if type(archive_bytes) is not int:
            raise TypeError("RunnerWorkspace archive_bytes must be an integer.")
        if archive_bytes < 0:
            raise ValueError("RunnerWorkspace archive_bytes must be non-negative.")
        result = await runner.exec_stream(
            ExecCommand.process(
                self.python_executable,
                "-c",
                _RUNNER_WORKSPACE_PROGRAM,
                "write_tar_stream",
                json.dumps(tuple(sorted(self._excluded_directory_keys))),
                json.dumps(self._excluded_path_regexes),
                str(archive_bytes),
            ),
            cwd=self.cwd,
            stdin=source,
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )
        if result.stdout_truncated:
            raise RuntimeError("Runner workspace operation output exceeded its transfer limit.")
        try:
            payload = _parse_json_object(result.stdout)
        except RuntimeError:
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Runner workspace operation failed with exit code {result.exit_code}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                ) from None
            raise
        if payload.get("ok") is not True:
            _raise_workspace_error(payload)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Runner workspace operation failed with exit code {result.exit_code}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    async def _run_json_operation(
        self,
        operation: str,
        *args: str,
        stdin: str | None = None,
        output_limit_bytes: int,
    ) -> dict[str, Any]:
        exec_result = await self._runner.exec(
            ExecCommand.process(
                self.python_executable,
                "-c",
                _RUNNER_WORKSPACE_PROGRAM,
                operation,
                *args,
            ),
            cwd=self.cwd,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        if exec_result.stdout_truncated:
            raise RuntimeError("Runner workspace operation output exceeded its transfer limit.")
        try:
            payload = _parse_json_object(exec_result.stdout)
        except RuntimeError:
            if exec_result.exit_code != 0:
                raise RuntimeError(
                    f"Runner workspace operation failed with exit code {exec_result.exit_code}: "
                    f"{exec_result.stderr.strip() or exec_result.stdout.strip()}"
                ) from None
            raise
        if payload.get("ok") is not True:
            _raise_workspace_error(payload)
        if exec_result.exit_code != 0:
            raise RuntimeError(
                f"Runner workspace operation failed with exit code {exec_result.exit_code}: "
                f"{exec_result.stderr.strip() or exec_result.stdout.strip()}"
            )
        return payload


def _validate_optional_cwd(cwd: str | None) -> str | None:
    if cwd is None:
        return None
    value = require_nonblank(cwd, "cwd")
    if posixpath.isabs(value):
        raise ValueError("RunnerWorkspace cwd must be relative to the runner root.")
    normalized = posixpath.normpath(value)
    if normalized == ".":
        return None
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("RunnerWorkspace cwd escapes the runner root.")
    return normalized


def _validate_relative_path(path: str) -> str:
    return _validate_workspace_relative_path(path)


def _validate_workspace_list_result(
    result: dict[str, Any],
    *,
    pattern: str,
    effective_limit: int,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> WorkspaceListResult | TypeError | ValueError:
    paths = result.get("paths")
    total_count = result.get("total_count")
    if type(paths) is not list:
        return TypeError("Runner workspace list returned invalid paths.")
    if type(total_count) is not int:
        return TypeError("Runner workspace list returned invalid total_count.")

    validated_paths: list[str] = []
    for path in paths:
        if type(path) is not str:
            return TypeError("Runner workspace list returned a non-string path.")
        try:
            path = require_durable_text(path, "Runner workspace list path")
            normalized = _validate_relative_path(path)
        except (TypeError, ValueError):
            return ValueError("Runner workspace list returned an invalid path.")
        if path != normalized:
            return ValueError("Runner workspace list returned a non-normalized path.")
        if _path_has_excluded_directory(path, excluded_directory_keys):
            return ValueError("Runner workspace list returned an excluded path.")
        if _path_matches_excluded_pattern(path, excluded_path_pattern_keys):
            return ValueError("Runner workspace list returned an excluded path.")
        validated_paths.append(path)

    if len(set(validated_paths)) != len(validated_paths):
        return ValueError("Runner workspace list returned duplicate paths.")
    if validated_paths != sorted(validated_paths):
        return ValueError("Runner workspace list returned paths in non-deterministic order.")
    if any(not matches_list_pattern(path, pattern) for path in validated_paths):
        return ValueError("Runner workspace list returned a path outside the requested pattern.")
    if total_count < 0:
        return ValueError("Runner workspace list returned a negative total_count.")
    if total_count > MAX_DURABLE_JSON_INTEGER:
        return ValueError("Runner workspace list returned an oversized total_count.")
    if total_count < len(validated_paths):
        return ValueError("Runner workspace list total_count is smaller than its paths.")
    if len(validated_paths) > effective_limit:
        return ValueError("Runner workspace list returned more paths than the effective limit.")

    return WorkspaceListResult(
        paths=tuple(validated_paths),
        total_count=total_count,
        truncated=total_count > len(validated_paths),
    )


def _validate_workspace_git_entry_result(
    result: dict[str, Any],
    *,
    effective_limit: int,
    excluded_directory_keys: frozenset[str],
    excluded_path_pattern_keys: tuple[str, ...],
) -> WorkspaceGitEntryListResult | TypeError | ValueError:
    raw_entries = result.get("entries")
    total_count = result.get("total_count")
    truncated = result.get("truncated")
    if type(raw_entries) is not list or type(total_count) is not int or type(truncated) is not bool:
        return TypeError("Runner workspace returned invalid Git entry evidence.")
    entries: list[WorkspaceGitEntry] = []
    for raw_entry in raw_entries:
        if type(raw_entry) is not dict or set(raw_entry) != {
            "path",
            "git_mode",
            "symlink_target_sha256",
            "symlink_target_bytes",
        }:
            return TypeError("Runner workspace returned an invalid Git entry.")
        path = raw_entry.get("path")
        if type(path) is not str:
            return TypeError("Runner workspace returned an invalid Git entry path.")
        try:
            path = require_durable_text(path, "Runner workspace Git entry path")
            normalized = _validate_relative_path(path)
            entry = WorkspaceGitEntry(
                path=path,
                git_mode=raw_entry.get("git_mode"),
                symlink_target_sha256=raw_entry.get("symlink_target_sha256"),
                symlink_target_bytes=raw_entry.get("symlink_target_bytes"),
            )
        except (TypeError, ValueError):
            return ValueError("Runner workspace returned invalid Git entry evidence.")
        if (
            path != normalized
            or _path_has_excluded_directory(path, excluded_directory_keys)
            or _path_matches_excluded_pattern(path, excluded_path_pattern_keys)
        ):
            return ValueError("Runner workspace returned an inadmissible Git entry path.")
        entries.append(entry)
    if total_count < len(entries) or total_count > effective_limit + 1:
        return ValueError("Runner workspace returned an invalid Git entry count.")
    if len(entries) > effective_limit or (not truncated and total_count != len(entries)):
        return ValueError("Runner workspace returned inconsistent Git entry truncation.")
    try:
        return WorkspaceGitEntryListResult(
            entries=tuple(entries),
            total_count=total_count,
            truncated=truncated,
        )
    except (TypeError, ValueError) as exc:
        return type(exc)("Runner workspace returned invalid Git entry evidence.")


def _validate_required_limit(value: int, field_name: str) -> int:
    return _validate_workspace_positive_limit(value, field_name, owner="RunnerWorkspace")


def _validate_optional_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_required_limit(value, field_name)


def _validate_excluded_directory_names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("RunnerWorkspace excluded_directory_names must be an iterable of strings.")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "RunnerWorkspace excluded_directory_names must be an iterable of strings."
        ) from exc
    normalized: dict[str, str] = {}
    for index, name in enumerate(names):
        value = require_clean_nonblank(name, f"excluded_directory_names[{index}]")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(
                "RunnerWorkspace excluded directory names must be single path segments."
            )
        key = _directory_name_key(value)
        if key in normalized:
            raise ValueError(
                "RunnerWorkspace excluded directory names must be case-insensitively unique."
            )
        normalized[key] = value
    return tuple(normalized[key] for key in sorted(normalized))


def _validate_excluded_path_patterns(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("RunnerWorkspace excluded_path_patterns must be an iterable of strings.")
    try:
        patterns = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "RunnerWorkspace excluded_path_patterns must be an iterable of strings."
        ) from exc
    normalized: dict[str, str] = {}
    for index, pattern in enumerate(patterns):
        value = require_clean_nonblank(pattern, f"excluded_path_patterns[{index}]")
        require_durable_text(value, f"excluded_path_patterns[{index}]")
        if "\\" in value:
            raise ValueError("RunnerWorkspace excluded path patterns must use POSIX separators.")
        value = validate_list_pattern(value)
        key = _normalized_exclusion_path(value)
        if key in normalized:
            raise ValueError(
                "RunnerWorkspace excluded path patterns must be case-insensitively unique."
            )
        normalized[key] = value
    return tuple(normalized[key] for key in sorted(normalized))


def _path_has_excluded_directory(path: str, excluded_directory_keys: frozenset[str]) -> bool:
    return any(_directory_name_key(part) in excluded_directory_keys for part in path.split("/"))


def _directory_name_key(value: str) -> str:
    return value.rstrip(" .").casefold()


def _normalized_exclusion_path(value: str) -> str:
    return "/".join(
        part.rstrip(" .").casefold()
        for part in value.replace("\\", "/").split("/")
        if part not in {"", "."}
    )


def _path_matches_excluded_pattern(
    path: str,
    excluded_path_pattern_keys: tuple[str, ...],
) -> bool:
    normalized_parts = tuple(part for part in _normalized_exclusion_path(path).split("/") if part)
    return any(
        matches_list_pattern("/".join(normalized_parts[:end]), pattern)
        for pattern in excluded_path_pattern_keys
        for end in range(1, len(normalized_parts) + 1)
    )


def _validate_tar_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, str) or not isinstance(paths, Sequence):
        raise TypeError("RunnerWorkspace read_tar_bytes paths must be a sequence of strings.")
    if not paths:
        raise ValueError("RunnerWorkspace read_tar_bytes requires at least one path.")
    return tuple(_validate_relative_path(path) for path in paths)


def _json_read_output_limit(max_bytes: int) -> int:
    return (4 * ((max_bytes + 2) // 3)) + RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES


def _json_list_output_limit() -> int:
    return RUNNER_WORKSPACE_LIST_PAYLOAD_LIMIT_BYTES + RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES


def _decode_base64(value: Any, field_name: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"Runner workspace returned invalid {field_name}.")
    return base64.b64decode(value.encode("ascii"), validate=True)


def _parse_json_object(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Runner workspace operation returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise TypeError("Runner workspace operation returned invalid JSON object.")
    return payload


def _raise_workspace_error(payload: dict[str, Any]) -> None:
    error_type = payload.get("error_type")
    message = payload.get("message")
    if type(message) is not str or not message:
        message = "Runner workspace operation failed."
    if error_type == "not_found":
        raise FileNotFoundError(message)
    if error_type in {"invalid_path", "invalid_pattern"}:
        raise ValueError(message)
    if error_type == "not_file":
        raise IsADirectoryError(message)
    raise RuntimeError(message)

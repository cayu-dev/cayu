"""Guest program for durable branches on an explicitly capable remote runner.

The program keeps raw baseline and overlay bytes inside the retained allocation.
Only bounded identities and lifecycle codes cross the runner boundary.
"""

from __future__ import annotations

from cayu.workspaces._guest_guard import GUEST_DESCRIPTOR_GUARD_SOURCE

RUNNER_WORKSPACE_BRANCH_PROGRAM = (
    GUEST_DESCRIPTOR_GUARD_SOURCE
    + r"""
import base64
import errno
import json
import re
import shutil
import sys
import time


PRIVATE_PREFIX = ".cayu-workspace-branch-"
CREATION_CLAIM_PENDING_SUFFIX = ".pending"
RECORD_NAME = "record.json"
TERMINAL_STATES = {"committed", "rolled_back", "expired", "failed", "ambiguous"}
OPEN_STATES = {"open", "conflicted"}
FAILURE_OUTPUT_ENABLED = True

# Match RunnerWorkspace's ordinary guest creation policy. Publication freezes
# the effective modes in its durable intent so a fresh process with a different
# umask cannot change the source outcome during recovery.
GUARDED_DIRECTORY_CREATE_MODE = 0o777
GUARDED_FILE_CREATE_MODE = 0o666


def require_branch_guard_support():
    if (
        not hasattr(os, "fchmod")
        or not hasattr(os, "fchdir")
        or os.rmdir not in os.supports_dir_fd
        or not getattr(shutil.rmtree, "avoids_symlink_attacks", False)
    ):
        fail("unsupported", "workspace_branch_guest_guard_unsupported")


def close_fd(fd):
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def read_all_bounded(fd, max_bytes):
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = int(max_bytes) + 1
    chunks = []
    while remaining > 0:
        chunk = os.read(fd, min(1 << 16, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def fail(error_type, detail_code):
    if FAILURE_OUTPUT_ENABLED:
        emit({"ok": False, "error_type": error_type, "detail_code": detail_code})
    raise SystemExit(1)


def identity(content):
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def stored_identity(content, mode):
    value = identity(content)
    value["mode"] = stat.S_IMODE(mode)
    return value


def public_identity(value):
    if value is None:
        return None
    return {"sha256": value["sha256"], "bytes": value["bytes"]}


def revision(content):
    return "sha256:" + hashlib.sha256(content).hexdigest()


def same_identity(first, second):
    return (
        first is not None
        and second is not None
        and first.get("sha256") == second.get("sha256")
        and first.get("bytes") == second.get("bytes")
    )


def same_source_identity(first, second):
    return (
        same_identity(first, second)
        and first.get("mode") == second.get("mode")
    )


def effective_creation_mode(configured):
    current = os.umask(0)
    os.umask(current)
    return int(configured) & ~current


def record_digest(record):
    material = dict(record)
    material.pop("integrity", None)
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sealed_record(record):
    material = dict(record)
    material.pop("integrity", None)
    material["integrity"] = record_digest(material)
    return material


def private_path(root_fd, branch_id):
    root_info = os.fstat(root_fd)
    token = hashlib.sha256(
        f"{root_info.st_dev}:{root_info.st_ino}\0{branch_id}".encode("utf-8")
    ).hexdigest()
    # Creation and every later operation use one permanent, claim-bound name.
    # A top-level directory rename or removal cannot be made conditional on an
    # already-open directory inode on all supported guests, so the internal
    # ``.stage`` suffix is retained for the full branch lifecycle.
    return os.path.join(os.path.abspath(".."), PRIVATE_PREFIX + token + ".stage")


def require_private_directory(path, root_fd):
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        fail("not_found", "workspace_branch_not_found")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        fail("fenced", "workspace_branch_private_root_invalid")
    if info.st_dev != os.fstat(root_fd).st_dev:
        fail("fenced", "workspace_branch_private_filesystem_changed")
    return info


def atomic_file_write(path, content, mode=0o600, temporary=None):
    parent = os.path.dirname(path)
    if temporary is None:
        temporary = os.path.join(parent, ".cayu-stage-" + os.urandom(16).hex())
    elif os.path.dirname(temporary) != parent:
        fail("fenced", "workspace_branch_private_temporary_invalid")
    fd = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        write_all(fd, content)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        directory_fd = os.open(parent, OPEN_BASE_FLAGS | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def durable_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    parent_fd = os.open(os.path.dirname(path), OPEN_BASE_FLAGS | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def write_record(private, record):
    record = dict(record)
    record["revision"] = int(record.get("revision", 0)) + 1
    record = sealed_record(record)
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    limit = min(
        16 * 1024 * 1024,
        max(64 * 1024, int(record["limits"]["max_evidence_bytes"]) * 8),
    )
    if len(encoded) > limit:
        fail("resource_exhausted", "branch_record_limit_exceeded")
    atomic_file_write(os.path.join(private, RECORD_NAME), encoded)
    return record


def load_record(private, root_fd):
    private_info = require_private_directory(private, root_fd)
    path = os.path.join(private, RECORD_NAME)
    try:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            fail("fenced", "workspace_branch_record_invalid")
        with open(path, "rb") as file:
            encoded = file.read(16 * 1024 * 1024 + 1)
    except FileNotFoundError:
        fail("not_found", "workspace_branch_record_missing")
    if len(encoded) > 16 * 1024 * 1024:
        fail("fenced", "workspace_branch_record_oversized")
    try:
        record = json.loads(encoded)
    except Exception:
        fail("fenced", "workspace_branch_record_corrupt")
    if not isinstance(record, dict) or record.get("integrity") != record_digest(record):
        fail("fenced", "workspace_branch_record_integrity_failed")
    root_info = os.fstat(root_fd)
    if record.get("source_root") != [root_info.st_dev, root_info.st_ino]:
        fail("fenced", "workspace_branch_source_root_changed")
    if record.get("private_root") != [private_info.st_dev, private_info.st_ino]:
        fail("fenced", "workspace_branch_private_root_changed")
    return record


def creation_claim_path(staging):
    return staging + ".claim"


def creation_bound_claim_path(path):
    return path + ".bound"


def creation_claim(root_fd, payload):
    root_info = os.fstat(root_fd)
    now_ms = int(time.time() * 1000)
    return sealed_record(
        {
            "schema": 2,
            "source_root": [root_info.st_dev, root_info.st_ino],
            "branch_id": payload.get("branch_id"),
            "source": payload.get("source"),
            "baseline_revision": payload.get("baseline_revision"),
            "allocation_fingerprint": payload.get("allocation_fingerprint"),
            "idempotency_key": payload.get("idempotency_key"),
            "authority": payload.get("authority"),
            "limits": payload.get("limits"),
            "cleanup_owner": os.urandom(16).hex(),
            "staging_root": None,
            "created_at_ms": now_ms,
            "expires_at_ms": now_ms + int(payload["limits"]["lifetime_ms"]),
        }
    )


def valid_creation_claim(claim, root_fd):
    root_info = os.fstat(root_fd)
    return (
        isinstance(claim, dict)
        and claim.get("integrity") == record_digest(claim)
        and claim.get("schema") == 2
        and claim.get("source_root") == [root_info.st_dev, root_info.st_ino]
        and isinstance(claim.get("cleanup_owner"), str)
        and re.fullmatch(r"[0-9a-f]{32}", claim["cleanup_owner"]) is not None
        and (
            claim.get("staging_root") is None
            or (
                isinstance(claim["staging_root"], list)
                and len(claim["staging_root"]) == 2
                and all(type(value) is int for value in claim["staging_root"])
            )
        )
    )


def creation_claim_pending_path(path):
    return path + CREATION_CLAIM_PENDING_SUFFIX


def load_creation_claim_file(path, root_fd, *, settle_linked=True):
    fd = None
    expected_links = 2 if settle_linked else 1
    try:
        info = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != expected_links
        ):
            fail("fenced", "workspace_branch_creation_claim_invalid")
        if info.st_size > 1024 * 1024:
            fail("fenced", "workspace_branch_creation_claim_invalid")
        fd = os.open(
            path,
            OPEN_BASE_FLAGS | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_info = os.fstat(fd)
        if (
            opened_info.st_dev != info.st_dev
            or opened_info.st_ino != info.st_ino
            or not stat.S_ISREG(opened_info.st_mode)
            or opened_info.st_nlink != expected_links
        ):
            fail("fenced", "workspace_branch_creation_claim_invalid")
        encoded = read_all_bounded(fd, 1024 * 1024)
    except FileNotFoundError:
        return None
    finally:
        if fd is not None:
            os.close(fd)
    try:
        claim = json.loads(encoded)
    except Exception:
        fail("fenced", "workspace_branch_creation_claim_invalid")
    if not valid_creation_claim(claim, root_fd):
        fail("fenced", "workspace_branch_creation_claim_invalid")
    if settle_linked:
        verify_creation_claim_publication(path, opened_info)
    return claim


def verify_creation_claim_publication(path, claim_info):
    pending = creation_claim_pending_path(path)
    pending_fd = None
    try:
        pending_fd = os.open(
            pending,
            OPEN_BASE_FLAGS | getattr(os, "O_NOFOLLOW", 0),
        )
        pending_info = os.fstat(pending_fd)
        if (
            pending_info.st_dev != claim_info.st_dev
            or pending_info.st_ino != claim_info.st_ino
        ):
            fail("fenced", "workspace_branch_creation_claim_invalid")
    finally:
        close_fd(pending_fd)
    fsync_parent_directory(path)
    try:
        published = os.stat(path, follow_symlinks=False)
        retained_pending = os.stat(pending, follow_symlinks=False)
    except FileNotFoundError:
        fail("fenced", "workspace_branch_creation_claim_invalid")
    if (
        published.st_dev != claim_info.st_dev
        or published.st_ino != claim_info.st_ino
        or published.st_nlink != 2
        or retained_pending.st_dev != claim_info.st_dev
        or retained_pending.st_ino != claim_info.st_ino
        or retained_pending.st_nlink != 2
    ):
        fail("fenced", "workspace_branch_creation_claim_invalid")


def load_creation_claim(path, root_fd):
    claim = load_creation_claim_file(path, root_fd)
    if claim is None:
        pending_path = creation_claim_pending_path(path)
        if os.path.lexists(pending_path):
            pending = load_creation_claim_file(
                pending_path,
                root_fd,
                settle_linked=False,
            )
            if pending is None:
                fail("fenced", "workspace_branch_creation_claim_invalid")
            claim = store_creation_claim(path, pending, root_fd)
        elif os.path.lexists(creation_bound_claim_path(path)):
            fail("fenced", "workspace_branch_creation_claim_invalid")
        else:
            return None
    bound_path = creation_bound_claim_path(path)
    bound = load_creation_claim_file(bound_path, root_fd)
    if bound is None and os.path.lexists(creation_claim_pending_path(bound_path)):
        pending_bound = load_creation_claim_file(
            creation_claim_pending_path(bound_path),
            root_fd,
            settle_linked=False,
        )
        if pending_bound is None:
            fail("fenced", "workspace_branch_creation_claim_invalid")
        bound = store_creation_claim(bound_path, pending_bound, root_fd)
    if bound is None:
        return claim
    if bound.get("staging_root") is None or any(
        bound.get(name) != claim.get(name)
        for name in (
            "schema",
            "source_root",
            "branch_id",
            "source",
            "baseline_revision",
            "allocation_fingerprint",
            "idempotency_key",
            "authority",
            "limits",
            "cleanup_owner",
            "created_at_ms",
            "expires_at_ms",
        )
    ):
        fail("fenced", "workspace_branch_creation_claim_invalid")
    return bound


def exact_creation_claim(claim, payload):
    return claim is not None and all(
        claim.get(name) == payload.get(name)
        for name in (
            "branch_id",
            "source",
            "baseline_revision",
            "allocation_fingerprint",
            "idempotency_key",
            "authority",
            "limits",
        )
    )


def same_pending_creation_claim(existing, requested):
    return (
        exact_creation_claim(existing, requested)
        and existing.get("schema") == requested.get("schema")
        and existing.get("source_root") == requested.get("source_root")
        and existing.get("staging_root") == requested.get("staging_root")
        and (
            requested.get("staging_root") is None
            or all(
                existing.get(name) == requested.get(name)
                for name in (
                    "cleanup_owner",
                    "created_at_ms",
                    "expires_at_ms",
                )
            )
        )
    )


def store_creation_claim(path, claim, root_fd):
    pending = creation_claim_pending_path(path)
    fd = None
    pending_info = None
    try:
        if os.path.lexists(pending):
            existing = load_creation_claim_file(
                pending,
                root_fd,
                settle_linked=False,
            )
            if existing is None or not same_pending_creation_claim(existing, claim):
                fail("fenced", "workspace_branch_creation_claim_changed")
            claim = existing
        else:
            encoded = json.dumps(
                claim,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > 1024 * 1024:
                fail("resource_exhausted", "workspace_branch_creation_claim_limit_exceeded")
            fd = os.open(
                pending,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            # CAYU_TEST_AFTER_CREATION_CLAIM_TEMPORARY_OPEN
            write_all(fd, encoded)
            os.fsync(fd)
            os.close(fd)
            fd = None
            # CAYU_TEST_AFTER_CREATION_CLAIM_TEMPORARY_SYNC
        pending_info = os.stat(pending, follow_symlinks=False)
        if (
            not stat.S_ISREG(pending_info.st_mode)
            or stat.S_ISLNK(pending_info.st_mode)
            or pending_info.st_nlink != 1
        ):
            fail("fenced", "workspace_branch_creation_claim_changed")
        os.link(pending, path, follow_symlinks=False)
        # CAYU_TEST_AFTER_CREATION_CLAIM_LINK
        fsync_parent_directory(path)
    except FileExistsError:
        fail("fenced", "workspace_branch_creation_claim_changed")
    finally:
        close_fd(fd)
    published = os.stat(path, follow_symlinks=False)
    if (
        pending_info is None
        or published.st_dev != pending_info.st_dev
        or published.st_ino != pending_info.st_ino
        or published.st_nlink != 2
    ):
        fail("fenced", "workspace_branch_creation_claim_changed")
    verify_creation_claim_publication(path, published)
    return claim


def write_creation_claim(path, root_fd, payload):
    return store_creation_claim(path, creation_claim(root_fd, payload), root_fd)


def bind_creation_claim(path, claim, staging_info, root_fd):
    bound = dict(claim)
    bound["staging_root"] = [staging_info.st_dev, staging_info.st_ino]
    bound = sealed_record(bound)
    bound_path = creation_bound_claim_path(path)
    if os.path.lexists(bound_path):
        existing = load_creation_claim_file(bound_path, root_fd)
        if existing != bound:
            fail("fenced", "workspace_branch_creation_claim_changed")
        return existing
    return store_creation_claim(bound_path, bound, root_fd)


def fsync_parent_directory(path):
    parent_fd = os.open(os.path.dirname(path), OPEN_BASE_FLAGS | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def require_claimed_creation_directory(path, claim, root_fd):
    info = require_private_directory(path, root_fd)
    if claim.get("staging_root") != [info.st_dev, info.st_ino]:
        fail("fenced", "workspace_branch_creation_staging_changed")
    return info


def same_directory_identity(left, right):
    return (
        stat.S_ISDIR(left.st_mode)
        and not stat.S_ISLNK(left.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
    )


@contextlib.contextmanager
def branch_lock(private, root_fd):
    cwd_fd = None
    private_fd = None
    locked = False
    try:
        cwd_fd = os.open(".", OPEN_BASE_FLAGS | os.O_DIRECTORY)
        try:
            private_fd = os.open(
                private,
                OPEN_BASE_FLAGS | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except FileNotFoundError:
            fail("not_found", "workspace_branch_not_found")
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                fail("fenced", "workspace_branch_private_root_invalid")
            raise
        private_info = os.fstat(private_fd)
        if not stat.S_ISDIR(private_info.st_mode):
            fail("fenced", "workspace_branch_private_root_invalid")
        if private_info.st_dev != os.fstat(root_fd).st_dev:
            fail("fenced", "workspace_branch_private_filesystem_changed")
        # Lock the directory inode itself. A replaceable child lock file could
        # split cooperating operations across two independent locks.
        fcntl.flock(private_fd, fcntl.LOCK_EX)
        locked = True
        # Every private-path helper below receives this pinned current-directory
        # alias. Renaming or replacing the canonical pathname can no longer
        # redirect record, blob, temporary, or cleanup operations.
        os.fchdir(private_fd)
        yield "."
    finally:
        try:
            if cwd_fd is not None:
                os.fchdir(cwd_fd)
        finally:
            if cwd_fd is not None:
                os.close(cwd_fd)
            if private_fd is not None:
                if locked:
                    fcntl.flock(private_fd, fcntl.LOCK_UN)
                os.close(private_fd)


def cleanup_pinned_directory(private):
    for name in sorted(os.listdir(private)):
        path = os.path.join(private, name)
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            shutil.rmtree(path)
            continue
        if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1:
            os.unlink(path)
            continue
        fail("fenced", "workspace_branch_creation_cleanup_invalid")
    directory_fd = os.open(private, OPEN_BASE_FLAGS | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def canonical_path(path):
    try:
        parts = guarded_parts(path)
    except GuardPathError:
        fail("invalid_path", "workspace_branch_path_invalid")
    normalized = "/".join(parts)
    if normalized != path:
        fail("invalid_path", "workspace_branch_path_noncanonical")
    return normalized


def require_bounded_operation_path(record, path):
    if len(path.encode("utf-8")) > int(record["limits"]["max_path_bytes"]):
        fail("resource_exhausted", "path_byte_limit_exceeded")


def private_blob_path(private, area, rel_path):
    return os.path.join(private, area, *canonical_path(rel_path).split("/"))


def ensure_private_parents(path, private):
    parent = os.path.dirname(path)
    relative = os.path.relpath(parent, private)
    current = private
    for part in (() if relative == "." else relative.split(os.sep)):
        current = os.path.join(current, part)
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            info = os.stat(current, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                fail("fenced", "workspace_branch_private_path_invalid")


def write_private_blob(private, area, rel_path, content):
    path = private_blob_path(private, area, rel_path)
    ensure_private_parents(path, private)
    atomic_file_write(path, content)


def read_private_blob(private, area, rel_path, expected):
    path = private_blob_path(private, area, rel_path)
    try:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            fail("fenced", "workspace_branch_private_content_invalid")
        with open(path, "rb") as file:
            content = file.read(int(expected["bytes"]) + 1)
    except FileNotFoundError:
        fail("fenced", "workspace_branch_private_content_missing")
    if not same_identity(identity(content), expected):
        fail("fenced", "workspace_branch_private_content_changed")
    return content


def remove_private_blob(private, area, rel_path):
    path = private_blob_path(private, area, rel_path)
    try:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            fail("fenced", "workspace_branch_private_content_invalid")
        os.unlink(path)
    except FileNotFoundError:
        pass
    area_root = os.path.join(private, area)
    current = os.path.dirname(path)
    while current != area_root:
        try:
            os.rmdir(current)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                break
            raise
        current = os.path.dirname(current)


def cleanup_private_content(private):
    for name in ("baseline", "overlay"):
        path = os.path.join(private, name)
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            fail("fenced", "workspace_branch_private_cleanup_invalid")
        shutil.rmtree(path)


def verify_private_content(private, record):
    directories = record.get("baseline_directories")
    if (
        not isinstance(directories, list)
        or any(not isinstance(path, str) for path in directories)
        or directories != sorted(set(directories))
    ):
        fail("fenced", "workspace_branch_private_index_invalid")
    for path in directories:
        canonical_path(path)
    for area in ("baseline", "overlay"):
        entries = record.get(area)
        if not isinstance(entries, dict):
            fail("fenced", "workspace_branch_private_index_invalid")
        for path, expected in sorted(entries.items()):
            if not isinstance(expected, dict):
                fail("fenced", "workspace_branch_private_identity_invalid")
            read_private_blob(private, area, path, expected)
        area_root = os.path.join(private, area)
        expected_paths = set(entries)
        try:
            area_info = os.stat(area_root, follow_symlinks=False)
        except FileNotFoundError:
            if expected_paths:
                fail("fenced", "workspace_branch_private_content_missing")
            continue
        if not stat.S_ISDIR(area_info.st_mode) or stat.S_ISLNK(area_info.st_mode):
            fail("fenced", "workspace_branch_private_content_invalid")
        observed_paths = set()
        for current, directory_names, file_names in os.walk(area_root, followlinks=False):
            for name in directory_names:
                info = os.stat(os.path.join(current, name), follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    fail("fenced", "workspace_branch_private_content_invalid")
            for name in file_names:
                absolute = os.path.join(current, name)
                info = os.stat(absolute, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
                    fail("fenced", "workspace_branch_private_content_invalid")
                observed_paths.add(os.path.relpath(absolute, area_root).replace(os.sep, "/"))
        if observed_paths != expected_paths:
            fail("fenced", "workspace_branch_private_content_unindexed")


def collect_source_files(root_fd, limits):
    files = {}
    modes = {}
    directories = []
    total_bytes = 0
    path_count = 0

    def visit(directory_fd, prefix, ancestors):
        nonlocal total_bytes, path_count
        info = os.fstat(directory_fd)
        directory_identity = (info.st_dev, info.st_ino)
        if directory_identity in ancestors:
            raise GuardPathError("escape")
        ancestors.add(directory_identity)
        try:
            for name in sorted(os.listdir(directory_fd)):
                rel_path = name if not prefix else prefix + "/" + name
                path_count += 1
                if path_count > int(limits["max_paths"]):
                    fail("resource_exhausted", "path_count_limit_exceeded")
                if len(rel_path.encode("utf-8")) > int(limits["max_path_bytes"]):
                    fail("resource_exhausted", "path_byte_limit_exceeded")
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                # CAYU_TEST_AFTER_SOURCE_FILE_STAT
                if stat.S_ISLNK(entry.st_mode):
                    fail("invalid_path", "workspace_branch_source_symlink")
                if stat.S_ISDIR(entry.st_mode):
                    directories.append(rel_path)
                    child_fd = open_guarded_directory(name, directory_fd, False, True)
                    try:
                        visit(child_fd, rel_path, ancestors)
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
                    fail("invalid_path", "workspace_branch_source_special_file")
                if len(files) >= int(limits["max_files"]):
                    fail("resource_exhausted", "file_count_limit_exceeded")
                if entry.st_size > int(limits["max_file_bytes"]):
                    fail("resource_exhausted", "file_byte_limit_exceeded")
                leaf_fd, opened = open_guarded_regular(name, directory_fd)
                try:
                    remaining_total = int(limits["max_baseline_bytes"]) - total_bytes
                    content = read_all_bounded(
                        leaf_fd,
                        min(int(limits["max_file_bytes"]), max(0, remaining_total)),
                    )
                finally:
                    os.close(leaf_fd)
                if opened.st_dev != entry.st_dev or opened.st_ino != entry.st_ino:
                    fail("conflicted", "workspace_branch_source_changed_during_capture")
                if len(content) > int(limits["max_file_bytes"]):
                    fail("resource_exhausted", "file_byte_limit_exceeded")
                total_bytes += len(content)
                if total_bytes > int(limits["max_baseline_bytes"]):
                    fail("resource_exhausted", "baseline_byte_limit_exceeded")
                files[rel_path] = content
                modes[rel_path] = stat.S_IMODE(opened.st_mode)
        finally:
            ancestors.remove(directory_identity)

    visit(root_fd, "", set())
    return files, modes, directories


def baseline_conflicts(expected, files):
    conflicts = []
    all_paths = sorted(set(expected) | set(files))
    for path in all_paths:
        wanted = expected.get(path)
        actual = None if path not in files else identity(files[path])
        if wanted is None or actual is None or wanted.get("sha256") != actual["sha256"]:
            conflicts.append(
                {
                    "path": path,
                    "expected": (
                        wanted
                        if isinstance(wanted, dict) and "bytes" in wanted
                        else None
                    ),
                    "actual": actual,
                    "actual_kind": "missing" if actual is None else "file",
                }
            )
    return conflicts


def require_bounded_conflicts(conflicts, max_bytes):
    encoded = json.dumps(
        conflicts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > int(max_bytes):
        fail("resource_exhausted", "conflict_evidence_limit_exceeded")


def count_active_branches(root_fd, excluded_names=()):
    parent = os.path.abspath("..")
    root_info = os.fstat(root_fd)
    active = set()
    try:
        names = os.listdir(parent)
    except OSError:
        fail("resource_exhausted", "active_branch_capacity_unavailable")
    present_names = set(names)
    for name in names:
        if not name.startswith(PRIVATE_PREFIX) or name in excluded_names:
            continue
        if name.endswith(".stage.claim.bound.pending"):
            staging_name = name.removesuffix(".claim.bound.pending")
            if staging_name in present_names:
                continue
        elif name.endswith(".stage.claim.pending"):
            staging_name = name.removesuffix(".claim.pending")
            if staging_name in present_names:
                continue
        elif name.endswith(".stage.claim.bound"):
            staging_name = name.removesuffix(".claim.bound")
            if staging_name in present_names:
                continue
        elif name.endswith(".stage.claim"):
            staging_name = name.removesuffix(".claim")
            if staging_name in present_names:
                continue
        match = re.fullmatch(
            re.escape(PRIVATE_PREFIX)
            + r"([0-9a-f]{64})(?:\.stage(?:\.claim(?:\.bound)?(?:\.pending)?)?)?",
            name,
        )
        capacity_identity = match.group(1) if match is not None else name
        candidate = os.path.join(parent, name)
        candidate_fd = None
        record_fd = None
        try:
            candidate_fd = os.open(
                candidate,
                OPEN_BASE_FLAGS | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            info = os.fstat(candidate_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_dev != root_info.st_dev:
                active.add(capacity_identity)
                continue
            record_info = os.stat(
                RECORD_NAME,
                dir_fd=candidate_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(record_info.st_mode)
                or stat.S_ISLNK(record_info.st_mode)
                or record_info.st_nlink != 1
                or record_info.st_size > 16 * 1024 * 1024
            ):
                active.add(capacity_identity)
                continue
            record_fd = os.open(
                RECORD_NAME,
                OPEN_BASE_FLAGS | os.O_NOFOLLOW,
                dir_fd=candidate_fd,
            )
            opened_record_info = os.fstat(record_fd)
            if (
                opened_record_info.st_dev != record_info.st_dev
                or opened_record_info.st_ino != record_info.st_ino
            ):
                active.add(capacity_identity)
                continue
            record = json.loads(read_all_bounded(record_fd, 16 * 1024 * 1024))
            valid_record = (
                isinstance(record, dict)
                and record.get("integrity") == record_digest(record)
                and record.get("private_root") == [info.st_dev, info.st_ino]
            )
            if valid_record and record.get("source_root") != [
                root_info.st_dev,
                root_info.st_ino,
            ]:
                continue
            retained_private_content = False
            for area in ("baseline", "overlay"):
                try:
                    os.stat(area, dir_fd=candidate_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                retained_private_content = True
                break
            if (
                not valid_record
                or record.get("state") not in TERMINAL_STATES
                or retained_private_content
            ):
                active.add(capacity_identity)
        except Exception:
            # Unknown retained material consumes capacity rather than being
            # guessed safe or deleted.
            active.add(capacity_identity)
        finally:
            close_fd(record_fd)
            close_fd(candidate_fd)
    return len(active)


def sweep_expired_process_local_branches(root_fd):
    global FAILURE_OUTPUT_ENABLED

    # A process-local creator can lose the command acknowledgement after the
    # guest has opened the branch. No durable authority exists from which a new
    # process could reconstruct that handle, so expiry is the bounded cleanup
    # owner. Sweep before taking the source lock: branch operations take their
    # branch lock before the source lock during publication, and reversing that
    # order here would deadlock with an expiring publication.
    parent = os.path.abspath("..")
    root_info = os.fstat(root_fd)
    try:
        names = os.listdir(parent)
    except OSError:
        return
    for name in names:
        if not name.startswith(PRIVATE_PREFIX):
            continue
        candidate = os.path.join(parent, name)
        previous_failure_output = FAILURE_OUTPUT_ENABLED
        FAILURE_OUTPUT_ENABLED = False
        try:
            info = require_private_directory(candidate, root_fd)
            if info.st_dev != root_info.st_dev:
                continue
            with branch_lock(candidate, root_fd) as pinned_private:
                record = load_record(pinned_private, root_fd)
                if (
                    record.get("source_root") != [root_info.st_dev, root_info.st_ino]
                    or record.get("authority") is not None
                ):
                    continue
                state = record.get("state")
                if state in ("creating", *OPEN_STATES) and int(time.time() * 1000) >= int(
                    record["expires_at_ms"]
                ):
                    expire_record(pinned_private, record)
                elif (
                    state == "rollback_intent"
                    and record.get("rollback", {}).get("reason") == "expired"
                ):
                    settle_rollback_intent(pinned_private, record)
        except BaseException:
            # Unknown or concurrently replaced material remains capacity
            # bearing.  Admission must not guess that it is safe to remove.
            continue
        finally:
            FAILURE_OUTPUT_ENABLED = previous_failure_output


def cleanup_expired_creation_claims(root_fd):
    global FAILURE_OUTPUT_ENABLED

    parent = os.path.abspath("..")
    now_ms = int(time.time() * 1000)
    try:
        names = os.listdir(parent)
    except OSError:
        return
    processed_claims = set()
    for name in names:
        if not name.startswith(PRIVATE_PREFIX):
            continue
        if name.endswith(".stage.claim"):
            claim_name = name
        elif name.endswith(".stage.claim.pending"):
            claim_name = name.removesuffix(CREATION_CLAIM_PENDING_SUFFIX)
        else:
            continue
        if claim_name in processed_claims:
            continue
        processed_claims.add(claim_name)
        claim_path = os.path.join(parent, claim_name)
        previous_failure_output = FAILURE_OUTPUT_ENABLED
        FAILURE_OUTPUT_ENABLED = False
        try:
            claim = load_creation_claim(claim_path, root_fd)
            if (
                claim is None
                or type(claim.get("expires_at_ms")) is not int
                or claim["expires_at_ms"] > now_ms
            ):
                continue
            staging = claim_path.removesuffix(".claim")
            created_staging = False
            if not os.path.lexists(staging):
                if claim.get("staging_root") is not None:
                    # The positively bound inode was displaced. Preserve its
                    # immutable claims and consume capacity rather than
                    # manufacturing replacement ownership.
                    continue
                try:
                    os.mkdir(staging, 0o700)
                    created_staging = True
                except FileExistsError:
                    continue
            elif claim.get("staging_root") is None:
                # POSIX has no portable atomic mkdir-and-return-fd primitive.
                # An entry retained in that pre-bind window cannot be proven
                # to be Cayu's, even when it is empty. Never adopt or remove it.
                continue
            with branch_lock(staging, root_fd) as pinned_staging:
                staging_info = require_private_directory(pinned_staging, root_fd)
                if claim.get("staging_root") is None:
                    if not created_staging:
                        fail("fenced", "workspace_branch_creation_claim_unbound")
                    if os.listdir(pinned_staging):
                        fail("fenced", "workspace_branch_creation_staging_unowned")
                    claim = bind_creation_claim(
                        claim_path,
                        claim,
                        staging_info,
                        root_fd,
                    )
                    fsync_parent_directory(staging)
                else:
                    require_claimed_creation_directory(
                        pinned_staging,
                        claim,
                        root_fd,
                    )
                if os.path.lexists(os.path.join(pinned_staging, RECORD_NAME)):
                    record = load_record(pinned_staging, root_fd)
                    if not exact_creation_record(record, claim):
                        fail("fenced", "workspace_branch_creation_claim_invalid")
                    if record.get("state") in TERMINAL_STATES:
                        continue
                    if record.get("state") == "rollback_intent":
                        settle_rollback_intent(pinned_staging, record)
                        continue
                    if record.get("state") not in ("creating", *OPEN_STATES):
                        fail("fenced", "workspace_branch_creation_claim_invalid")
                else:
                    cleanup_pinned_directory(pinned_staging)
                    record = new_creation_record(
                        root_fd,
                        claim,
                        staging_info,
                        {},
                        {},
                        [],
                    )
                    record = write_record(pinned_staging, record)
                # CAYU_TEST_BEFORE_EXPIRED_CREATION_TERMINALIZATION
                expire_record(pinned_staging, record)
        except BaseException:
            # Unknown creation material remains capacity bearing.
            continue
        finally:
            FAILURE_OUTPUT_ENABLED = previous_failure_output


def durable_operation_authority(
    record,
    *,
    mismatch_error_type="fenced",
    mismatch_detail_code="workspace_branch_operation_authority_changed",
):
    creation = record.get("authority")
    operation = record.get("operation_authority")
    if creation is None:
        if operation is not None:
            fail(mismatch_error_type, mismatch_detail_code)
        return None
    if not isinstance(creation, dict) or not isinstance(operation, dict):
        fail(mismatch_error_type, mismatch_detail_code)
    expected_fields = {
        "session_id",
        "expected_run_epoch",
        "environment_name",
        "binding_generation",
        "binding_identity",
        "creating_authority",
        "resource_policy",
    }
    if set(creation) != expected_fields or set(operation) != expected_fields:
        fail(mismatch_error_type, mismatch_detail_code)
    if any(
        operation.get(name) != creation.get(name)
        for name in (
            "session_id",
            "environment_name",
            "binding_generation",
            "binding_identity",
            "resource_policy",
        )
    ):
        fail(mismatch_error_type, mismatch_detail_code)
    if (
        type(creation.get("expected_run_epoch")) is not int
        or type(operation.get("expected_run_epoch")) is not int
        or not isinstance(creation.get("creating_authority"), str)
        or not isinstance(operation.get("creating_authority"), str)
    ):
        fail(mismatch_error_type, mismatch_detail_code)
    if (
        operation["expected_run_epoch"] < creation["expected_run_epoch"]
        or (
            operation["expected_run_epoch"] == creation["expected_run_epoch"]
            and operation != creation
        )
    ):
        fail(mismatch_error_type, mismatch_detail_code)
    return operation


def require_record_envelope(record, payload, *, allow_authority_handoff=False):
    if record.get("branch_id") != payload.get("branch_id"):
        fail("operation_conflict", "workspace_branch_identity_mismatch")
    if record.get("allocation_fingerprint") != payload.get("allocation_fingerprint"):
        fail("operation_conflict", "workspace_branch_allocation_changed")
    current = payload.get("binding_authority")
    recorded = record.get("authority")
    if recorded is None:
        if current is not None or payload.get("operation_authority") is not None:
            fail("operation_conflict", "workspace_branch_unexpected_binding_authority")
    else:
        operation = durable_operation_authority(
            record,
            mismatch_error_type=("operation_conflict" if allow_authority_handoff else "fenced"),
            mismatch_detail_code=(
                "workspace_branch_recovery_authority_changed"
                if allow_authority_handoff
                else "workspace_branch_operation_authority_changed"
            ),
        )
        if not isinstance(current, dict) or any(
            current.get(name) != recorded.get(name)
            for name in ("environment_name", "binding_generation", "binding_identity")
        ):
            fail("operation_conflict", "workspace_branch_binding_authority_changed")
        invoked = payload.get("operation_authority")
        if not isinstance(invoked, dict):
            fail("operation_conflict", "workspace_branch_operation_authority_changed")
        if not allow_authority_handoff and invoked != operation:
            fail("operation_conflict", "workspace_branch_operation_authority_changed")


def settle_rollback_intent(private, record):
    if record.get("state") != "rollback_intent":
        fail("fenced", "workspace_branch_rollback_intent_invalid")
    rollback = record.get("rollback")
    if (
        not isinstance(rollback, dict)
        or rollback.get("branch_id") != record.get("branch_id")
        or not isinstance(rollback.get("idempotency_key"), str)
        or not rollback["idempotency_key"]
        or rollback.get("reason") not in ("explicit", "expired")
        or not isinstance(record.get("terminal_digest"), str)
    ):
        fail("fenced", "workspace_branch_rollback_intent_invalid")
    cleanup_private_content(private)
    terminal = "expired" if rollback["reason"] == "expired" else "rolled_back"
    record["state"] = terminal
    record["detail_code"] = "workspace_branch_" + terminal
    return write_record(private, record)


def expire_record(private, record):
    if record.get("state") != "rollback_intent":
        changes = changes_for(record)
        digest = change_set_digest(record, changes)
        record["state"] = "rollback_intent"
        operation_authority = durable_operation_authority(record)
        record["rollback"] = {
            "branch_id": record["branch_id"],
            "idempotency_key": "automatic-expiry",
            "expected_run_epoch": (
                0 if operation_authority is None else operation_authority["expected_run_epoch"]
            ),
            "binding_generation": (
                "process-local"
                if operation_authority is None
                else operation_authority["binding_generation"]
            ),
            "reason": "expired",
        }
        record["terminal_digest"] = digest
        record = write_record(private, record)
    # CAYU_TEST_AFTER_EXPIRED_ROLLBACK_INTENT
    return settle_rollback_intent(private, record)


def require_open(private, record):
    if record.get("state") not in OPEN_STATES:
        fail("branch_closed", "workspace_branch_not_open")
    if int(time.time() * 1000) >= int(record["expires_at_ms"]):
        expire_record(private, record)
        fail("branch_closed", "workspace_branch_expired")


def logical_paths(record):
    paths = (set(record["baseline"]) | set(record["overlay"])) - set(record["deleted"])
    return sorted(paths)


def logical_directories(record):
    directories = set(record["baseline_directories"])
    for path in logical_paths(record):
        parts = path.split("/")[:-1]
        directories.update("/".join(parts[: index + 1]) for index in range(len(parts)))
    return directories


def require_file_shape(record, path, *, existing_required=False):
    paths = logical_paths(record)
    if path in logical_directories(record):
        fail("not_file", "workspace_branch_path_is_directory")
    parts = path.split("/")
    for index in range(1, len(parts)):
        if "/".join(parts[:index]) in paths:
            fail("not_file", "workspace_branch_parent_is_file")
    if any(candidate.startswith(path + "/") for candidate in paths):
        fail("not_file", "workspace_branch_path_is_directory")
    exists = path in paths
    if existing_required and not exists:
        fail("not_found", "workspace_branch_file_not_found")
    return exists


def logical_content(private, record, path):
    if path in record["deleted"]:
        fail("not_found", "workspace_branch_file_not_found")
    if path in record["overlay"]:
        return read_private_blob(private, "overlay", path, record["overlay"][path])
    if path in record["baseline"]:
        return read_private_blob(private, "baseline", path, record["baseline"][path])
    fail("not_found", "workspace_branch_file_not_found")


def changes_for(record):
    changes = []
    baseline = record["baseline"]
    overlay = record["overlay"]
    deleted = set(record["deleted"])
    for path in sorted(set(overlay) | deleted):
        before = baseline.get(path)
        after = None if path in deleted else overlay.get(path)
        if before is None and after is not None:
            operation = "created"
        elif before is not None and after is None:
            operation = "deleted"
        elif before is not None and after is not None:
            operation = "modified"
        else:
            continue
        changes.append(
            {
                "path": path,
                "operation": operation,
                "before": public_identity(before),
                "after": public_identity(after),
            }
        )
    return changes


def change_set_digest(record, changes):
    payload = {
        "baseline_revision": record["baseline_revision"],
        "branch_id": record["branch_id"],
        "changes": changes,
        "source": record["source"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def require_bounded_change_set(record, changes):
    payload = {
        "branch_id": record["branch_id"],
        "source": record["source"],
        "baseline_revision": record["baseline_revision"],
        "changes": changes,
        "digest": change_set_digest(record, changes),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > int(record["limits"]["max_evidence_bytes"]):
        fail("resource_exhausted", "change_evidence_limit_exceeded")


def enforce_overlay_limits(record):
    limits = record["limits"]
    changes = changes_for(record)
    changed = len(changes)
    if changed > int(limits["max_changed_paths"]):
        fail("resource_exhausted", "changed_path_limit_exceeded")
    overlay_bytes = sum(int(value["bytes"]) for value in record["overlay"].values())
    if overlay_bytes > int(limits["max_overlay_bytes"]):
        fail("resource_exhausted", "overlay_byte_limit_exceeded")
    if len(logical_paths(record)) + len(logical_directories(record)) > int(
        limits["max_paths"]
    ):
        fail("resource_exhausted", "path_count_limit_exceeded")
    require_bounded_change_set(record, changes)


def private_blob_observation(path, max_bytes):
    try:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            fail("fenced", "workspace_branch_private_content_invalid")
        with open(path, "rb") as file:
            content = file.read(int(max_bytes) + 1)
    except FileNotFoundError:
        return None
    if len(content) > int(max_bytes):
        fail("fenced", "workspace_branch_private_content_changed")
    return identity(content)


def mutation_identity_matches(observed, expected):
    return (observed is None and expected is None) or same_identity(observed, expected)


def apply_private_mutation_state(record, mutation, prefix):
    overlay = mutation[prefix + "_overlay"]
    path = mutation["path"]
    if overlay is None:
        record["overlay"].pop(path, None)
    else:
        record["overlay"][path] = overlay
    deleted = set(record["deleted"])
    if mutation[prefix + "_deleted"]:
        deleted.add(path)
    else:
        deleted.discard(path)
    record["deleted"] = sorted(deleted)
    record["mutation"] = None
    record["state"] = "open"
    record["publication"] = None
    record["detail_code"] = None


def settle_private_mutation(private, record):
    mutation = record.get("mutation")
    if mutation is None:
        return record
    if (
        not isinstance(mutation, dict)
        or not isinstance(mutation.get("path"), str)
        or type(mutation.get("before_deleted")) is not bool
        or type(mutation.get("after_deleted")) is not bool
    ):
        fail("fenced", "workspace_branch_private_mutation_invalid")
    path = canonical_path(mutation["path"])
    before = mutation.get("before_overlay")
    after = mutation.get("after_overlay")
    for expected in (before, after):
        if expected is not None and (
            not isinstance(expected, dict)
            or type(expected.get("sha256")) is not str
            or type(expected.get("bytes")) is not int
            or expected["bytes"] < 0
        ):
            fail("fenced", "workspace_branch_private_mutation_invalid")
    destination = private_blob_path(private, "overlay", path)
    bound = max(
        0 if before is None else int(before["bytes"]),
        0 if after is None else int(after["bytes"]),
    )
    observed = private_blob_observation(destination, bound)
    temporary_name = mutation.get("temporary_name")
    if temporary_name is not None and (
        type(temporary_name) is not str
        or re.fullmatch(r"\.cayu-mutation-[0-9a-f]{32}\.tmp", temporary_name) is None
    ):
        fail("fenced", "workspace_branch_private_mutation_invalid")
    temporary = (
        None
        if temporary_name is None
        else os.path.join(os.path.dirname(destination), temporary_name)
    )
    if mutation_identity_matches(observed, after):
        if temporary is not None:
            durable_unlink(temporary)
        apply_private_mutation_state(record, mutation, "after")
        return write_record(private, record)
    temporary_observed = (
        None if temporary is None else private_blob_observation(temporary, bound)
    )
    if after is not None and mutation_identity_matches(temporary_observed, after):
        if not mutation_identity_matches(observed, before):
            fail("fenced", "workspace_branch_private_mutation_ambiguous")
        os.replace(temporary, destination)
        parent_fd = os.open(os.path.dirname(destination), OPEN_BASE_FLAGS | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        apply_private_mutation_state(record, mutation, "after")
        return write_record(private, record)
    if temporary_observed is not None:
        if not mutation_identity_matches(observed, before):
            fail("fenced", "workspace_branch_private_mutation_ambiguous")
        durable_unlink(temporary)
        apply_private_mutation_state(record, mutation, "before")
        return write_record(private, record)
    if mutation_identity_matches(observed, before):
        apply_private_mutation_state(record, mutation, "before")
        return write_record(private, record)
    fail("fenced", "workspace_branch_private_mutation_ambiguous")


def commit_private_mutation(
    private,
    record,
    path,
    *,
    after_content,
    after_overlay,
    after_deleted,
):
    before_overlay = record["overlay"].get(path)
    before_content = (
        None
        if before_overlay is None
        else read_private_blob(private, "overlay", path, before_overlay)
    )
    before_deleted = path in record["deleted"]
    projected = dict(record)
    projected["overlay"] = dict(record["overlay"])
    projected["deleted"] = list(record["deleted"])
    projected_mutation = {
        "path": path,
        "before_overlay": before_overlay,
        "after_overlay": after_overlay,
        "before_deleted": before_deleted,
        "after_deleted": after_deleted,
        "temporary_name": (
            None
            if after_overlay is None
            else ".cayu-mutation-" + os.urandom(16).hex() + ".tmp"
        ),
    }
    apply_private_mutation_state(projected, projected_mutation, "after")
    enforce_overlay_limits(projected)
    record["mutation"] = projected_mutation
    record = write_record(private, record)
    destination = private_blob_path(private, "overlay", path)
    temporary_name = projected_mutation["temporary_name"]
    temporary = (
        None
        if temporary_name is None
        else os.path.join(os.path.dirname(destination), temporary_name)
    )
    try:
        if after_overlay is None:
            remove_private_blob(private, "overlay", path)
        else:
            ensure_private_parents(destination, private)
            atomic_file_write(destination, after_content, temporary=temporary)
        # CAYU_TEST_AFTER_PRIVATE_MUTATION
        apply_private_mutation_state(record, projected_mutation, "after")
        return write_record(private, record)
    except BaseException:
        if before_overlay is None:
            remove_private_blob(private, "overlay", path)
        else:
            write_private_blob(private, "overlay", path, before_content)
        if temporary is not None:
            durable_unlink(temporary)
        apply_private_mutation_state(record, projected_mutation, "before")
        write_record(private, record)
        raise


def mutate_content(private, record, path, content):
    limits = record["limits"]
    if len(content) > int(limits["max_file_bytes"]):
        fail("resource_exhausted", "file_byte_limit_exceeded")
    if path not in record["baseline"] and path not in record["overlay"]:
        current_paths = set(logical_paths(record))
        if len(current_paths) >= int(limits["max_files"]):
            fail("resource_exhausted", "file_count_limit_exceeded")
    after = identity(content)
    before = record["baseline"].get(path)
    after_overlay = None if before is not None and same_identity(before, after) else after
    return commit_private_mutation(
        private,
        record,
        path,
        after_content=content,
        after_overlay=after_overlay,
        after_deleted=False,
    )


def source_path_state(root_fd, path, max_file_bytes):
    parent_fd = None
    leaf_fd = None
    try:
        parent_fd, leaf = open_guarded_parent(root_fd, guarded_parts(path), False)
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            return "symlink", None, None
        if stat.S_ISDIR(info.st_mode):
            return "directory", None, None
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return "special", None, None
        if info.st_size > int(max_file_bytes):
            return "special", None, None
        leaf_fd, opened = open_guarded_regular(leaf, parent_fd)
        content = read_all_bounded(leaf_fd, max_file_bytes)
        if len(content) > int(max_file_bytes):
            return "special", None, None
        observed = stored_identity(content, opened.st_mode)
        return "file", observed, content
    except GuardPathError as exc:
        if exc.status in ("enoent", "notdir"):
            return "missing", None, None
        if exc.status == "escape":
            return "symlink", None, None
        return "special", None, None
    except FileNotFoundError:
        return "missing", None, None
    finally:
        close_fd(leaf_fd)
        close_fd(parent_fd)


def publication_conflicts(root_fd, record, changes):
    conflicts = []
    checked_ancestors = set()
    deleted_paths = {
        change["path"] for change in changes if change["operation"] == "deleted"
    }
    for change in changes:
        path = change["path"]
        parts = path.split("/")
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            if ancestor in checked_ancestors:
                continue
            checked_ancestors.add(ancestor)
            kind, actual, _ = source_path_state(
                root_fd,
                ancestor,
                record["limits"]["max_file_bytes"],
            )
            if kind not in ("missing", "directory") and ancestor not in deleted_paths:
                conflicts.append(
                    {
                        "path": ancestor,
                        "expected": None,
                        "actual": public_identity(actual),
                        "actual_kind": kind,
                    }
                )
        kind, actual, _ = source_path_state(
            root_fd,
            path,
            record["limits"]["max_file_bytes"],
        )
        expected = record["baseline"].get(path)
        if change["operation"] == "created":
            matches = kind == "missing"
        else:
            matches = kind == "file" and same_source_identity(expected, actual)
        if not matches:
            conflicts.append(
                {
                    "path": path,
                    "expected": public_identity(expected),
                    "actual": public_identity(actual),
                    "actual_kind": kind,
                }
            )
    unique = {}
    for conflict in conflicts:
        unique.setdefault(conflict["path"], conflict)
    return [unique[path] for path in sorted(unique)]


def atomic_source_write(root_fd, path, content, *, require_missing, mode):
    parent_fd = None
    temporary = None
    try:
        parent_fd, leaf = open_guarded_parent(
            root_fd,
            guarded_parts(path),
            True,
            True,
        )
        if require_missing:
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise GuardPathError("exists")
        temporary = _temporary_name(leaf)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            GUARDED_FILE_CREATE_MODE,
            dir_fd=parent_fd,
        )
        try:
            write_all(fd, content)
            os.fchmod(fd, int(mode))
            os.fsync(fd)
        finally:
            os.close(fd)
        if require_missing:
            os.link(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=parent_fd)
            temporary = None
        else:
            os.rename(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temporary = None
        # CAYU_TEST_AFTER_SOURCE_WRITE_MUTATION
        os.fsync(parent_fd)
    finally:
        if temporary is not None and parent_fd is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        close_fd(parent_fd)


def atomic_source_delete(root_fd, path):
    parent_fd = None
    try:
        parent_fd, leaf = open_guarded_parent(
            root_fd,
            guarded_parts(path),
            False,
            True,
        )
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise GuardPathError("notfile")
        os.unlink(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        close_fd(parent_fd)


def publication_created_directories(root_fd, record, changes):
    deleted_paths = {
        change["path"] for change in changes if change["operation"] == "deleted"
    }
    created = set()
    for change in changes:
        if change["operation"] != "created":
            continue
        parts = change["path"].split("/")
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            kind, _, _ = source_path_state(
                root_fd,
                ancestor,
                record["limits"]["max_file_bytes"],
            )
            if kind == "missing" or ancestor in deleted_paths:
                created.add(ancestor)
    return sorted(created, key=lambda value: (value.count("/"), value))


def remove_empty_source_directory(root_fd, path):
    parent_fd = None
    try:
        parent_fd, leaf = open_guarded_parent(
            root_fd,
            guarded_parts(path),
            False,
            True,
        )
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise GuardPathError("notdir")
        os.rmdir(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except (FileNotFoundError, NotADirectoryError):
        return
    except GuardPathError as exc:
        if exc.status in ("enoent", "notdir"):
            return
        raise
    finally:
        close_fd(parent_fd)


def remove_created_directory_subtree(root_fd, path, created_directories):
    prefix = path + "/"
    for directory in reversed(created_directories):
        if directory == path or directory.startswith(prefix):
            remove_empty_source_directory(root_fd, directory)


def normalize_created_directory_modes(root_fd, created_directories, mode):
    for path in created_directories:
        parent_fd = None
        directory_fd = None
        try:
            parent_fd, leaf = open_guarded_parent(root_fd, guarded_parts(path), False)
            directory_fd = open_guarded_directory(leaf, parent_fd, False, True)
            os.fchmod(directory_fd, int(mode))
            os.fsync(directory_fd)
        finally:
            close_fd(directory_fd)
            close_fd(parent_fd)


def publication_after_identity(record, change):
    if change["after"] is None:
        return None
    after = dict(change["after"])
    if change["operation"] == "created":
        after["mode"] = int(record["publication"]["source_file_create_mode"])
    else:
        after["mode"] = int(record["baseline"][change["path"]]["mode"])
    return after


def apply_change(root_fd, private, record, change):
    path = change["path"]
    operation = change["operation"]
    if operation == "deleted":
        atomic_source_delete(root_fd, path)
        return
    content = read_private_blob(private, "overlay", path, change["after"])
    after = publication_after_identity(record, change)
    atomic_source_write(
        root_fd,
        path,
        content,
        require_missing=operation == "created",
        mode=after["mode"],
    )


def restore_change(root_fd, private, record, change, created_directories):
    path = change["path"]
    before = record["baseline"].get(path)
    after = publication_after_identity(record, change)
    kind, actual, _ = source_path_state(
        root_fd,
        path,
        record["limits"]["max_file_bytes"],
    )
    if before is not None and kind == "file" and same_source_identity(before, actual):
        return
    if before is None:
        if kind == "missing":
            return
        if kind != "file" or not same_source_identity(after, actual):
            raise GuardPathError("changed")
        atomic_source_delete(root_fd, path)
        return
    if kind == "directory" and path in created_directories:
        remove_created_directory_subtree(root_fd, path, created_directories)
        kind, actual, _ = source_path_state(
            root_fd,
            path,
            record["limits"]["max_file_bytes"],
        )
    if change["operation"] == "modified":
        if kind != "file" or not same_source_identity(after, actual):
            raise GuardPathError("changed")
    elif change["operation"] == "deleted":
        if kind != "missing":
            raise GuardPathError("changed")
    else:
        raise GuardPathError("changed")
    content = read_private_blob(private, "baseline", path, before)
    atomic_source_write(
        root_fd,
        path,
        content,
        require_missing=kind == "missing",
        mode=before["mode"],
    )


def publication_signature(record, request, digest):
    return {
        "branch_id": request.get("branch_id"),
        "baseline_revision": request.get("baseline_revision"),
        "change_set_digest": digest,
        "idempotency_key": request.get("idempotency_key"),
        "expected_run_epoch": request.get("expected_run_epoch"),
        "binding_generation": request.get("binding_generation"),
    }


def publication_key_digest(value):
    if not isinstance(value, str) or not value:
        fail("invalid_request", "workspace_branch_publication_request_invalid")
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def admit_publication_attempt(record, request, digest):
    attempts = record.get("publication_identities")
    if not isinstance(attempts, list):
        fail("fenced", "workspace_branch_publication_attempts_invalid")
    key = request.get("idempotency_key")
    if key is not None:
        key_digest = publication_key_digest(key)
        previous = next(
            (
                attempt
                for attempt in attempts
                if isinstance(attempt, dict)
                and attempt.get("idempotency_key_digest") == key_digest
            ),
            None,
        )
        if previous is not None:
            if previous.get("change_set_digest") != digest:
                fail("operation_conflict", "workspace_branch_publication_identity_reused")
            return
    if int(record.get("publication_attempts", 0)) >= int(
        record["limits"]["max_publication_attempts"]
    ):
        fail("resource_exhausted", "publication_attempt_limit_exceeded")
    record["publication_attempts"] = int(record.get("publication_attempts", 0)) + 1
    if key is not None:
        attempts.append(
            {
                "idempotency_key_digest": publication_key_digest(key),
                "change_set_digest": digest,
            }
        )


def require_publication_authority(record, request):
    authority = durable_operation_authority(record)
    durable = authority is not None
    durable_fields = (
        request.get("idempotency_key"),
        request.get("expected_run_epoch"),
        request.get("binding_generation"),
    )
    if durable:
        if any(value is None for value in durable_fields):
            fail("operation_conflict", "workspace_branch_publication_authority_missing")
        if request.get("expected_run_epoch") != authority.get("expected_run_epoch"):
            fail("operation_conflict", "workspace_branch_run_epoch_changed")
        if request.get("binding_generation") != authority.get("binding_generation"):
            fail("operation_conflict", "workspace_branch_binding_generation_changed")
    elif any(value is not None for value in durable_fields):
        fail("operation_conflict", "workspace_branch_publication_authority_unexpected")


def settle_publication(root_fd, private, record):
    publication = record.get("publication")
    if not isinstance(publication, dict):
        fail("fenced", "workspace_branch_publication_record_missing")
    changes = publication.get("changes")
    if not isinstance(changes, list):
        fail("fenced", "workspace_branch_publication_changes_missing")
    created_directories = publication.get("created_directories")
    if not isinstance(created_directories, list) or any(
        not isinstance(path, str) for path in created_directories
    ):
        fail("fenced", "workspace_branch_publication_directories_missing")
    applied = []
    try:
        for change in changes:
            kind, actual, _ = source_path_state(
                root_fd,
                change["path"],
                record["limits"]["max_file_bytes"],
            )
            before = record["baseline"].get(change["path"])
            after = publication_after_identity(record, change)
            before_matches = (
                kind == "missing"
                if before is None
                else kind == "file" and same_source_identity(before, actual)
            )
            after_matches = (
                (kind == "missing" or (kind == "directory" and change["path"] in created_directories))
                if after is None
                else kind == "file" and same_source_identity(after, actual)
            )
            if after_matches:
                applied.append(change)
                continue
            if not before_matches:
                record["state"] = "ambiguous"
                record["detail_code"] = "workspace_branch_publication_source_ambiguous"
                write_record(private, record)
                return record, "ambiguous", []
            applied.append(change)
            apply_change(root_fd, private, record, change)
            record["state"] = "publication_progress"
            record["publication"]["applied_paths"] = [item["path"] for item in applied]
            record = write_record(private, record)
            # CAYU_TEST_AFTER_PUBLICATION_PROGRESS
        normalize_created_directory_modes(
            root_fd,
            created_directories,
            publication["source_directory_create_mode"],
        )
        # CAYU_TEST_AFTER_PUBLICATION_APPLY
        for change in changes:
            kind, actual, _ = source_path_state(
                root_fd,
                change["path"],
                record["limits"]["max_file_bytes"],
            )
            after = publication_after_identity(record, change)
            matches = (
                (kind == "missing" or (kind == "directory" and change["path"] in created_directories))
                if after is None
                else kind == "file" and same_source_identity(after, actual)
            )
            if not matches:
                record["state"] = "ambiguous"
                record["detail_code"] = "workspace_branch_publication_verification_failed"
                write_record(private, record)
                return record, "ambiguous", []
    except BaseException:
        rollback_failed = False
        for change in reversed(applied):
            try:
                restore_change(root_fd, private, record, change, created_directories)
            except BaseException:
                rollback_failed = True
        for directory in reversed(created_directories):
            try:
                remove_empty_source_directory(root_fd, directory)
            except BaseException:
                rollback_failed = True
        record["state"] = "ambiguous" if rollback_failed else "failed"
        record["detail_code"] = (
            "workspace_branch_publication_rollback_failed"
            if rollback_failed
            else "workspace_branch_publication_failed"
        )
        record = write_record(private, record)
        if not rollback_failed:
            # CAYU_TEST_BEFORE_FAILED_CLEANUP
            cleanup_private_content(private)
        return record, record["state"], []
    record["state"] = "committed"
    record["detail_code"] = "workspace_branch_committed"
    record = write_record(private, record)
    # CAYU_TEST_BEFORE_COMMITTED_CLEANUP
    cleanup_private_content(private)
    return record, "committed", []


def exact_creation_record(record, material):
    return all(
        record.get(name) == material.get(name)
        for name in (
            "branch_id",
            "source",
            "baseline_revision",
            "allocation_fingerprint",
            "idempotency_key",
            "authority",
            "limits",
        )
    )


def new_creation_record(
    root_fd,
    payload,
    private_info,
    files,
    source_modes,
    source_directories,
):
    root_info = os.fstat(root_fd)
    now_ms = int(time.time() * 1000)
    created_at_ms = payload.get("created_at_ms")
    expires_at_ms = payload.get("expires_at_ms")
    if type(created_at_ms) is not int:
        created_at_ms = now_ms
    if type(expires_at_ms) is not int:
        expires_at_ms = now_ms + int(payload["limits"]["lifetime_ms"])
    return {
        "schema": 2,
        "revision": 0,
        "state": "creating",
        "branch_id": payload["branch_id"],
        "source": payload["source"],
        "source_root": [root_info.st_dev, root_info.st_ino],
        "private_root": [private_info.st_dev, private_info.st_ino],
        "baseline_revision": payload["baseline_revision"],
        "allocation_fingerprint": payload["allocation_fingerprint"],
        "idempotency_key": payload.get("idempotency_key"),
        "authority": payload.get("authority"),
        "operation_authority": payload.get("authority"),
        "limits": payload["limits"],
        "created_at_ms": created_at_ms,
        "expires_at_ms": expires_at_ms,
        "baseline": {
            path: stored_identity(content, source_modes[path])
            for path, content in sorted(files.items())
        },
        "baseline_directories": sorted(source_directories),
        "overlay": {},
        "deleted": [],
        "publication": None,
        "publication_attempts": 0,
        "publication_identities": [],
        "rollback": None,
        "mutation": None,
        "detail_code": None,
    }


def replay_created_branch(root_fd, payload, private, branch_id, claim_path):
    with branch_lock(private, root_fd) as pinned_private:
        private_info = require_private_directory(pinned_private, root_fd)
        existing = load_record(pinned_private, root_fd)
        if not exact_creation_record(existing, payload):
            fail("operation_conflict", "workspace_branch_creation_identity_reused")
        if existing.get("state") in OPEN_STATES:
            verify_private_content(pinned_private, existing)
            claim = load_creation_claim(claim_path, root_fd)
            if claim is not None and (
                not exact_creation_claim(claim, payload)
                or claim.get("staging_root")
                != [private_info.st_dev, private_info.st_ino]
            ):
                fail("fenced", "workspace_branch_creation_claim_invalid")
            emit({"ok": True, "status": "created", "branch_id": branch_id})
            return True
        if existing.get("state") == "creating":
            return False
        fail("operation_conflict", "workspace_branch_identity_terminal")


def create_operation(root_fd, payload):
    branch_id = payload.get("branch_id")
    if not isinstance(branch_id, str) or not branch_id:
        fail("invalid_request", "workspace_branch_id_invalid")
    private = private_path(root_fd, branch_id)
    staging = private
    claim_path = creation_claim_path(staging)
    expected_baseline = payload.get("baseline")
    if not isinstance(expected_baseline, dict):
        fail("invalid_request", "workspace_branch_baseline_invalid")
    record_path = os.path.join(private, RECORD_NAME)
    if os.path.lexists(record_path) and replay_created_branch(
        root_fd,
        payload,
        private,
        branch_id,
        claim_path,
    ):
        return
    sweep_expired_process_local_branches(root_fd)
    source_lock = workspace_source_lock(root_fd, True)
    source_lock.__enter__()
    source_lock_active = True
    try:
        cleanup_expired_creation_claims(root_fd)
        claim = load_creation_claim(claim_path, root_fd)
        if claim is not None and not exact_creation_claim(claim, payload):
            fail("operation_conflict", "workspace_branch_creation_identity_reused")
        reuse_staging = False
        if os.path.lexists(staging):
            if claim is None:
                fail("fenced", "workspace_branch_creation_claim_missing")
            if claim.get("staging_root") is None:
                # A directory retained before its first durable inode binding is
                # not adopted. It remains capacity-bearing for inspection.
                fail("fenced", "workspace_branch_creation_claim_unbound")
            with branch_lock(staging, root_fd) as pinned_staging:
                require_claimed_creation_directory(pinned_staging, claim, root_fd)
                if os.path.lexists(os.path.join(pinned_staging, RECORD_NAME)):
                    existing = load_record(pinned_staging, root_fd)
                    if not exact_creation_record(existing, payload):
                        fail("operation_conflict", "workspace_branch_creation_identity_reused")
                    if existing.get("state") in OPEN_STATES:
                        source_lock.__exit__(None, None, None)
                        source_lock_active = False
                        replay_created_branch(
                            root_fd,
                            payload,
                            private,
                            branch_id,
                            claim_path,
                        )
                        return
                    if existing.get("state") != "creating":
                        fail("operation_conflict", "workspace_branch_identity_terminal")
            reuse_staging = True
        elif claim is not None and claim.get("staging_root") is not None:
            fail("fenced", "workspace_branch_creation_staging_changed")

        excluded_names = []
        if claim is not None:
            excluded_names.extend(
                (
                    os.path.basename(claim_path),
                    os.path.basename(creation_bound_claim_path(claim_path)),
                    os.path.basename(creation_claim_pending_path(claim_path)),
                    os.path.basename(
                        creation_claim_pending_path(
                            creation_bound_claim_path(claim_path),
                        )
                    ),
                )
            )
        if reuse_staging:
            excluded_names.append(os.path.basename(staging))
        if count_active_branches(root_fd, tuple(excluded_names)) >= int(
            payload["limits"]["max_active_branches"]
        ):
            fail("resource_exhausted", "active_branch_limit_exceeded")

        files, source_modes, source_directories = collect_source_files(
            root_fd,
            payload["limits"],
        )
        conflicts = baseline_conflicts(expected_baseline, files)
        require_bounded_conflicts(conflicts, payload["limits"]["max_evidence_bytes"])
        if conflicts:
            emit({"ok": True, "status": "conflicted", "conflicts": conflicts})
            return

        if claim is None:
            claim = write_creation_claim(claim_path, root_fd, payload)
        if not reuse_staging:
            # CAYU_TEST_BEFORE_CREATION_DIRECTORY
            try:
                os.mkdir(staging, 0o700)
            except FileExistsError:
                fail("operation_conflict", "workspace_branch_private_identity_collision")
            # CAYU_TEST_AFTER_UNBOUND_CREATION_DIRECTORY

        with branch_lock(staging, root_fd) as pinned_staging:
            private_info = require_private_directory(pinned_staging, root_fd)
            if reuse_staging:
                require_claimed_creation_directory(pinned_staging, claim, root_fd)
                if os.path.lexists(os.path.join(pinned_staging, RECORD_NAME)):
                    existing = load_record(pinned_staging, root_fd)
                    if not exact_creation_record(existing, payload):
                        fail("operation_conflict", "workspace_branch_creation_identity_reused")
                    if existing.get("state") != "creating":
                        fail("operation_conflict", "workspace_branch_identity_terminal")
                # CAYU_TEST_BEFORE_CREATION_RETRY_RESET
                cleanup_pinned_directory(pinned_staging)
            else:
                if claim.get("staging_root") is not None:
                    fail("fenced", "workspace_branch_creation_claim_invalid")
                claim = bind_creation_claim(
                    claim_path,
                    claim,
                    private_info,
                    root_fd,
                )
                fsync_parent_directory(staging)
            # CAYU_TEST_AFTER_CREATION_DIRECTORY
            private_info = require_claimed_creation_directory(
                pinned_staging,
                claim,
                root_fd,
            )
            record = new_creation_record(
                root_fd,
                claim,
                private_info,
                files,
                source_modes,
                source_directories,
            )
            record = write_record(pinned_staging, record)
            # CAYU_TEST_AFTER_CREATION_RECORD
            for path, content in sorted(files.items()):
                write_private_blob(pinned_staging, "baseline", path, content)
            os.mkdir(os.path.join(pinned_staging, "overlay"), 0o700)
            record["state"] = "open"
            record = write_record(pinned_staging, record)
            # CAYU_TEST_AFTER_BRANCH_CAPTURE
            # CAYU_TEST_BEFORE_CREATION_ACKNOWLEDGEMENT
            published_info = require_private_directory(private, root_fd)
            if not same_directory_identity(published_info, private_info):
                fail("fenced", "workspace_branch_private_root_changed")
            fsync_parent_directory(private)
    finally:
        if source_lock_active:
            source_lock.__exit__(*sys.exc_info())
    emit({"ok": True, "status": "created", "branch_id": branch_id})


def load_for_operation(
    root_fd,
    payload,
    *,
    allow_terminal=False,
    allow_authority_handoff=False,
):
    canonical_private = private_path(root_fd, payload.get("branch_id"))
    lock = branch_lock(canonical_private, root_fd)
    private = lock.__enter__()
    try:
        # CAYU_TEST_AFTER_OPERATION_BRANCH_LOCK
        record = load_record(private, root_fd)
        # CAYU_TEST_AFTER_OPERATION_RECORD_LOAD
        require_record_envelope(
            record,
            payload,
            allow_authority_handoff=allow_authority_handoff,
        )
        if not allow_terminal:
            require_open(private, record)
        return private, record, lock
    except BaseException:
        lock.__exit__(*sys.exc_info())
        raise


def read_operation(root_fd, payload):
    private, record, lock = load_for_operation(root_fd, payload)
    try:
        path = canonical_path(payload.get("path"))
        require_bounded_operation_path(record, path)
        require_file_shape(record, path, existing_required=True)
        content = logical_content(private, record, path)
        offset = int(payload.get("offset"))
        limit = int(payload.get("limit"))
        if offset > len(content):
            emit({"ok": False, "error_type": "offset", "total_bytes": len(content)})
            return
        page = content[offset : offset + limit]
        complete = offset == 0 and len(page) == len(content)
        emit(
            {
                "ok": True,
                "content_base64": base64.b64encode(page).decode("ascii"),
                "total_bytes": len(content),
                "revision": revision(content) if complete else None,
                "sha256": identity(content)["sha256"] if complete else None,
            }
        )
    finally:
        lock.__exit__(None, None, None)


def write_operation(root_fd, payload, mode):
    private, record, lock = load_for_operation(root_fd, payload)
    try:
        path = canonical_path(payload.get("path"))
        require_bounded_operation_path(record, path)
        exists = require_file_shape(record, path, existing_required=mode in ("replace", "delete_if"))
        before_content = logical_content(private, record, path) if exists else None
        if mode == "create" and exists:
            fail("exists", "workspace_branch_file_exists")
        if mode in ("replace", "delete_if"):
            expected_revision = payload.get("expected_revision")
            actual_revision = revision(before_content)
            if expected_revision != actual_revision:
                emit(
                    {
                        "ok": False,
                        "error_type": "stale",
                        "expected_revision": expected_revision,
                        "actual_revision": actual_revision,
                    }
                )
                return
        if mode in ("delete", "delete_if"):
            if not exists:
                emit({"ok": True, "mutation": None})
                return
            record = commit_private_mutation(
                private,
                record,
                path,
                after_content=None,
                after_overlay=None,
                after_deleted=path in record["baseline"],
            )
            emit(
                {
                    "ok": True,
                    "mutation": {
                        "operation": "delete",
                        "before": identity(before_content),
                        "after": None,
                    },
                }
            )
            return
        try:
            content = base64.b64decode(payload.get("content_base64"), validate=True)
        except Exception:
            fail("invalid_request", "workspace_branch_content_invalid")
        record = mutate_content(private, record, path, content)
        emit(
            {
                "ok": True,
                "mutation": {
                    "operation": "create" if not exists else "replace",
                    "before": None if before_content is None else identity(before_content),
                    "after": identity(content),
                },
            }
        )
    finally:
        lock.__exit__(None, None, None)


def list_operation(root_fd, payload):
    private, record, lock = load_for_operation(root_fd, payload)
    try:
        pattern = re.compile(payload.get("pattern"))
        limit = int(payload.get("limit"))
        matches = [path for path in logical_paths(record) if pattern.fullmatch(path)]
        response = {
            "ok": True,
            "paths": matches[:limit],
            "total_count": len(matches),
        }
        encoded = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > int(record["limits"]["max_evidence_bytes"]):
            fail("resource_exhausted", "list_evidence_limit_exceeded")
        emit(response)
    finally:
        lock.__exit__(None, None, None)


def changes_operation(root_fd, payload):
    private, record, lock = load_for_operation(root_fd, payload)
    try:
        changes = changes_for(record)
        emit({"ok": True, "changes": changes, "digest": change_set_digest(record, changes)})
    finally:
        lock.__exit__(None, None, None)


def publish_operation(root_fd, payload):
    private, record, lock = load_for_operation(root_fd, payload, allow_terminal=True)
    try:
        request = payload.get("request")
        if not isinstance(request, dict):
            fail("invalid_request", "workspace_branch_publication_request_invalid")
        require_publication_authority(record, request)
        if record.get("state") == "committed":
            if record.get("publication", {}).get("signature") != publication_signature(
                record, request, request.get("change_set_digest")
            ):
                fail("operation_conflict", "workspace_branch_publication_identity_reused")
            cleanup_private_content(private)
            emit(
                {
                    "ok": True,
                    "status": "committed",
                    "changes": record["publication"]["changes"],
                    "digest": record["publication"]["signature"]["change_set_digest"],
                }
            )
            return
        if record.get("state") == "failed":
            if record.get("publication", {}).get("signature") != publication_signature(
                record, request, request.get("change_set_digest")
            ):
                fail("operation_conflict", "workspace_branch_publication_identity_reused")
            cleanup_private_content(private)
            emit(
                {
                    "ok": True,
                    "status": "failed",
                    "changes": record["publication"]["changes"],
                    "digest": record["publication"]["signature"]["change_set_digest"],
                    "detail_code": record.get("detail_code"),
                }
            )
            return
        if record.get("state") in ("publication_intent", "publication_progress"):
            signature = record.get("publication", {}).get("signature")
            if signature != publication_signature(record, request, request.get("change_set_digest")):
                fail("operation_conflict", "workspace_branch_publication_identity_reused")
            with workspace_source_lock(root_fd, True):
                record, status, conflicts = settle_publication(root_fd, private, record)
            emit(
                {
                    "ok": True,
                    "status": status,
                    "conflicts": conflicts,
                    "changes": record["publication"]["changes"],
                    "digest": record["publication"]["signature"]["change_set_digest"],
                    "detail_code": record.get("detail_code"),
                }
            )
            return
        require_open(private, record)
        changes = changes_for(record)
        digest = change_set_digest(record, changes)
        if (
            request.get("branch_id") != record["branch_id"]
            or request.get("baseline_revision") != record["baseline_revision"]
            or request.get("change_set_digest") != digest
        ):
            fail("operation_conflict", "workspace_branch_change_set_mismatch")
        signature = publication_signature(record, request, digest)
        admit_publication_attempt(record, request, digest)
        with workspace_source_lock(root_fd, True):
            conflicts = publication_conflicts(root_fd, record, changes)
            require_bounded_conflicts(conflicts, record["limits"]["max_evidence_bytes"])
            if conflicts:
                record["state"] = "conflicted"
                record["publication"] = {
                    "signature": signature,
                    "changes": changes,
                    "conflicts": conflicts,
                    "applied_paths": [],
                }
                record["detail_code"] = "workspace_branch_source_conflict"
                write_record(private, record)
                emit(
                    {
                        "ok": True,
                        "status": "conflicted",
                        "conflicts": conflicts,
                        "changes": changes,
                        "digest": digest,
                    }
                )
                return
            created_directories = publication_created_directories(root_fd, record, changes)
            record["state"] = "publication_intent"
            record["publication"] = {
                "signature": signature,
                "changes": changes,
                "applied_paths": [],
                "created_directories": created_directories,
                "source_file_create_mode": effective_creation_mode(
                    GUARDED_FILE_CREATE_MODE
                ),
                "source_directory_create_mode": effective_creation_mode(
                    GUARDED_DIRECTORY_CREATE_MODE
                ),
            }
            record["detail_code"] = None
            record = write_record(private, record)
            # CAYU_TEST_AFTER_PUBLICATION_INTENT
            record, status, conflicts = settle_publication(root_fd, private, record)
        emit(
            {
                "ok": True,
                "status": status,
                "conflicts": conflicts,
                "changes": changes,
                "digest": digest,
                "detail_code": record.get("detail_code"),
            }
        )
    finally:
        lock.__exit__(None, None, None)


def rollback_operation(root_fd, payload):
    private, record, lock = load_for_operation(root_fd, payload, allow_terminal=True)
    try:
        request = payload.get("request")
        if not isinstance(request, dict):
            fail("invalid_request", "workspace_branch_rollback_request_invalid")
        authority = durable_operation_authority(record)
        if request.get("branch_id") != record.get("branch_id"):
            fail("operation_conflict", "workspace_branch_rollback_authority_changed")
        if authority is not None:
            if (
                request.get("expected_run_epoch") != authority.get("expected_run_epoch")
                or request.get("binding_generation") != authority.get("binding_generation")
            ):
                fail("operation_conflict", "workspace_branch_rollback_authority_changed")
        if record.get("state") in ("rolled_back", "expired"):
            if record.get("rollback") != request:
                fail("operation_conflict", "workspace_branch_rollback_identity_reused")
            emit({"ok": True, "status": record["state"], "digest": record.get("terminal_digest")})
            return
        if record.get("state") == "rollback_intent":
            if record.get("rollback") != request:
                fail("operation_conflict", "workspace_branch_rollback_identity_reused")
            record = settle_rollback_intent(private, record)
            terminal = record["state"]
            emit({"ok": True, "status": terminal, "digest": record.get("terminal_digest")})
            return
        if record.get("state") == "committed":
            fail("branch_closed", "workspace_branch_already_committed")
        if record.get("state") in ("publication_intent", "publication_progress", "ambiguous"):
            fail("branch_closed", "workspace_branch_publication_unsettled")
        reason = request.get("reason", "explicit")
        terminal = "expired" if reason == "expired" else "rolled_back"
        changes = changes_for(record)
        digest = change_set_digest(record, changes)
        record["state"] = "rollback_intent"
        record["rollback"] = request
        record["terminal_digest"] = digest
        record = write_record(private, record)
        # CAYU_TEST_AFTER_ROLLBACK_INTENT
        record = settle_rollback_intent(private, record)
        emit({"ok": True, "status": terminal, "digest": digest})
    finally:
        lock.__exit__(None, None, None)


def recover_operation(root_fd, payload):
    private, record, lock = load_for_operation(
        root_fd,
        payload,
        allow_terminal=True,
        allow_authority_handoff=True,
    )
    try:
        request = payload.get("request")
        authority = record.get("authority")
        current_authority = payload.get("operation_authority")
        previous_authority = durable_operation_authority(
            record,
            mismatch_error_type="operation_conflict",
            mismatch_detail_code="workspace_branch_recovery_authority_changed",
        )
        if (
            authority is None
            or not isinstance(request, dict)
            or not isinstance(current_authority, dict)
        ):
            fail("operation_conflict", "workspace_branch_recovery_not_durable")
        candidate = dict(record)
        candidate["operation_authority"] = current_authority
        current_authority = durable_operation_authority(
            candidate,
            mismatch_error_type="operation_conflict",
            mismatch_detail_code="workspace_branch_recovery_authority_changed",
        )
        if any(
            authority.get(name) != current_authority.get(name)
            for name in (
                "session_id",
                "environment_name",
                "binding_generation",
                "binding_identity",
                "resource_policy",
            )
        ):
            fail("operation_conflict", "workspace_branch_recovery_authority_changed")
        if any(
            request.get(field) != current_authority.get(field)
            for field in (
                "session_id",
                "expected_run_epoch",
                "binding_generation",
                "binding_identity",
            )
        ):
            fail("operation_conflict", "workspace_branch_recovery_authority_changed")
        previous_epoch = previous_authority.get("expected_run_epoch")
        current_epoch = current_authority.get("expected_run_epoch")
        if (
            type(previous_epoch) is not int
            or type(current_epoch) is not int
            or current_epoch < previous_epoch
            or (current_epoch == previous_epoch and current_authority != previous_authority)
        ):
            fail("operation_conflict", "workspace_branch_recovery_authority_changed")
        if current_authority != previous_authority:
            record["operation_authority"] = current_authority
            record = write_record(private, record)
        # CAYU_TEST_AFTER_RECOVERY_AUTHORITY_HANDOFF
        record = settle_private_mutation(private, record)
        state = record.get("state")
        if state in OPEN_STATES and int(time.time() * 1000) >= int(record["expires_at_ms"]):
            record = expire_record(private, record)
            state = "expired"
        if state in ("publication_intent", "publication_progress"):
            with workspace_source_lock(root_fd, True):
                record, state, _ = settle_publication(root_fd, private, record)
        elif state == "rollback_intent":
            record = settle_rollback_intent(private, record)
            state = record["state"]
        elif state in OPEN_STATES:
            verify_private_content(private, record)
        elif state in ("committed", "failed"):
            cleanup_private_content(private)
        response = {
            "ok": True,
            "state": state,
            "branch_id": record["branch_id"],
            "source": record["source"],
            "baseline_revision": record["baseline_revision"],
            "limits": record["limits"],
            "authority": durable_operation_authority(record),
            "detail_code": record.get("detail_code"),
        }
        if state == "committed":
            response.update(
                {
                    "changes": record["publication"]["changes"],
                    "digest": record["publication"]["signature"]["change_set_digest"],
                }
            )
        elif state in ("rolled_back", "expired"):
            response["digest"] = record.get("terminal_digest")
        emit(response)
    finally:
        lock.__exit__(None, None, None)


def main():
    operation = sys.argv[1]
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        fail("invalid_request", "workspace_branch_request_invalid")
    if not isinstance(payload, dict):
        fail("invalid_request", "workspace_branch_request_invalid")
    root_fd = None
    try:
        require_branch_guard_support()
        root_fd = open_guard_root(".", True)
        if operation == "create":
            create_operation(root_fd, payload)
        elif operation == "read":
            read_operation(root_fd, payload)
        elif operation in ("write", "create_file", "replace", "delete", "delete_if"):
            mode = {
                "write": "write",
                "create_file": "create",
                "replace": "replace",
                "delete": "delete",
                "delete_if": "delete_if",
            }[operation]
            write_operation(root_fd, payload, mode)
        elif operation == "list":
            list_operation(root_fd, payload)
        elif operation == "changes":
            changes_operation(root_fd, payload)
        elif operation == "publish":
            publish_operation(root_fd, payload)
        elif operation == "rollback":
            rollback_operation(root_fd, payload)
        elif operation == "recover":
            recover_operation(root_fd, payload)
        else:
            fail("invalid_request", "workspace_branch_operation_invalid")
    except GuardPathError as exc:
        mapping = {
            "enoent": ("not_found", "workspace_branch_path_not_found"),
            "notdir": ("not_file", "workspace_branch_parent_not_directory"),
            "notfile": ("not_file", "workspace_branch_path_not_file"),
            "isdir": ("not_file", "workspace_branch_path_is_directory"),
            "escape": ("invalid_path", "workspace_branch_path_escape"),
            "hardlink": ("invalid_path", "workspace_branch_hardlink_unsupported"),
            "exists": ("exists", "workspace_branch_path_exists"),
            "unsupported": ("unsupported", "workspace_branch_guest_guard_unsupported"),
        }
        error_type, detail_code = mapping.get(
            exc.status,
            ("fenced", "workspace_branch_guard_failed"),
        )
        fail(error_type, detail_code)
    except SystemExit:
        raise
    except BaseException:
        # Provider output is intentionally not forwarded. A caller can recover
        # the durable record to distinguish failed from ambiguous publication.
        fail("workspace_error", "workspace_branch_guest_operation_failed")
    finally:
        close_fd(root_fd)


main()
"""
)


__all__ = ["RUNNER_WORKSPACE_BRANCH_PROGRAM"]

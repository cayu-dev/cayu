from __future__ import annotations

import hashlib
import os
import secrets
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cayu.workspaces.base import WorkspaceMutationOperation, WorkspaceMutationResult


def content_identity(content: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}", digest


def mutation_result(
    operation: WorkspaceMutationOperation,
    *,
    before: bytes | None,
    after: bytes | None,
) -> WorkspaceMutationResult:
    before_revision, before_sha256 = (
        content_identity(before) if before is not None else (None, None)
    )
    after_revision, after_sha256 = content_identity(after) if after is not None else (None, None)
    return WorkspaceMutationResult(
        operation=operation,
        before_revision=before_revision,
        after_revision=after_revision,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        before_bytes=len(before) if before is not None else None,
        after_bytes=len(after) if after is not None else None,
    )


def mutation_result_from_identities(
    operation: WorkspaceMutationOperation,
    *,
    before: tuple[str, str, int] | None,
    after: bytes | None,
) -> WorkspaceMutationResult:
    after_revision, after_sha256 = content_identity(after) if after is not None else (None, None)
    return WorkspaceMutationResult(
        operation=operation,
        before_revision=before[0] if before is not None else None,
        after_revision=after_revision,
        before_sha256=before[1] if before is not None else None,
        after_sha256=after_sha256,
        before_bytes=before[2] if before is not None else None,
        after_bytes=len(after) if after is not None else None,
    )


def file_content_identity(path: Path) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1 << 16):
            digest.update(chunk)
            size += len(chunk)
    hexdigest = digest.hexdigest()
    return f"sha256:{hexdigest}", hexdigest, size


@contextmanager
def workspace_path_lock(root: Path, relative_path: str) -> Iterator[None]:
    """Serialize cooperative workspace clients addressing one root/path."""

    root_info = root.stat()
    normalized_path = unicodedata.normalize(
        "NFC",
        relative_path.replace("\\", "/"),
    ).casefold()
    key = hashlib.sha256(
        f"{root_info.st_dev}:{root_info.st_ino}\0{normalized_path}".encode()
    ).hexdigest()
    lock_root = Path(tempfile.gettempdir()) / "cayu-workspace-locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock_root / key, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = _open_create_temp(path)
    try:
        with os.fdopen(descriptor, "wb") as temp:
            temp.write(content)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _open_create_temp(path: Path) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _attempt in range(100):
        candidate = path.parent / f".{path.name}.cayu-{secrets.token_hex(12)}"
        try:
            return os.open(candidate, flags, 0o666), candidate
        except FileExistsError:
            continue
    raise OSError("Could not allocate an atomic workspace temporary file.")


def atomic_replace(path: Path, content: bytes) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.cayu-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        mode = path.stat().st_mode
        with os.fdopen(descriptor, "wb") as temp:
            temp.write(content)
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

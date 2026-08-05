from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cayu._filesystem_lock import cooperative_path_lock
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


@contextmanager
def workspace_path_lock(root: Path, relative_path: str) -> Iterator[None]:
    """Serialize cooperative workspace clients addressing one root/path."""

    with cooperative_path_lock(
        root,
        relative_path,
        lock_directory_name="cayu-workspace-locks",
    ):
        yield

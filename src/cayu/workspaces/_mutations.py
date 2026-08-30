from __future__ import annotations

import hashlib
import threading
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from cayu._filesystem_lock import cooperative_path_lock
from cayu.workspaces.base import (
    WorkspaceMoveFidelity,
    WorkspaceMoveResult,
    WorkspaceMutationOperation,
    WorkspaceMutationResult,
)


@dataclass(slots=True)
class _LocalSourceGate:
    readers: int = 0
    writer: bool = False
    waiters: int = 0
    waiting_writers: int = 0


_LOCAL_SOURCE_CONDITION = threading.Condition()
_LOCAL_SOURCE_GATES: dict[str, _LocalSourceGate] = {}
_FENCED_LOCAL_SOURCES: set[str] = set()


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


def move_result_from_identity(
    identity: tuple[str, str, int],
    *,
    fidelity: WorkspaceMoveFidelity,
) -> WorkspaceMoveResult:
    revision, digest, size = identity
    return WorkspaceMoveResult(
        source_before_revision=revision,
        source_after_revision=None,
        destination_before_revision=None,
        destination_after_revision=revision,
        source_before_sha256=digest,
        source_after_sha256=None,
        destination_before_sha256=None,
        destination_after_sha256=digest,
        source_before_bytes=size,
        source_after_bytes=None,
        destination_before_bytes=None,
        destination_after_bytes=size,
        fidelity=fidelity,
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


@contextmanager
def workspace_path_locks(root: Path, *relative_paths: str) -> Iterator[None]:
    """Acquire multiple cooperative path locks in one deterministic order."""

    ordered = sorted(
        relative_paths,
        key=lambda value: unicodedata.normalize("NFC", value.replace("\\", "/")).casefold(),
    )
    identities = [
        unicodedata.normalize("NFC", value.replace("\\", "/")).casefold() for value in ordered
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Workspace paths must identify distinct files.")
    with ExitStack() as stack:
        for path in ordered:
            stack.enter_context(workspace_path_lock(root, path))
        yield


@contextmanager
def workspace_source_lock(
    root: Path,
    *,
    exclusive: bool,
    fence_on_cleanup_failure: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Coordinate source-wide publication with ordinary local operations.

    Every cooperative LocalWorkspace operation takes the shared side before a
    path lock. Branch creation and publication take the exclusive side and
    invoke descriptor-guarded primitives directly while they own it.
    """

    source_identity = _local_source_identity(root)
    _acquire_local_source_gate(source_identity, exclusive=exclusive)
    body_completed = False
    try:
        try:
            with cooperative_path_lock(
                root,
                "__cayu_workspace_source__",
                lock_directory_name="cayu-workspace-source-locks",
                shared=not exclusive,
                retain_on_exit=(
                    (lambda: local_workspace_source_is_fenced(root)) if exclusive else None
                ),
            ):
                _raise_if_local_workspace_source_fenced(root)
                yield
                body_completed = True
        except BaseException:
            if (
                body_completed
                and fence_on_cleanup_failure is not None
                and fence_on_cleanup_failure()
            ):
                fence_local_workspace_source(root)
            raise
    finally:
        _release_local_source_gate(source_identity, exclusive=exclusive)


def fence_local_workspace_source(root: Path) -> None:
    """Permanently fence process-local reuse after uncertain publication rollback."""

    with _LOCAL_SOURCE_CONDITION:
        _FENCED_LOCAL_SOURCES.add(_local_source_identity(root))
        _LOCAL_SOURCE_CONDITION.notify_all()


def local_workspace_source_is_fenced(root: Path) -> bool:
    with _LOCAL_SOURCE_CONDITION:
        return _local_source_identity(root) in _FENCED_LOCAL_SOURCES


def _raise_if_local_workspace_source_fenced(root: Path) -> None:
    if local_workspace_source_is_fenced(root):
        raise _local_workspace_fenced_error()


def _acquire_local_source_gate(source_identity: str, *, exclusive: bool) -> None:
    acquired = False
    with _LOCAL_SOURCE_CONDITION:
        gate = _LOCAL_SOURCE_GATES.setdefault(source_identity, _LocalSourceGate())
        gate.waiters += 1
        if exclusive:
            gate.waiting_writers += 1
        try:
            while True:
                if source_identity in _FENCED_LOCAL_SOURCES:
                    raise _local_workspace_fenced_error()
                if exclusive:
                    if not gate.writer and gate.readers == 0:
                        gate.writer = True
                        gate.waiting_writers -= 1
                        acquired = True
                        return
                elif not gate.writer and gate.waiting_writers == 0:
                    gate.readers += 1
                    acquired = True
                    return
                _LOCAL_SOURCE_CONDITION.wait()
        finally:
            gate.waiters -= 1
            if exclusive and not acquired:
                gate.waiting_writers -= 1
            _discard_local_source_gate_if_idle(source_identity, gate)


def _release_local_source_gate(source_identity: str, *, exclusive: bool) -> None:
    with _LOCAL_SOURCE_CONDITION:
        gate = _LOCAL_SOURCE_GATES.get(source_identity)
        if gate is None:  # pragma: no cover - acquisition/release invariant
            raise RuntimeError("Local workspace source gate ownership disappeared.")
        if exclusive:
            if not gate.writer:  # pragma: no cover - acquisition/release invariant
                raise RuntimeError("Local workspace exclusive source gate was not owned.")
            gate.writer = False
        else:
            if gate.readers <= 0:  # pragma: no cover - acquisition/release invariant
                raise RuntimeError("Local workspace shared source gate was not owned.")
            gate.readers -= 1
        _LOCAL_SOURCE_CONDITION.notify_all()
        _discard_local_source_gate_if_idle(source_identity, gate)


def _discard_local_source_gate_if_idle(
    source_identity: str,
    gate: _LocalSourceGate,
) -> None:
    if gate.readers == 0 and not gate.writer and gate.waiters == 0 and gate.waiting_writers == 0:
        _LOCAL_SOURCE_GATES.pop(source_identity, None)


def _local_workspace_fenced_error() -> BaseException:
    from cayu.workspaces.branches import WorkspaceBranchFencedError

    return WorkspaceBranchFencedError(
        "Local workspace is fenced after incomplete branch publication settlement."
    )


def _local_source_identity(root: Path) -> str:
    return str(root)

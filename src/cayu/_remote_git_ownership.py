"""Nonblocking, process-shared ownership of a broker's private repository."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

from cayu.artifacts.settlement import ArtifactWriteSettlementObserver

_OBSERVERS: dict[tuple[str, str], ArtifactWriteSettlementObserver] = {}
_RETAINED_LOCKS: dict[tuple[str, str], int] = {}
_OBSERVER_LOCK = RLock()
_MAX_OBSERVERS = 128
_CURRENT_OBSERVER: ContextVar[ArtifactWriteSettlementObserver | None] = ContextVar(
    "remote_git_artifact_owner", default=None
)


def delivery_write_pending() -> bool:
    observer = _CURRENT_OBSERVER.get()
    return observer is not None and bool(observer.record_active_candidates())


class RemoteGitOwnershipUnavailable(RuntimeError):
    """An existing delivery owner must settle before another call can enter."""


@contextmanager
def exclusive_delivery(root: Path, delivery_id_sha256: str) -> Iterator[None]:
    key = (str(root), delivery_id_sha256)
    with _OBSERVER_LOCK:
        retained = _RETAINED_LOCKS.get(key)
        if retained is not None:
            prior = _OBSERVERS.get(key)
            if prior is None or prior.record_active_candidates():
                raise RemoteGitOwnershipUnavailable(
                    "Remote Git artifact settlement still owns the delivery lock."
                )
            os.close(retained)
            del _RETAINED_LOCKS[key]
    # Never unlink a lock: a waiter and a new caller must name the same inode.
    descriptor = os.open(
        root / f"owner-{delivery_id_sha256}.lock",
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    retain_descriptor = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise RemoteGitOwnershipUnavailable("Remote Git ownership file is unsafe.")
        try:
            if os.name == "nt":
                import msvcrt

                if metadata.st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RemoteGitOwnershipUnavailable(
                "Remote Git delivery is already owned by another call."
            ) from None
        os.lseek(descriptor, 0, os.SEEK_SET)
        uncertain = os.read(descriptor, 1) == b"U"
        with _OBSERVER_LOCK:
            prior = _OBSERVERS.get(key)
            if uncertain and (prior is None or prior.record_active_candidates()):
                raise RemoteGitOwnershipUnavailable(
                    "Remote Git artifact settlement requires reconstruction before reuse."
                )
            if key not in _OBSERVERS and len(_OBSERVERS) >= _MAX_OBSERVERS:
                raise RemoteGitOwnershipUnavailable("Remote Git ownership capacity is exhausted.")
            observer = ArtifactWriteSettlementObserver(max_operations=1)
            _OBSERVERS[key] = observer
        if uncertain:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        with observer:
            token = _CURRENT_OBSERVER.set(observer)
            primary: BaseException | None = None
            try:
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                _CURRENT_OBSERVER.reset(token)
                if observer.record_active_candidates():
                    # An interrupted store can return while its thread still
                    # mutates durable state. Keep a cross-process fence, and
                    # retain the observer for a later positive local settlement.
                    try:
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        if os.write(descriptor, b"U") != 1:
                            raise OSError("Remote Git ownership marker made no progress.")
                        os.fsync(descriptor)
                    except OSError as secondary:
                        # If durable fencing fails, the exact OS lock must
                        # remain owned until the active artifact write settles.
                        with _OBSERVER_LOCK:
                            _RETAINED_LOCKS[key] = descriptor
                        retain_descriptor = True
                        if primary is not None:
                            causes = [] if primary.__cause__ is None else [primary.__cause__]
                            causes.append(secondary)
                            raise primary from BaseExceptionGroup(
                                "Remote Git ownership-marker settlement failed", causes
                            )
                        raise RemoteGitOwnershipUnavailable(
                            "Remote Git artifact settlement retains its delivery lock."
                        ) from secondary
                else:
                    with _OBSERVER_LOCK:
                        _OBSERVERS.pop(key, None)
    finally:
        if not retain_descriptor:
            os.close(descriptor)

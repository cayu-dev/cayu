"""Credentials the sidecar authenticates itself and its host with in Kubernetes mode."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import stat
import threading
from collections.abc import Iterator

# The sidecar runs as a non-root user in its own container, which holds nothing
# of the agent's. Its token and TLS key belong to root or to that user; its
# group may read them (a Secret volume with an fsGroup), but not write them,
# and nobody else may access them.
ROOT_UID = 0
_GROUP_WRITE = 0o020
_OTHER_ACCESS = 0o007
MIN_TOKEN_BYTES = 32
MAX_TOKEN_FILE_BYTES = 4096


class TokenFileError(RuntimeError):
    """The token or TLS key file is not safe to use."""


@contextlib.contextmanager
def _private_file(path: str, label: str) -> Iterator[tuple[int, os.stat_result]]:
    """Open a regular file that only root or this process's user may have written."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise TokenFileError(f"cannot open {label} {path}: {exc.strerror}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise TokenFileError(f"{label} {path} is not a regular file")
        uid = os.geteuid()
        if status.st_uid not in (ROOT_UID, uid):
            raise TokenFileError(
                f"{label} {path} must be owned by uid 0 or by the sidecar's uid {uid}"
            )
        if status.st_mode & _GROUP_WRITE:
            raise TokenFileError(f"{label} {path} must not be writable by its group")
        if status.st_mode & _OTHER_ACCESS:
            raise TokenFileError(f"{label} {path} must not be accessible to others")
        yield descriptor, status
    finally:
        os.close(descriptor)


def check_private_key_file(path: str) -> None:
    """Refuse a TLS private key file that another user could read or could have written."""
    with _private_file(path, "sidecar TLS key file"):
        pass


def _load(path: str) -> tuple[bytes, int]:
    """Return the token's SHA-256 digest and the file's modification time."""
    with _private_file(path, "sidecar token file") as (descriptor, status):
        token = os.read(descriptor, MAX_TOKEN_FILE_BYTES + 1).strip()
    if len(token) > MAX_TOKEN_FILE_BYTES:
        raise TokenFileError(f"sidecar token file {path} is too large")
    if len(token) < MIN_TOKEN_BYTES:
        raise TokenFileError(f"sidecar token in {path} must be at least {MIN_TOKEN_BYTES} bytes")
    return hashlib.sha256(token).digest(), status.st_mtime_ns


class BearerToken:
    """The host's token, read from a private file and compared in constant time.

    Every check first looks at the file's modification time, so a token rotated
    on disk replaces the earlier one on the next request. A replacement that
    is not a safe token file is ignored and the earlier token stays.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._digest, self._modified_ns = _load(path)

    def accepts(self, presented: bytes) -> bool:
        presented_digest = hashlib.sha256(presented).digest()
        with self._lock:
            try:
                modified_ns = os.stat(self._path).st_mtime_ns
            except OSError:
                return False
            if modified_ns != self._modified_ns:
                with contextlib.suppress(TokenFileError):
                    self._digest, self._modified_ns = _load(self._path)
            return hmac.compare_digest(presented_digest, self._digest)

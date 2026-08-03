from __future__ import annotations

import hashlib
import os
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def cooperative_path_lock(
    root: Path,
    relative_path: str,
    *,
    lock_directory_name: str,
) -> Iterator[None]:
    """Serialize cooperative processes addressing one root-relative path."""

    root_info = root.stat()
    normalized_path = unicodedata.normalize(
        "NFC",
        relative_path.replace("\\", "/"),
    ).casefold()
    key = hashlib.sha256(
        f"{root_info.st_dev}:{root_info.st_ino}\0{normalized_path}".encode()
    ).hexdigest()
    lock_root = Path(tempfile.gettempdir()) / lock_directory_name
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

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
    acquired = False
    operation_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
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
            acquired = True
            yield
        except BaseException as error:
            # Leave the except block before lock cleanup so a cleanup exception
            # does not acquire the primary error as an implicit context edge.
            operation_error = error
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException as error:
                if error.__cause__ is None and error.__context__ is operation_error:
                    error.__context__ = None
                cleanup_errors.append(error)
        try:
            # Closing the descriptor is the final lock-release mechanism and
            # must run even when explicit unlock or acquisition failed.
            os.close(descriptor)
        except BaseException as error:
            if error.__cause__ is None and error.__context__ is operation_error:
                error.__context__ = None
            cleanup_errors.append(error)

    if operation_error is not None:
        if cleanup_errors:
            previous_evidence = operation_error.__cause__
            if (
                previous_evidence is None
                and operation_error.__context__ is not None
                and not operation_error.__suppress_context__
            ):
                previous_evidence = operation_error.__context__
            evidence = [
                *(() if previous_evidence is None else (previous_evidence,)),
                *cleanup_errors,
            ]
            cause = (
                evidence[0]
                if len(evidence) == 1
                else BaseExceptionGroup(
                    "Filesystem lock operation and cleanup failures.",
                    evidence,
                )
            )
            raise operation_error from cause
        raise operation_error
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    if cleanup_errors:
        primary_cleanup_error, *later_cleanup_errors = cleanup_errors
        cause = (
            later_cleanup_errors[0]
            if len(later_cleanup_errors) == 1
            else BaseExceptionGroup(
                "Additional filesystem lock cleanup failures.",
                later_cleanup_errors,
            )
        )
        raise primary_cleanup_error from cause

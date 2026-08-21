"""Durable private local-state writes for the Cayu Cloud CLI."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_WINDOWS_MOVEFILE_REPLACE_EXISTING = 0x1
_WINDOWS_MOVEFILE_WRITE_THROUGH = 0x8
_PRIVATE_JSON_PUBLICATION_RETRY_DELAY_SECONDS = 0.05
_PRIVATE_JSON_COW_METADATA_HEADROOM_BYTES = 64 * 1024
_PRIVATE_JSON_STAGING_PREFIX = ".cayu-private-json-"
_PRIVATE_JSON_HEADROOM_PREFIX = ".cayu-private-json-headroom-"


@dataclass(slots=True)
class PreparedPrivateJsonWrite:
    """Capacity-reserved private JSON publication owned by one caller."""

    path: Path
    temporary_path: Path
    descriptor: int
    headroom: BinaryIO | None
    reserved_bytes: int
    headroom_released: bool = False
    replacement_ready: bool = False
    published: bool = False

    def publish(self, payload: dict[str, Any]) -> None:
        """Synchronize and atomically publish ``payload`` from the reservation."""

        if self.published:
            raise ValueError("Private JSON staging has already been published.")
        self._prepare_replacement(payload)
        self._publish_replacement()

    def retry_publication(self, payload: dict[str, Any]) -> None:
        """Retry preparation or replacement while the exact payload is owned."""

        if self.published:
            raise ValueError("Private JSON staging has already been published.")
        time.sleep(_PRIVATE_JSON_PUBLICATION_RETRY_DELAY_SECONDS)
        if not self.replacement_ready:
            self._prepare_replacement(payload)
        self._publish_replacement()

    def _prepare_replacement(self, payload: dict[str, Any]) -> None:
        encoded = _private_json_payload(payload)
        if len(encoded) > self.reserved_bytes:
            raise ValueError("Private JSON staging capacity is insufficient.")
        self.release_headroom()
        _overwrite_reserved_payload(
            self.descriptor,
            encoded,
            reserved_bytes=self.reserved_bytes,
        )
        os.fsync(self.descriptor)
        self.release_descriptor()
        self.replacement_ready = True

    def _publish_replacement(self) -> None:
        _replace_private_json(self.temporary_path, self.path)
        self.published = True
        self.synchronize_publication()

    def release_descriptor(self) -> None:
        """Close and relinquish the staging descriptor exactly once."""

        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)

    def release_headroom(self) -> None:
        """Release separately allocated capacity before a copy-on-write rewrite."""

        if self.headroom_released:
            return
        headroom = self.headroom
        if headroom is not None:
            headroom.close()
        self.headroom = None
        self.headroom_released = True

    def synchronize_publication(self) -> None:
        """Synchronize the directory entry for an already published payload."""

        if not self.published:
            raise ValueError("Private JSON staging has not been published.")
        _sync_private_json_directory(self.path)

    def adopt_observed_publication(self) -> None:
        """Adopt exact destination readback and synchronize its directory entry."""

        self.published = True
        self.synchronize_publication()

    def cleanup(self) -> None:
        """Release owned resources and remove unpublished staging exactly once."""

        failures: list[OSError] = []
        try:
            self.release_descriptor()
        except OSError as exc:
            failures.append(exc)
        try:
            self.release_headroom()
        except OSError as exc:
            failures.append(exc)
        if not self.published:
            try:
                self.temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Private JSON staging cleanup failed.", failures)


@contextmanager
def prepare_private_json(
    path: Path,
    reservation_payload: dict[str, Any],
    *,
    staging_prefix: str = _PRIVATE_JSON_STAGING_PREFIX,
) -> Iterator[PreparedPrivateJsonWrite]:
    """Reserve a synchronized private staging file before an external mutation."""

    _validate_staging_prefix(staging_prefix)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=staging_prefix,
    )
    temporary_path = Path(temporary_name)
    prepared: PreparedPrivateJsonWrite | None = None
    try:
        os.chmod(temporary_path, 0o600)
        reserved_bytes = len(_private_json_payload(reservation_payload))
        # Incompressible bytes force a real allocation on filesystems that may
        # otherwise optimize a zero- or whitespace-filled staging file.
        _overwrite_payload(descriptor, os.urandom(reserved_bytes))
        os.fsync(descriptor)
        # Keep copy-on-write headroom anonymous/delete-on-close. On platforms
        # that require delete authority, TemporaryFile establishes it before
        # the external credential rotation rather than unlinking afterward.
        with tempfile.TemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=_PRIVATE_JSON_HEADROOM_PREFIX,
        ) as headroom:
            headroom_bytes = reserved_bytes + _PRIVATE_JSON_COW_METADATA_HEADROOM_BYTES
            _overwrite_payload(headroom.fileno(), os.urandom(headroom_bytes))
            os.fsync(headroom.fileno())
            prepared = PreparedPrivateJsonWrite(
                path=path,
                temporary_path=temporary_path,
                descriptor=descriptor,
                headroom=headroom,
                reserved_bytes=reserved_bytes,
            )
            primary: BaseException | None = None
            primary_traceback = None
            try:
                yield prepared
            except BaseException as exc:
                primary = exc
                primary_traceback = exc.__traceback__
            try:
                prepared.cleanup()
            except (OSError, ExceptionGroup) as cleanup_error:
                if primary is None:
                    raise
                _attach_cleanup_failure(primary, cleanup_error)
            if primary is not None:
                raise primary.with_traceback(primary_traceback)
    finally:
        if prepared is None:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def remove_private_json_staging(path: Path, *, staging_prefix: str) -> int:
    """Remove regular private staging owned by one exact destination."""

    _validate_staging_prefix(staging_prefix)
    try:
        with os.scandir(path.parent) as entries:
            candidates = sorted(
                (entry for entry in entries if entry.name.startswith(staging_prefix)),
                key=lambda entry: entry.name,
            )
    except FileNotFoundError:
        return 0
    observed = len(candidates)
    validated_candidates = []
    for entry in candidates:
        try:
            evidence = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            # A refresh owner may finish removing its staging after this scan.
            # Absence is positive cleanup evidence, not a logout failure.
            continue
        if not stat.S_ISREG(evidence.st_mode):
            raise OSError("Cloud authentication staging is not a regular file.")
        if os.name != "nt":
            if stat.S_IMODE(evidence.st_mode) & 0o077:
                raise OSError("Cloud authentication staging is not private.")
            getuid = getattr(os, "getuid", None)
            if callable(getuid) and evidence.st_uid != getuid():
                raise OSError("Cloud authentication staging has another owner.")
            if evidence.st_nlink != 1:
                raise OSError("Cloud authentication staging has unexpected links.")
        validated_candidates.append(entry)
    removed = 0
    failures: list[OSError] = []
    for entry in validated_candidates:
        try:
            Path(entry.path).unlink()
        except FileNotFoundError:
            # The validated owner may have completed cleanup before this unlink.
            continue
        except OSError as exc:
            failures.append(exc)
        else:
            removed += 1
    if removed:
        try:
            _sync_private_json_directory(path)
        except OSError as exc:
            failures.append(exc)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup("Private JSON staging removal failed.", failures)
    return observed


def _validate_staging_prefix(value: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    ):
        raise ValueError("Private JSON staging prefix is invalid.")


def _attach_cleanup_failure(
    primary: BaseException,
    cleanup_error: Exception,
) -> None:
    existing_cause = primary.__cause__
    if existing_cause is None:
        cause: BaseException = cleanup_error
    else:
        cause = BaseExceptionGroup(
            "Private JSON operation and cleanup failures.",
            [existing_cause, cleanup_error],
        )
    primary.__cause__ = cause
    primary.__suppress_context__ = True


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one private JSON document and its directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=_PRIVATE_JSON_STAGING_PREFIX,
    )
    temporary_path = Path(temporary_name)
    try:
        os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = -1
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_private_json(temporary_path, path)
        path.chmod(0o600)
        _sync_private_json_directory(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _private_json_payload(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sync_private_json_directory(path: Path) -> None:
    if os.name == "nt":
        # _replace_private_json() publishes with MOVEFILE_WRITE_THROUGH on
        # Windows, where ordinary directory descriptors cannot be fsynced.
        return
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _replace_private_json(source: Path, destination: Path) -> None:
    if os.name == "nt":
        _move_file_ex_windows(
            source,
            destination,
            flags=(_WINDOWS_MOVEFILE_REPLACE_EXISTING | _WINDOWS_MOVEFILE_WRITE_THROUGH),
        )
        return
    os.replace(source, destination)


def _move_file_ex_windows(source: Path, destination: Path, *, flags: int) -> None:
    import ctypes
    from ctypes import wintypes

    win_dll = getattr(ctypes, "WinDLL", None)
    win_error = getattr(ctypes, "WinError", None)
    get_last_error = getattr(ctypes, "get_last_error", None)
    if not callable(win_dll) or not callable(win_error) or not callable(get_last_error):
        raise OSError("Windows write-through replacement is unavailable")
    move_file_ex = win_dll("kernel32", use_last_error=True).MoveFileExW
    move_file_ex.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
    move_file_ex.restype = wintypes.BOOL
    if not move_file_ex(os.fspath(source), os.fspath(destination), flags):
        raise win_error(get_last_error())


def _overwrite_payload(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write private JSON staging")
        remaining = remaining[written:]


def _overwrite_reserved_payload(
    descriptor: int,
    payload: bytes,
    *,
    reserved_bytes: int,
) -> None:
    _overwrite_payload(descriptor, payload.ljust(reserved_bytes, b" "))
    if len(payload) == reserved_bytes:
        return
    try:
        os.ftruncate(descriptor, len(payload))
    except OSError:
        size = os.fstat(descriptor).st_size
        if size < len(payload) or size > reserved_bytes:
            raise

"""Bounded, regular-file-only reads of native process receipts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

PROCESS_DOCUMENT_MAX_BYTES = 64 * 1024 * 1024
PROCESS_INSPECTION_MAX_BYTES = 256 * 1024 * 1024


def write_process_document(path: Path, value: object) -> None:
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(data) > PROCESS_DOCUMENT_MAX_BYTES:
        raise ValueError("Process eval document exceeds its 64 MiB bound.")
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("xb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Process receipt contains a duplicate JSON key.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Process receipt contains nonfinite JSON: {value}.")


def decode_document(data: bytes) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if type(value) is not dict:
        raise ValueError("Process receipt must be a JSON object.")
    return value


def read_regular_file(path: Path, *, max_bytes: int) -> bytes | None:
    if path.is_symlink():
        raise ValueError(f"Inspection refuses a symbolic link: {path.name}.")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Inspection requires a regular file: {path.name}.")
        if info.st_size > max_bytes:
            raise ValueError(f"Inspection byte limit exceeded by {path.name}.")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Inspection byte limit exceeded by {path.name}.")
    return data


class ProcessDocuments:
    """Retain one bounded observation; exports use these same inspected bytes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.files: dict[str, bytes] = {}
        self.remaining = PROCESS_INSPECTION_MAX_BYTES

    def read(self, name: str, *, required: bool = False) -> dict[str, Any] | None:
        if Path(name).name != name:
            raise ValueError("Process receipt names must be direct directory entries.")
        data = read_regular_file(
            self.directory / name,
            max_bytes=min(PROCESS_DOCUMENT_MAX_BYTES, self.remaining),
        )
        if data is None:
            if required:
                raise ValueError(f"Missing process receipt: {name}.")
            return None
        self.remaining -= len(data)
        self.files[name] = data
        return decode_document(data)

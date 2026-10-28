"""Free-disk checks for storage operations that write large copies.

A check is a list of :class:`SpaceNeed` values: a directory, how many bytes the
operation will write there, and what for. Needs on the same file system add up.
:func:`require_free_space` raises :class:`InsufficientDiskSpaceError` naming
every file system that is short, with how much is needed and how much is free,
before the operation writes anything.

The SQLite helpers estimate what ``cayu storage migrate`` and ``VACUUM`` write;
other commands can build their own needs and reuse the same check.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: Working space a SQLite migration reserves next to the database, as a share
#: of the database's size, on top of the migrated copy itself. Backfills that
#: rewrite every row grow the copy's write-ahead log and the copy before the
#: log checkpoints: a 7.3 GB store used about 5.3 GB beyond its backup and
#: copy during the revision 129 backfill.
SQLITE_MIGRATION_WORKING_SPACE_RATIO = 0.75

#: The ``purpose`` of the retained-backup need in :func:`sqlite_migration_space_needs`.
SQLITE_BACKUP_PURPOSE = "for the backup"


@dataclass(frozen=True)
class SpaceNeed:
    """``bytes`` the operation will write under ``directory``, for ``purpose``."""

    directory: Path
    bytes: int
    purpose: str


@dataclass(frozen=True)
class SpaceShortfall:
    """A file system that lacks room for the needs placed on it."""

    directory: Path
    needed: int
    available: int
    needs: tuple[SpaceNeed, ...]


class InsufficientDiskSpaceError(RuntimeError):
    """Raised before an operation starts when a file system lacks room for it."""

    def __init__(self, message: str, shortfalls: tuple[SpaceShortfall, ...]) -> None:
        super().__init__(message)
        self.shortfalls = shortfalls


def format_bytes(count: int) -> str:
    """Render a byte count for operators, for example ``7.3 GB``."""

    value = float(count)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{count} bytes" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")


def existing_directory(path: Path) -> Path:
    """The nearest existing directory at or above ``path``.

    A directory the operation will create still lives on its closest existing
    ancestor's file system.
    """

    candidate = path.absolute()
    while not candidate.is_dir():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def file_system_id(directory: Path) -> int:
    """An identifier shared by every directory on the same file system."""

    return directory.stat().st_dev


def find_space_shortfalls(needs: Iterable[SpaceNeed]) -> tuple[SpaceShortfall, ...]:
    """Group ``needs`` by file system and return the ones without enough room."""

    groups: dict[int, list[SpaceNeed]] = {}
    directories: dict[int, Path] = {}
    for need in needs:
        if need.bytes <= 0:
            continue
        directory = existing_directory(need.directory)
        device = file_system_id(directory)
        groups.setdefault(device, []).append(need)
        directories.setdefault(device, directory)
    shortfalls: list[SpaceShortfall] = []
    for device, grouped in groups.items():
        directory = directories[device]
        needed = sum(need.bytes for need in grouped)
        available = shutil.disk_usage(directory).free
        if available < needed:
            shortfalls.append(
                SpaceShortfall(
                    directory=directory,
                    needed=needed,
                    available=available,
                    needs=tuple(grouped),
                )
            )
    return tuple(shortfalls)


def require_free_space(
    needs: Iterable[SpaceNeed], *, operation: str, hint: str | None = None
) -> None:
    """Raise :class:`InsufficientDiskSpaceError` when any file system is short.

    The message reads ``<operation> needs about N free in DIR (parts); M is
    free.`` for each short file system, followed by ``hint`` when given.
    """

    raise_for_shortfalls(find_space_shortfalls(needs), operation=operation, hint=hint)


def raise_for_shortfalls(
    shortfalls: tuple[SpaceShortfall, ...], *, operation: str, hint: str | None = None
) -> None:
    """Raise :class:`InsufficientDiskSpaceError` for ``shortfalls``, if there are any.

    Use this instead of :func:`require_free_space` when the hint depends on
    which file systems are short.
    """

    if not shortfalls:
        return
    sentences = []
    for shortfall in shortfalls:
        parts = ", ".join(f"{format_bytes(need.bytes)} {need.purpose}" for need in shortfall.needs)
        sentences.append(
            f"{operation} needs about {format_bytes(shortfall.needed)} free in "
            f"{shortfall.directory} ({parts}); {format_bytes(shortfall.available)} is free."
        )
    if hint:
        sentences.append(hint)
    raise InsufficientDiskSpaceError(" ".join(sentences), shortfalls)


def sqlite_file_set_bytes(path: Path) -> int:
    """The size of a SQLite database file plus its write-ahead log."""

    total = 0
    for candidate in (path, Path(f"{path}-wal")):
        with contextlib.suppress(FileNotFoundError):
            total += candidate.stat().st_size
    return total


def sqlite_temp_directory() -> Path:
    """The directory SQLite uses for temporary databases (such as VACUUM's) on POSIX."""

    for name in ("SQLITE_TMPDIR", "TMPDIR"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    for candidate in ("/var/tmp", "/usr/tmp", "/tmp"):
        if Path(candidate).is_dir():
            return Path(candidate)
    return Path(tempfile.gettempdir())


def sqlite_migration_space_needs(
    database: Path,
    *,
    backup_directory: Path | None,
    snapshot_directory: Path | None = None,
    migrates: bool = True,
    working_space_ratio: float = SQLITE_MIGRATION_WORKING_SPACE_RATIO,
) -> tuple[SpaceNeed, ...]:
    """Estimate what a copy-and-swap SQLite migration writes, from the database's size.

    The migration writes a snapshot of the database (the retained backup in
    ``backup_directory``, or a temporary snapshot in ``snapshot_directory``
    when the backup is waived; pass neither when the snapshot already exists),
    then a migrated copy next to the database that replaces it. ``migrates``
    adds working space for the copy's write-ahead log and growth while
    revisions or an alias-key backfill apply; leave it off when nothing will
    be written to the copy.
    """

    size = sqlite_file_set_bytes(database)
    if size == 0:
        return ()
    needs: list[SpaceNeed] = []
    if backup_directory is not None:
        needs.append(SpaceNeed(backup_directory, size, SQLITE_BACKUP_PURPOSE))
    elif snapshot_directory is not None:
        needs.append(SpaceNeed(snapshot_directory, size, "for a temporary snapshot"))
    needs.append(SpaceNeed(database.parent, size, "for the migrated copy"))
    if migrates and working_space_ratio > 0:
        needs.append(
            SpaceNeed(
                database.parent,
                math.ceil(size * working_space_ratio),
                "for the copy's write-ahead log and growth while the migration runs",
            )
        )
    return tuple(needs)


def sqlite_vacuum_space_needs(database: Path, data_bytes: int) -> tuple[SpaceNeed, ...]:
    """What ``VACUUM`` writes: a temporary copy of the data, then a rewrite of the file."""

    return (
        SpaceNeed(sqlite_temp_directory(), data_bytes, "for the temporary copy"),
        SpaceNeed(database.resolve().parent, data_bytes, f"to rewrite {database}"),
    )

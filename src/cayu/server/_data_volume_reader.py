"""Read-only, confined reads of the host's data volume.

Every read resolves its path on the volume itself, one name at a time from a directory file
descriptor of the volume's root: it never asks the operating system to resolve a whole path.
Each name is examined without following it (``lstat``) and opened with ``O_NOFOLLOW``, and
the opened file must be the one examined, so a name swapped for a link meanwhile is refused
rather than followed. A symbolic link is followed only by this resolver: its target is read
and resolved the same way, and a target that leads above the root, or an absolute target
outside the volume's directory, is refused. Only regular files are read and only directories
are listed or archived; a FIFO, socket or device is never opened.
"""

from __future__ import annotations

import base64
import bisect
import errno
import io
import os
import posixpath
import stat
import tarfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn

from cayu._data_volume import (
    DATA_VOLUME_DEADLINE_SECONDS,
    DATA_VOLUME_ENV,
    DATA_VOLUME_ROOT_NAME,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_DEPTH,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_INPUT_BYTES,
    MAX_ARCHIVE_SKIPPED_NAMED,
    MAX_DIRECTORY_SCAN,
    MAX_READ_BYTES,
    DataVolumeArchive,
    DataVolumeContent,
    DataVolumeEntry,
    DataVolumeEntryType,
    DataVolumeListing,
    DataVolumeRefusalCode,
    DataVolumeSkipped,
    DataVolumeSkipReason,
)

# The kernel's own limit on links followed while resolving one path.
_MAX_SYMLINKS = 40
# Confinement needs opening and examining names relative to a directory descriptor, and
# opening without following a link. A platform without them (Windows) serves no volume.
_OPEN_FLAGS = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
CONFINED_READS_SUPPORTED = (
    all(hasattr(os, flag) for flag in _OPEN_FLAGS)
    and {os.open, os.stat, os.readlink} <= os.supports_dir_fd
    and os.scandir in os.supports_fd
)
_O_DIRECTORY, _O_NOFOLLOW, _O_NONBLOCK, _O_CLOEXEC = (getattr(os, flag, 0) for flag in _OPEN_FLAGS)
_DIRECTORY_FLAGS = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
# Non-blocking, so a name swapped for a FIFO after it was examined can't hang the open.
_FILE_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC | _O_NONBLOCK
_COPY_BYTES = 1024 * 1024
_TOO_SLOW = "took longer to read than one archive may"


class DataVolumeRefusal(Exception):
    """A read the volume refuses or can't answer, with its HTTP status and stable code."""

    def __init__(self, status_code: int, code: DataVolumeRefusalCode, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def data_volume_from_environment(environ: Mapping[str, str]) -> DataVolume | None:
    """The volume ``CAYU_DATA_VOLUME`` names, or ``None`` when it isn't set."""

    root = environ.get(DATA_VOLUME_ENV)
    return None if not root else DataVolume(root)


@dataclass(frozen=True)
class _Opened:
    """A resolved entry: its open descriptor (``None`` for a special file) and its status."""

    fd: int | None
    info: os.stat_result


class DataVolume:
    """The directory ``root`` and everything beneath it, read without leaving it."""

    def __init__(self, root: str, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._root = root
        self._clock = clock

    def entry(self, parts: tuple[str, ...]) -> DataVolumeEntry:
        """The entry ``parts`` names, after resolving links within the volume."""

        opened = self._open(parts)
        try:
            return _entry(parts, opened.info)
        finally:
            if opened.fd is not None:
                os.close(opened.fd)

    def entries(
        self, parts: tuple[str, ...], *, limit: int, after: str | None
    ) -> DataVolumeListing:
        """Up to ``limit`` entries of the directory ``parts`` after the name ``after``."""

        deadline = self._deadline()
        fd = self._open_directory(parts)
        try:
            names, omitted, complete = _scan(fd)
            start = 0 if after is None else bisect.bisect_right(names, after)
            page = names[start : start + limit]
            listed: list[DataVolumeEntry] = []
            last: str | None = None
            stopped = False
            for name in page:
                if last is not None and self._clock() > deadline:
                    stopped = True
                    break
                last = name
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue  # Removed since the directory was read.
                except OSError as exc:
                    _unreadable((*parts, name), exc)
                listed.append(_entry((*parts, name), info))
            more = stopped or start + len(page) < len(names)
            return DataVolumeListing(
                path=_display(parts),
                entries=listed,
                next_after=last if more else None,
                complete=complete,
                omitted=omitted,
            )
        finally:
            os.close(fd)

    def content(self, parts: tuple[str, ...], *, offset: int, length: int) -> DataVolumeContent:
        """Up to ``length`` bytes of the regular file ``parts`` from ``offset``."""

        opened = self._open(parts)
        if opened.fd is None or not stat.S_ISREG(opened.info.st_mode):
            if opened.fd is not None:
                os.close(opened.fd)
            raise DataVolumeRefusal(
                409, "not_a_file", f"{_quoted(parts)} is not a file; list it, or get it whole."
            )
        try:
            info = os.fstat(opened.fd)
            wanted = max(0, min(length, MAX_READ_BYTES, info.st_size - offset))
            chunks: list[bytes] = []
            position = offset
            while wanted > 0:
                chunk = os.pread(opened.fd, wanted, position)
                if not chunk:
                    break
                chunks.append(chunk)
                position += len(chunk)
                wanted -= len(chunk)
        except OSError as exc:
            _unreadable(parts, exc)
        finally:
            os.close(opened.fd)
        return DataVolumeContent(
            path=_display(parts),
            size=info.st_size,
            modified_at=_time(info.st_mtime),
            offset=offset,
            content_base64=base64.b64encode(b"".join(chunks)).decode("ascii"),
        )

    def archive(self, parts: tuple[str, ...]) -> DataVolumeArchive:
        """The directory ``parts`` as a gzip-compressed tar archive, within every bound."""

        deadline = self._deadline()
        fd = self._open_directory(parts)
        top = parts[-1] if parts else DATA_VOLUME_ROOT_NAME
        buffer = _BoundedBuffer(MAX_ARCHIVE_BYTES)
        writer = _ArchiveWriter(parts, deadline=deadline, clock=self._clock)
        try:
            with tarfile.open(
                fileobj=buffer, mode="w:gz", compresslevel=6, format=tarfile.PAX_FORMAT
            ) as tar:
                writer.add_directory(tar, fd, top, os.fstat(fd), depth=0)
        except _ArchiveLimit as exc:
            raise DataVolumeRefusal(
                413,
                "archive_too_large",
                f"{_quoted(parts)} {exc}. Get a smaller directory, or its files one by one.",
            ) from None
        finally:
            os.close(fd)
        return DataVolumeArchive(
            path=_display(parts),
            name=f"{top}.tar.gz",
            files=writer.files,
            directories=writer.directories,
            bytes=writer.bytes,
            skipped=writer.skipped,
            skipped_count=writer.skipped_count,
            content_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )

    def _deadline(self) -> float:
        return self._clock() + DATA_VOLUME_DEADLINE_SECONDS

    def _open_directory(self, parts: tuple[str, ...]) -> int:
        opened = self._open(parts)
        if opened.fd is None or not stat.S_ISDIR(opened.info.st_mode):
            if opened.fd is not None:
                os.close(opened.fd)
            raise DataVolumeRefusal(409, "not_a_directory", f"{_quoted(parts)} is not a directory.")
        return opened.fd

    def _open_root(self) -> int:
        if not CONFINED_READS_SUPPORTED:
            raise DataVolumeRefusal(
                409,
                "data_volume_unavailable",
                "This platform can't confine reads to the data volume, so it serves none.",
            )
        if not os.path.isabs(self._root):
            raise DataVolumeRefusal(
                409,
                "data_volume_unavailable",
                f"{DATA_VOLUME_ENV} must name the data volume's directory by an absolute path.",
            )
        try:
            # The host names the volume, so a link in its own path is the host's choice.
            return os.open(self._root, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
        except OSError:
            raise DataVolumeRefusal(
                409,
                "data_volume_unavailable",
                f"The directory {DATA_VOLUME_ENV} names can't be opened, so the data volume "
                "isn't available to this process.",
            ) from None

    def _inside(self, target: str) -> list[str] | None:
        """The names an absolute link target has below the volume's directory, or ``None``."""

        for prefix in dict.fromkeys((posixpath.normpath(self._root), os.path.realpath(self._root))):
            if target == prefix or target.startswith(prefix.rstrip("/") + "/"):
                return target[len(prefix) :].split("/")
        return None

    def _open(self, parts: tuple[str, ...]) -> _Opened:
        """Resolve ``parts`` beneath the root, one name at a time, never leaving the volume."""

        chain = [self._open_root()]
        pending = list(reversed(parts))
        links = 0
        try:
            while pending:
                name = pending.pop()
                if name in {"", "."}:
                    continue
                if name == "..":
                    if len(chain) == 1:
                        _outside(parts)
                    os.close(chain.pop())
                    continue
                parent = chain[-1]
                try:
                    info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except (FileNotFoundError, NotADirectoryError):
                    raise DataVolumeRefusal(
                        404, "path_not_found", f"{_quoted(parts)} isn't in the data volume."
                    ) from None
                except OSError as exc:
                    _unreadable(parts, exc)
                if stat.S_ISLNK(info.st_mode):
                    links += 1
                    if links > _MAX_SYMLINKS:
                        raise DataVolumeRefusal(
                            422, "path_refused", f"{_quoted(parts)} goes through too many links."
                        )
                    try:
                        target = os.readlink(name, dir_fd=parent)
                    except OSError:
                        # Removed or replaced since it was examined.
                        _changed(parts)
                    if target.startswith("/"):
                        inside = self._inside(target)
                        if inside is None:
                            _outside(parts)
                        for fd in chain[1:]:
                            os.close(fd)
                        del chain[1:]
                        pending.extend(reversed(inside))
                    else:
                        pending.extend(reversed(target.split("/")))
                    continue
                if stat.S_ISDIR(info.st_mode):
                    chain.append(_open_at(name, parent, _DIRECTORY_FLAGS, info, parts))
                    continue
                if pending:
                    raise DataVolumeRefusal(
                        404, "path_not_found", f"{_quoted(parts)} isn't in the data volume."
                    )
                if stat.S_ISREG(info.st_mode):
                    return _Opened(_open_at(name, parent, _FILE_FLAGS, info, parts), info)
                return _Opened(None, info)
            # The path names a directory: the last one opened, the root for an empty path.
            fd = chain.pop()
            return _Opened(fd, os.fstat(fd))
        finally:
            for fd in chain:
                os.close(fd)


class _ArchiveLimit(Exception):
    """An archive would pass one of its bounds; the message says which."""


class _BoundedBuffer(io.BytesIO):
    """The compressed archive, refused once it would pass ``limit`` bytes.

    After the refusal it takes the rest of what the compressor flushes and discards it, so
    closing the abandoned archive can't raise a second time.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._refused = False

    def write(self, data, /) -> int:  # type: ignore[override]
        if self._refused:
            return len(data)
        if self.tell() + len(data) > self._limit:
            self._refused = True
            raise _ArchiveLimit(
                f"compresses to more than {_mib(self._limit)}, the most one archive holds"
            )
        return super().write(data)


class _ExactFile:
    """A regular file's first ``size`` bytes; a shorter file was changed while archived."""

    def __init__(
        self,
        fd: int,
        size: int,
        parts: tuple[str, ...],
        *,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._fd = fd
        self._remaining = size
        self._parts = parts
        self._deadline = deadline
        self._clock = clock

    def read(self, size: int = -1, /) -> bytes:
        """Exactly ``size`` bytes, or the rest; fewer only at the end of the archived size."""

        chunks: list[bytes] = []
        wanted = self._remaining if size < 0 else min(size, self._remaining)
        while wanted > 0:
            if self._clock() > self._deadline:
                raise _ArchiveLimit(_TOO_SLOW)
            data = os.read(self._fd, min(wanted, _COPY_BYTES))
            if not data:
                raise DataVolumeRefusal(
                    409,
                    "changed_while_read",
                    f"{_quoted(self._parts)} changed while it was archived; try again.",
                )
            chunks.append(data)
            wanted -= len(data)
            self._remaining -= len(data)
        return b"".join(chunks)


class _ArchiveWriter:
    """Adds a directory tree to a tar archive, one open directory per level."""

    def __init__(
        self, parts: tuple[str, ...], *, deadline: float, clock: Callable[[], float]
    ) -> None:
        self._parts = parts
        self._deadline = deadline
        self._clock = clock
        self._scanned = 0
        self.files = 0
        self.directories = 0
        self.bytes = 0
        self.skipped: list[DataVolumeSkipped] = []
        self.skipped_count = 0

    def add_directory(
        self, tar: tarfile.TarFile, fd: int, member: str, info: os.stat_result, *, depth: int
    ) -> None:
        tar.addfile(_tar_info(member, info, tarfile.DIRTYPE))
        self.directories += 1
        names: list[str] = []
        for name in _names(fd):
            self._scanned += 1
            if self._scanned > MAX_ARCHIVE_ENTRIES:
                raise _ArchiveLimit(f"holds more than {MAX_ARCHIVE_ENTRIES} entries")
            names.append(name)
        for name in sorted(names):
            if self._clock() > self._deadline:
                raise _ArchiveLimit(_TOO_SLOW)
            if not _utf8(name):
                # Named with its undecodable bytes escaped, so the answer stays valid text.
                self._skip(f"{member}/{_escaped(name)}", "unreadable_name")
                continue
            child = f"{member}/{name}"
            try:
                child_info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue  # Removed since the directory was read.
            except OSError as exc:
                _unreadable(self._parts, exc)
            if stat.S_ISLNK(child_info.st_mode):
                self._skip(child, "symlink")
            elif stat.S_ISDIR(child_info.st_mode):
                if depth + 1 > MAX_ARCHIVE_DEPTH:
                    raise _ArchiveLimit(f"nests directories more than {MAX_ARCHIVE_DEPTH} deep")
                child_fd = _open_at(name, fd, _DIRECTORY_FLAGS, child_info, self._parts)
                try:
                    self.add_directory(tar, child_fd, child, child_info, depth=depth + 1)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(child_info.st_mode):
                self._add_file(tar, fd, name, child, child_info)
            else:
                self._skip(child, "special_file")

    def _add_file(
        self, tar: tarfile.TarFile, parent: int, name: str, member: str, info: os.stat_result
    ) -> None:
        file_fd = _open_at(name, parent, _FILE_FLAGS, info, self._parts)
        try:
            opened = os.fstat(file_fd)
            if self.bytes + opened.st_size > MAX_ARCHIVE_INPUT_BYTES:
                raise _ArchiveLimit(f"holds more than {_mib(MAX_ARCHIVE_INPUT_BYTES)} of files")
            source = _ExactFile(
                file_fd,
                opened.st_size,
                self._parts,
                deadline=self._deadline,
                clock=self._clock,
            )
            tar.addfile(_tar_info(member, opened, tarfile.REGTYPE), source)
        finally:
            os.close(file_fd)
        self.files += 1
        self.bytes += opened.st_size

    def _skip(self, member: str, reason: DataVolumeSkipReason) -> None:
        self.skipped_count += 1
        if len(self.skipped) < MAX_ARCHIVE_SKIPPED_NAMED:
            self.skipped.append(DataVolumeSkipped(member=member, reason=reason))


def _tar_info(member: str, info: os.stat_result, kind: bytes) -> tarfile.TarInfo:
    entry = tarfile.TarInfo(member)
    entry.type = kind
    entry.size = info.st_size if kind == tarfile.REGTYPE else 0
    entry.mode = stat.S_IMODE(info.st_mode)
    entry.mtime = int(info.st_mtime)
    # The Agent's own user and group mean nothing where the archive is unpacked.
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = ""
    return entry


def _scan(fd: int) -> tuple[list[str], int, bool]:
    """A directory's UTF-8 names in order, how many others it left out, and whether it read
    them all (at most ``MAX_DIRECTORY_SCAN``)."""

    names: list[str] = []
    omitted = 0
    for name in _names(fd):
        if len(names) + omitted >= MAX_DIRECTORY_SCAN:
            return sorted(names), omitted, False
        if _utf8(name):
            names.append(name)
        else:
            omitted += 1
    return sorted(names), omitted, True


def _names(fd: int) -> Iterator[str]:
    """The names in the directory ``fd``, in the order the file system gives them."""

    with os.scandir(fd) as scan:
        for entry in scan:
            yield entry.name


def _open_at(
    name: str, parent: int, flags: int, expected: os.stat_result, parts: tuple[str, ...]
) -> int:
    """Open ``name`` in ``parent`` without following it, as the entry ``expected`` examined."""

    try:
        fd = os.open(name, flags, dir_fd=parent)
    except FileNotFoundError:
        raise DataVolumeRefusal(
            404, "path_not_found", f"{_quoted(parts)} isn't in the data volume."
        ) from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENXIO}:
            _changed(parts)
        _unreadable(parts, exc)
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
        os.close(fd)
        _changed(parts)
    return fd


def _entry(parts: tuple[str, ...], info: os.stat_result) -> DataVolumeEntry:
    kind: DataVolumeEntryType
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    else:
        kind = "other"
    return DataVolumeEntry(
        path=_display(parts),
        name=parts[-1] if parts else "",
        type=kind,
        size=info.st_size if kind == "file" else None,
        modified_at=_time(info.st_mtime),
    )


def _mib(size: int) -> str:
    return f"{size / (1024 * 1024):g} MiB"


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)


def _escaped(name: str) -> str:
    """``name`` with the bytes that aren't UTF-8 written as ``\\xNN``."""

    return os.fsencode(name).decode("utf-8", "backslashreplace")


def _utf8(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _display(parts: tuple[str, ...]) -> str:
    return "/".join(parts)


def _quoted(parts: tuple[str, ...]) -> str:
    return f"`{_display(parts)}`" if parts else "The data volume's root"


def _outside(parts: tuple[str, ...]) -> NoReturn:
    raise DataVolumeRefusal(
        422,
        "path_outside_volume",
        f"{_quoted(parts)} leads outside the data volume through a symbolic link, so it is "
        "refused.",
    )


def _changed(parts: tuple[str, ...]) -> NoReturn:
    raise DataVolumeRefusal(
        409, "changed_while_read", f"{_quoted(parts)} changed while it was read; try again."
    )


def _unreadable(parts: tuple[str, ...], error: OSError) -> NoReturn:
    reason = errno.errorcode.get(error.errno or 0, "an error")
    raise DataVolumeRefusal(
        409, "path_unreadable", f"{_quoted(parts)} can't be read ({reason})."
    ) from None

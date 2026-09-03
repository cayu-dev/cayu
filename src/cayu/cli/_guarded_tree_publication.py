"""Crash-aware publication of one complete CLI-owned directory tree."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
    exception_suppresses_context,
    iter_exception_tree,
)
from cayu._filesystem_lock import cooperative_path_lock

_JOURNAL_SCHEMA_VERSION = 3
_JOURNAL_LIMIT_BYTES = 64 * 1024
_PUBLICATION_METADATA_CENSUS_LIMIT = 1024
_PARENT_DIRECTORY_CENSUS_LIMIT = 16_384
_CLEANUP_MANIFEST_SCHEMA_VERSION = 2
_CLEANUP_MANIFEST_LIMIT_BYTES = 8 * 1024 * 1024
_TREE_ENTRY_LIMIT = 16_384
_TREE_DEPTH_LIMIT = 128
_OWNER_MARKER = ".cayu-tree-publication-owner"
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_WINDOWS_FILE_ATTRIBUTE_READONLY = 0x1
_WINDOWS_PRIVATE_DIRECTORY_SDDL = "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAME_STEMS = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{suffix}" for suffix in "123456789¹²³"),
        *(f"lpt{suffix}" for suffix in "123456789¹²³"),
    }
)
_LINUX_IOCTL_READ = 2
_LINUX_FS_IOC_GETFLAGS_TYPE = ord("f")
_LINUX_FS_IOC_GETFLAGS_NUMBER = 1
_LINUX_FS_IOC_GETVERSION_TYPE = ord("v")
_LINUX_FS_IOC_GETVERSION_NUMBER = 1
_LINUX_FS_IMMUTABLE_FL = 0x00000010
_LINUX_FS_APPEND_FL = 0x00000020
_LINUX_FS_CASEFOLD_FL = 0x40000000
_DARWIN_PC_CASE_SENSITIVE = 11
_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)
_SETTLEMENT_RECOVERY_NOTE_PREFIX = (
    "guarded publication remains recoverable; automatic settlement failed with "
)
_LINUX_AT_EMPTY_PATH = 0x1000
_LINUX_AT_SYMLINK_NOFOLLOW = 0x100
_LINUX_STATX_BTIME = 0x0800
_LINUX_STATX_INO = 0x0100
_LINUX_STATX_TYPE = 0x0001
_DARWIN_ATTR_BIT_MAP_COUNT = 5
_DARWIN_ATTR_VOL_INFO = 0x80000000
_DARWIN_ATTR_VOL_CAPABILITIES = 0x00020000
_DARWIN_VOL_CAP_FMT_PATH_FROM_ID = 0x00004000
_WINDOWS_FILE_BASIC_INFO_CLASS = 0
_WINDOWS_FILE_ID_INFO_CLASS = 18
_WINDOWS_FILE_READ_ATTRIBUTES = 0x0080
_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FSCTL_CREATE_OR_GET_OBJECT_ID = 0x000900C0
_PUBLICATION_METADATA_NAME_PATTERN = re.compile(
    r"\A\.cayu-tree-publication-(?P<collision_key>[0-9a-f]{32})-"
    r"(?P<canonical_key>[0-9a-f]{32})-(?P<raw_key>[0-9a-f]{32})"
    r"(?:\.jsonl(?:\.pending-[0-9a-f]{64})?|-receipt\.jsonl)\Z"
)


class _LinuxStatxTimestamp(ctypes.Structure):
    _fields_ = (
        ("seconds", ctypes.c_int64),
        ("nanoseconds", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    )


class _LinuxStatx(ctypes.Structure):
    _fields_ = (
        ("mask", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("link_count", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("spare_zero", ctypes.c_uint16),
        ("inode", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("access_time", _LinuxStatxTimestamp),
        ("birth_time", _LinuxStatxTimestamp),
        ("change_time", _LinuxStatxTimestamp),
        ("modify_time", _LinuxStatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mount_id", ctypes.c_uint64),
        ("direct_io_memory_alignment", ctypes.c_uint32),
        ("direct_io_offset_alignment", ctypes.c_uint32),
        ("spare", ctypes.c_uint64 * 12),
    )


class _DarwinAttrList(ctypes.Structure):
    _fields_ = (
        ("bitmap_count", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("common_attributes", ctypes.c_uint32),
        ("volume_attributes", ctypes.c_uint32),
        ("directory_attributes", ctypes.c_uint32),
        ("file_attributes", ctypes.c_uint32),
        ("fork_attributes", ctypes.c_uint32),
    )


class _DarwinVolumeCapabilities(ctypes.Structure):
    _fields_ = (
        ("capabilities", ctypes.c_uint32 * 4),
        ("valid", ctypes.c_uint32 * 4),
    )


class _DarwinVolumeCapabilitiesBuffer(ctypes.Structure):
    _fields_ = (
        ("length", ctypes.c_uint32),
        ("capabilities", _DarwinVolumeCapabilities),
    )


class _WindowsFileIdInfo(ctypes.Structure):
    _fields_ = (
        ("volume_serial_number", ctypes.c_uint64),
        ("file_id", ctypes.c_ubyte * 16),
    )


class _WindowsFileBasicInfo(ctypes.Structure):
    _fields_ = (
        ("creation_time", ctypes.c_int64),
        ("last_access_time", ctypes.c_int64),
        ("last_write_time", ctypes.c_int64),
        ("change_time", ctypes.c_int64),
        ("file_attributes", ctypes.c_uint32),
    )


class _WindowsObjectIdBuffer(ctypes.Structure):
    _fields_ = (
        ("object_id", ctypes.c_ubyte * 16),
        ("extended_info", ctypes.c_ubyte * 48),
    )


class DestinationPolicy(StrEnum):
    """Allowed state of the destination before a whole-tree publication."""

    ABSENT_OR_EMPTY = "absent_or_empty"
    REPLACE_DIRECTORY = "replace_directory"


class _DirectoryLookupSemantics(StrEnum):
    UNKNOWN = "unknown"
    CASE_SENSITIVE = "case_sensitive"
    UNICODE_NORMALIZED = "unicode_normalized"
    UNICODE_CASEFOLDED = "unicode_casefolded"


class GuardedTreePublicationError(RuntimeError):
    """A guarded publication could not prove a safe filesystem transition."""

    def __init__(self, code: str, message: str, *, paths: tuple[str, ...] = ()) -> None:
        self.code = code
        self.paths = tuple(path[:256] for path in paths[:8])
        super().__init__(message)


@dataclass(frozen=True)
class GuardedTreePublicationResult:
    destination: Path
    recovered: bool = False


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    kind: int
    incarnation: int | None = None

    @classmethod
    def capture(cls, value: os.stat_result) -> _Identity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            kind=stat.S_IFMT(value.st_mode),
        )

    def matches(self, value: os.stat_result) -> bool:
        """Match one live observation; durable callers compare complete identities."""

        return (
            self.device == int(value.st_dev)
            and self.inode == int(value.st_ino)
            and self.kind == stat.S_IFMT(value.st_mode)
        )

    def as_json(self) -> list[int]:
        if self.incarnation is None:
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "durable filesystem authority lacks stable incarnation evidence",
            )
        return [self.device, self.inode, self.kind, self.incarnation]

    @classmethod
    def from_json(cls, value: object, *, field: str) -> _Identity:
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(type(item) is not int or item < 0 for item in value)
        ):
            raise _invalid_journal(f"invalid {field} identity")
        return cls(
            device=cast("int", value[0]),
            inode=cast("int", value[1]),
            kind=cast("int", value[2]),
            incarnation=cast("int", value[3]),
        )


@dataclass(frozen=True)
class _FileMutationObservation:
    """One non-durable observation used to reject writes during inspection."""

    size: int
    mode: int
    modified_ns: int
    changed_token: int
    windows_attributes: int

    @classmethod
    def capture(
        cls,
        value: os.stat_result,
        *,
        changed_token: int | None = None,
    ) -> _FileMutationObservation:
        return cls(
            size=int(value.st_size),
            mode=stat.S_IMODE(value.st_mode),
            modified_ns=int(value.st_mtime_ns),
            changed_token=(int(value.st_ctime_ns) if changed_token is None else changed_token),
            windows_attributes=int(getattr(value, "st_file_attributes", 0)),
        )


def _capture_stable_identity(
    value: os.stat_result,
    *,
    path: Path | None = None,
    descriptor: int | None = None,
    dir_fd: int | None = None,
    name: str | None = None,
) -> _Identity:
    if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
        raise GuardedTreePublicationError(
            "unsafe_entry",
            "stable identity cannot be captured from a link or reparse point",
            paths=(() if path is None else (path.name,)),
        )
    if not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)):
        raise GuardedTreePublicationError(
            "unsupported_entry",
            "stable identity requires an ordinary file or directory",
            paths=(() if path is None else (path.name,)),
        )
    identity = _Identity.capture(value)
    if sys.platform.startswith("linux"):
        incarnation = _linux_incarnation(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
    elif sys.platform == "darwin":
        incarnation = _darwin_incarnation(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
    elif os.name == "nt":
        incarnation = _windows_incarnation(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
    else:
        generation = getattr(value, "st_gen", None)
        if type(generation) is not int or generation <= 0:
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "the filesystem does not expose a non-reusable object generation",
                paths=(() if path is None else (path.name,)),
            )
        incarnation = generation
    if incarnation < 0:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the filesystem returned invalid object-incarnation evidence",
            paths=(() if path is None else (path.name,)),
        )
    return replace(identity, incarnation=incarnation)


def _linux_incarnation(
    expected: os.stat_result,
    *,
    path: Path | None,
    descriptor: int | None,
    dir_fd: int | None,
    name: str | None,
) -> int:
    pinned, owned = _pin_identity_descriptor(
        expected,
        path=path,
        descriptor=descriptor,
        dir_fd=dir_fd,
        name=name,
    )
    try:
        birth_time_ns = _linux_birth_time_ns(expected, descriptor=pinned, path=path)
        generation = _linux_inode_generation(pinned, path=path)
    finally:
        if owned:
            _close_descriptor(pinned)
    # Keep creation time as independent supporting evidence. The generation,
    # rather than the timestamp, is the positive proof against inode reuse.
    return (generation << 128) | birth_time_ns


def _linux_birth_time_ns(
    expected: os.stat_result,
    *,
    descriptor: int,
    path: Path | None,
) -> int:
    lookup_name = b""
    flags = _LINUX_AT_EMPTY_PATH | _LINUX_AT_SYMLINK_NOFOLLOW

    _libc, statx = _linux_statx_binding()

    observed = _LinuxStatx()
    requested = _LINUX_STATX_TYPE | _LINUX_STATX_INO | _LINUX_STATX_BTIME
    if (
        statx(
            descriptor,
            lookup_name,
            flags,
            requested,
            ctypes.byref(observed),
        )
        != 0
    ):
        error_code = ctypes.get_errno()
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Linux statx could not capture stable filesystem identity",
            paths=(() if path is None else (path.name,)),
        ) from OSError(error_code, os.strerror(error_code))
    if not observed.mask & _LINUX_STATX_BTIME:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the Linux filesystem does not expose object birth time",
            paths=(() if path is None else (path.name,)),
        )
    observed_device = os.makedev(observed.dev_major, observed.dev_minor)
    if (
        observed_device != int(expected.st_dev)
        or observed.inode != int(expected.st_ino)
        or stat.S_IFMT(observed.mode) != stat.S_IFMT(expected.st_mode)
    ):
        raise GuardedTreePublicationError(
            "identity_changed",
            "filesystem identity changed while its incarnation was captured",
            paths=(() if path is None else (path.name,)),
        )
    if observed.birth_time.seconds < 0 or observed.birth_time.nanoseconds >= 1_000_000_000:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the Linux filesystem returned invalid object birth time",
        )
    return observed.birth_time.seconds * 1_000_000_000 + observed.birth_time.nanoseconds


def _linux_inode_generation(descriptor: int, *, path: Path | None) -> int:
    import array
    import fcntl

    generation = array.array("L", [0])
    ioctl_request = (
        (_LINUX_IOCTL_READ << 30)
        | (generation.itemsize << 16)
        | (_LINUX_FS_IOC_GETVERSION_TYPE << 8)
        | _LINUX_FS_IOC_GETVERSION_NUMBER
    )
    try:
        fcntl.ioctl(descriptor, ioctl_request, generation, True)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the Linux filesystem does not expose an object generation",
            paths=(() if path is None else (path.name,)),
        ) from exc
    if generation[0] == 0:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the Linux filesystem returned no object generation",
            paths=(() if path is None else (path.name,)),
        )
    return generation[0]


def _darwin_incarnation(
    expected: os.stat_result,
    *,
    path: Path | None,
    descriptor: int | None,
    dir_fd: int | None,
    name: str | None,
) -> int:
    pinned, owned = _pin_identity_descriptor(
        expected,
        path=path,
        descriptor=descriptor,
        dir_fd=dir_fd,
        name=name,
    )
    try:
        if _darwin_volume_has_nonrecycled_object_ids(pinned, path=path):
            # On such a volume (including APFS), device + inode already forms
            # the non-recycled incarnation. Zero records that positive mode.
            return 0
        observed = os.fstat(pinned)
        generation = getattr(observed, "st_gen", None)
        if type(generation) is int and generation > 0:
            return generation
    finally:
        if owned:
            _close_descriptor(pinned)
    raise GuardedTreePublicationError(
        "stable_identity_unavailable",
        "the Darwin filesystem exposes neither non-recycled IDs nor an object generation",
        paths=(() if path is None else (path.name,)),
    )


def _darwin_volume_has_nonrecycled_object_ids(
    descriptor: int,
    *,
    path: Path | None,
) -> bool:
    _libc, fgetattrlist = _darwin_fgetattrlist_binding()
    attributes = _DarwinAttrList(
        bitmap_count=_DARWIN_ATTR_BIT_MAP_COUNT,
        volume_attributes=_DARWIN_ATTR_VOL_INFO | _DARWIN_ATTR_VOL_CAPABILITIES,
    )
    result = _DarwinVolumeCapabilitiesBuffer()
    if (
        fgetattrlist(
            descriptor,
            ctypes.byref(attributes),
            ctypes.byref(result),
            ctypes.sizeof(result),
            0,
        )
        != 0
    ):
        error_code = ctypes.get_errno()
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Darwin volume capabilities could not be authenticated",
            paths=(() if path is None else (path.name,)),
        ) from OSError(error_code, os.strerror(error_code))
    if result.length < ctypes.sizeof(result):
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Darwin returned incomplete volume capabilities",
            paths=(() if path is None else (path.name,)),
        )
    valid = result.capabilities.valid[0]
    supported = result.capabilities.capabilities[0]
    return bool(
        valid & _DARWIN_VOL_CAP_FMT_PATH_FROM_ID and supported & _DARWIN_VOL_CAP_FMT_PATH_FROM_ID
    )


def _windows_incarnation(
    expected: os.stat_result,
    *,
    path: Path | None,
    descriptor: int | None,
    dir_fd: int | None,
    name: str | None,
) -> int:
    if dir_fd is not None or name is not None:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Windows stable identity capture requires a descriptor or exact path",
        )
    owned = False
    if descriptor is not None:
        import msvcrt

        windows_msvcrt: Any = msvcrt
        handle = windows_msvcrt.get_osfhandle(descriptor)
    elif path is not None:
        _kernel32, create_file, close_handle, _get_file_id = _windows_file_id_bindings()
        handle = create_file(
            str(path),
            _WINDOWS_FILE_READ_ATTRIBUTES,
            _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE,
            None,
            _WINDOWS_OPEN_EXISTING,
            _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            error_code = _windows_last_error()
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "Windows could not open the filesystem object for identity capture",
                paths=(path.name,),
            ) from OSError(error_code, os.strerror(error_code))
        owned = True
    else:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Windows stable identity capture requires a descriptor or exact path",
        )

    _kernel32, _create_file, close_handle, get_file_id = _windows_file_id_bindings()
    try:
        info = _WindowsFileIdInfo()
        if not get_file_id(
            handle,
            _WINDOWS_FILE_ID_INFO_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            error_code = _windows_last_error()
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "Windows could not capture the persistent file ID",
                paths=(() if path is None else (path.name,)),
            ) from OSError(error_code, os.strerror(error_code))
        if path is not None:
            current = path.stat(follow_symlinks=False)
        else:
            current = os.fstat(cast("int", descriptor))
        if not _Identity.capture(expected).matches(current):
            raise GuardedTreePublicationError(
                "identity_changed",
                "filesystem identity changed while its incarnation was captured",
                paths=(() if path is None else (path.name,)),
            )
        object_id = _windows_object_id(handle, path=path)
        incarnation = (int(info.volume_serial_number) << 128) | object_id
        if object_id == 0:
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "Windows returned no persistent object identity",
                paths=(() if path is None else (path.name,)),
            )
        return incarnation
    finally:
        if owned and not close_handle(handle):
            error_code = _windows_last_error()
            primary = sys.exception()
            close_error = OSError(error_code, os.strerror(error_code))
            if primary is None:
                raise close_error
            _raise_primary_with_secondary_failure(
                primary,
                close_error,
                group_message="Identity capture and Windows handle cleanup failures.",
            )


def _windows_object_id(handle: int, *, path: Path | None) -> int:
    device_io_control = _windows_object_id_binding()
    result = _WindowsObjectIdBuffer()
    returned = ctypes.c_uint32()
    if not device_io_control(
        handle,
        _WINDOWS_FSCTL_CREATE_OR_GET_OBJECT_ID,
        None,
        0,
        ctypes.byref(result),
        ctypes.sizeof(result),
        ctypes.byref(returned),
        None,
    ):
        error_code = _windows_last_error()
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "the Windows filesystem does not provide persistent object identities",
            paths=(() if path is None else (path.name,)),
        ) from OSError(error_code, os.strerror(error_code))
    if returned.value != ctypes.sizeof(result):
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Windows returned incomplete persistent object identity evidence",
            paths=(() if path is None else (path.name,)),
        )
    return int.from_bytes(bytes(result.object_id), byteorder="little")


def _pin_identity_descriptor(
    expected: os.stat_result,
    *,
    path: Path | None,
    descriptor: int | None,
    dir_fd: int | None,
    name: str | None,
) -> tuple[int, bool]:
    if descriptor is not None:
        pinned = descriptor
        owned = False
    else:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        if stat.S_ISDIR(expected.st_mode):
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            if dir_fd is not None and name is not None:
                pinned = os.open(name, flags, dir_fd=dir_fd)
            elif path is not None:
                pinned = os.open(path, flags)
            else:
                raise GuardedTreePublicationError(
                    "stable_identity_unavailable",
                    "stable identity capture requires a pinned descriptor or exact path",
                )
        except GuardedTreePublicationError:
            raise
        except OSError as exc:
            code = (
                "identity_changed"
                if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}
                else "stable_identity_unavailable"
            )
            raise GuardedTreePublicationError(
                code,
                "filesystem identity could not be pinned for stable inspection",
                paths=(() if path is None else (path.name,)),
            ) from exc
        owned = True
    try:
        if not _Identity.capture(expected).matches(os.fstat(pinned)):
            raise GuardedTreePublicationError(
                "identity_changed",
                "filesystem identity changed while its incarnation was captured",
                paths=(() if path is None else (path.name,)),
            )
    except BaseException:
        if owned:
            _close_descriptor(pinned)
        raise
    return pinned, owned


def _windows_last_error() -> int:
    windows_ctypes: Any = ctypes
    return int(windows_ctypes.get_last_error())


def _capture_windows_file_mutation_observation(
    value: os.stat_result,
    *,
    path: Path | None = None,
    descriptor: int | None = None,
    handle: int | None = None,
) -> _FileMutationObservation:
    return _FileMutationObservation.capture(
        value,
        changed_token=_windows_file_change_time(
            path=path,
            descriptor=descriptor,
            handle=handle,
        ),
    )


def _windows_file_change_time(
    *,
    path: Path | None = None,
    descriptor: int | None = None,
    handle: int | None = None,
) -> int:
    supplied = sum(value is not None for value in (path, descriptor, handle))
    if supplied != 1:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Windows change-time inspection requires one pinned object authority",
        )

    owned = False
    _kernel32, create_file, close_handle, get_file_info = _windows_file_id_bindings()
    if descriptor is not None:
        import msvcrt

        windows_msvcrt: Any = msvcrt
        pinned_handle = windows_msvcrt.get_osfhandle(descriptor)
    elif handle is not None:
        pinned_handle = handle
    else:
        assert path is not None
        pinned_handle = create_file(
            str(path),
            _WINDOWS_FILE_READ_ATTRIBUTES,
            _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE,
            None,
            _WINDOWS_OPEN_EXISTING,
            _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if pinned_handle == ctypes.c_void_p(-1).value:
            error_code = _windows_last_error()
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "Windows could not pin a file for change-time inspection",
                paths=(path.name,),
            ) from OSError(error_code, os.strerror(error_code))
        owned = True

    try:
        info = _WindowsFileBasicInfo()
        if not get_file_info(
            pinned_handle,
            _WINDOWS_FILE_BASIC_INFO_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            error_code = _windows_last_error()
            raise GuardedTreePublicationError(
                "stable_identity_unavailable",
                "Windows could not capture a file change-time observation",
                paths=(() if path is None else (path.name,)),
            ) from OSError(error_code, os.strerror(error_code))
        return int(info.change_time)
    finally:
        if owned and not close_handle(pinned_handle):
            error_code = _windows_last_error()
            primary = sys.exception()
            close_error = OSError(error_code, os.strerror(error_code))
            if primary is None:
                raise close_error
            _raise_primary_with_secondary_failure(
                primary,
                close_error,
                group_message="Change-time inspection and Windows handle cleanup failures.",
            )


@cache
def _linux_statx_binding() -> tuple[Any, Any]:
    libc = ctypes.CDLL(None, use_errno=True)
    statx = getattr(libc, "statx", None)
    if statx is None:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Linux statx birth-time identity is unavailable",
        )
    statx.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_LinuxStatx),
    )
    statx.restype = ctypes.c_int
    return libc, statx


@cache
def _darwin_fgetattrlist_binding() -> tuple[Any, Any]:
    libc = ctypes.CDLL(None, use_errno=True)
    fgetattrlist = getattr(libc, "fgetattrlist", None)
    if fgetattrlist is None:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Darwin volume-capability inspection is unavailable",
        )
    fgetattrlist.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(_DarwinAttrList),
        ctypes.POINTER(_DarwinVolumeCapabilitiesBuffer),
        ctypes.c_size_t,
        ctypes.c_ulong,
    )
    fgetattrlist.restype = ctypes.c_int
    return libc, fgetattrlist


@cache
def _windows_kernel32_binding() -> Any:
    windows_ctypes: Any = ctypes
    win_dll = getattr(windows_ctypes, "WinDLL", None)
    if win_dll is None:
        raise GuardedTreePublicationError(
            "stable_identity_unavailable",
            "Windows native filesystem inspection is unavailable",
        )
    return win_dll("kernel32", use_last_error=True)


@cache
def _windows_file_id_bindings() -> tuple[Any, Any, Any, Any]:
    kernel32 = _windows_kernel32_binding()
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    get_file_id = kernel32.GetFileInformationByHandleEx
    get_file_id.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    get_file_id.restype = ctypes.c_int
    return kernel32, create_file, close_handle, get_file_id


@cache
def _windows_object_id_binding() -> Any:
    device_io_control = _windows_kernel32_binding().DeviceIoControl
    device_io_control.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    )
    device_io_control.restype = ctypes.c_int
    return device_io_control


@dataclass(frozen=True)
class _PreparedStageFile:
    path: PurePosixPath
    content: bytes
    mode: int | None


@dataclass(frozen=True)
class _PreparedStageDirectory:
    path: PurePosixPath
    mode: int | None


@dataclass(frozen=True)
class GuardedTreeStage:
    """An exact stage authority with guarded, bounded population methods."""

    _path: Path
    _publication_identity: _Identity
    _original_destination_mode: int | None

    @classmethod
    def _owned(
        cls,
        path: Path,
        *,
        expected: _Identity,
        original_destination_mode: int | None,
    ) -> GuardedTreeStage:
        return cls(
            _path=path,
            _publication_identity=expected,
            _original_destination_mode=original_destination_mode,
        )

    def publication_root_mode(self) -> int:
        """Return the original root mode or a safely observed creation-umask mode."""

        if self._original_destination_mode is not None:
            _validate_stage_mode(self._original_destination_mode, directory=True)
            return self._original_destination_mode
        return _capture_stage_default_directory_mode(
            self._path,
            expected=self._publication_identity,
        )

    def write_files(
        self,
        contents: Mapping[str, bytes],
        *,
        file_mode: int = 0o644,
        directory_mode: int = 0o755,
    ) -> None:
        """Populate an initially empty stage without reopening it by pathname."""

        self.write_tree(
            contents,
            file_mode=file_mode,
            directory_mode=directory_mode,
        )

    def write_tree(
        self,
        contents: Mapping[str, bytes],
        *,
        directories: Iterable[str] = (),
        file_modes: Mapping[str, int] | None = None,
        file_mode: int | None = 0o644,
        directory_mode: int | None = 0o755,
        root_mode: int | None = None,
    ) -> None:
        """Populate a complete tree; ``None`` modes retain creation-umask behavior."""

        if root_mode is None and directory_mode is None:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage root mode must be explicit when directory modes use the umask",
            )
        effective_root_mode = directory_mode if root_mode is None else root_mode
        assert effective_root_mode is not None
        _validate_stage_mode(effective_root_mode, directory=True)
        files, prepared_directories = _prepare_stage_files(
            contents,
            directories=directories,
            file_modes=file_modes,
            file_mode=file_mode,
            directory_mode=directory_mode,
        )
        _write_stage_files(
            self._path,
            expected=self._publication_identity,
            files=files,
            directories=prepared_directories,
            root_mode=effective_root_mode,
        )

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        encoding: str = "utf-8",
        file_mode: int = 0o644,
        directory_mode: int = 0o755,
    ) -> None:
        """Encode and publish one text file through the guarded stage writer."""

        self.write_files(
            {relative_path: content.encode(encoding)},
            file_mode=file_mode,
            directory_mode=directory_mode,
        )

    def capture_owned_identity(self) -> os.stat_result:
        """Authenticate the current path against the publisher-owned stage."""

        try:
            _reject_link_components(self._path)
            current = self._path.stat(follow_symlinks=False)
        except (GuardedTreePublicationError, OSError) as exc:
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed before population",
                paths=(self._path.name,),
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or _capture_stable_identity(current, path=self._path) != self._publication_identity
        ):
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed before population",
                paths=(self._path.name,),
            )
        return current

    def _specialized_path(self) -> Path:
        """Return the path only to a consumer that separately pins every write."""

        self.capture_owned_identity()
        return self._path


def validate_guarded_tree_files(
    contents: Mapping[str, bytes],
    *,
    file_mode: int = 0o644,
    directory_mode: int = 0o755,
) -> None:
    """Validate that one file tree can be populated and recovered within bounds."""

    _prepare_stage_files(
        contents,
        directories=(),
        file_modes=None,
        file_mode=file_mode,
        directory_mode=directory_mode,
    )


class _Phase(StrEnum):
    PREPARED = "prepared"
    STAGING = "staging"
    STAGED = "staged"
    COMMIT_INTENT = "commit_intent"
    ORIGINAL_BACKED_UP = "original_backed_up"
    PUBLISHED = "published"
    CLEANUP_OWNED = "cleanup_owned"
    CLEANUP_SEALED = "cleanup_sealed"
    ROLLBACK_CLEANUP_OWNED = "rollback_cleanup_owned"
    ROLLBACK_CLEANUP_SEALED = "rollback_cleanup_sealed"
    SETTLED = "settled"


_ALLOWED_PHASE_TRANSITIONS = {
    _Phase.PREPARED: frozenset({_Phase.STAGING, _Phase.ROLLBACK_CLEANUP_OWNED}),
    _Phase.STAGING: frozenset({_Phase.STAGED, _Phase.ROLLBACK_CLEANUP_OWNED}),
    _Phase.STAGED: frozenset({_Phase.COMMIT_INTENT, _Phase.ROLLBACK_CLEANUP_OWNED}),
    _Phase.COMMIT_INTENT: frozenset(
        {
            _Phase.ORIGINAL_BACKED_UP,
            _Phase.PUBLISHED,
            _Phase.CLEANUP_OWNED,
            _Phase.ROLLBACK_CLEANUP_OWNED,
        }
    ),
    _Phase.ORIGINAL_BACKED_UP: frozenset(
        {
            _Phase.PUBLISHED,
            _Phase.CLEANUP_OWNED,
            _Phase.ROLLBACK_CLEANUP_OWNED,
        }
    ),
    _Phase.PUBLISHED: frozenset({_Phase.CLEANUP_OWNED, _Phase.SETTLED}),
    _Phase.CLEANUP_OWNED: frozenset({_Phase.CLEANUP_SEALED}),
    _Phase.CLEANUP_SEALED: frozenset({_Phase.SETTLED}),
    _Phase.ROLLBACK_CLEANUP_OWNED: frozenset({_Phase.ROLLBACK_CLEANUP_SEALED}),
    _Phase.ROLLBACK_CLEANUP_SEALED: frozenset(),
    _Phase.SETTLED: frozenset(),
}


@dataclass(frozen=True)
class _CleanupEntry:
    path: str
    identity: _Identity
    mode: int
    size: int | None
    content_sha256: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "identity": self.identity.as_json(),
            "mode": self.mode,
            "size": self.size,
            "content_sha256": self.content_sha256,
        }


@dataclass
class _TreeEntryBudget:
    remaining: int

    def reserve(self) -> None:
        if self.remaining <= 0:
            raise GuardedTreePublicationError(
                "tree_limit",
                "guarded publication tree exceeds its bounded entry limit",
            )
        self.remaining -= 1


@dataclass(frozen=True)
class _CapturedCleanupEntry:
    entry: _CleanupEntry
    file_observation: _FileMutationObservation | None


@dataclass(frozen=True)
class _CleanupManifest:
    path: Path
    identity: _Identity
    content_sha256: str
    token: str
    cleanup_name: str
    root_identity: _Identity
    root_sha256: str
    entries: tuple[_CleanupEntry, ...]


@dataclass(frozen=True)
class _Record:
    consumer: str
    request_digest: str
    predecessor_request_digest: str | None
    predecessor_receipt_identity: _Identity | None
    predecessor_receipt_sha256: str | None
    token: str
    destination_name: str
    policy: DestinationPolicy
    parent_identity: _Identity
    original_identity: _Identity | None
    original_sha256: str | None
    stage_name: str
    stage_identity: _Identity | None
    stage_sha256: str | None
    backup_name: str
    cleanup_manifest_identity: _Identity | None
    cleanup_manifest_sha256: str | None
    phase: _Phase
    sequence: int = 0

    def payload(self, *, previous_sha256: str | None) -> dict[str, object]:
        return {
            "schema_version": _JOURNAL_SCHEMA_VERSION,
            "sequence": self.sequence,
            "previous_sha256": previous_sha256,
            "consumer": self.consumer,
            "request_digest": self.request_digest,
            "predecessor_request_digest": self.predecessor_request_digest,
            "predecessor_receipt_identity": (
                None
                if self.predecessor_receipt_identity is None
                else self.predecessor_receipt_identity.as_json()
            ),
            "predecessor_receipt_sha256": self.predecessor_receipt_sha256,
            "token": self.token,
            "destination_name": self.destination_name,
            "policy": self.policy.value,
            "parent_identity": self.parent_identity.as_json(),
            "original_identity": (
                None if self.original_identity is None else self.original_identity.as_json()
            ),
            "original_sha256": self.original_sha256,
            "stage_name": self.stage_name,
            "stage_identity": (
                None if self.stage_identity is None else self.stage_identity.as_json()
            ),
            "stage_sha256": self.stage_sha256,
            "backup_name": self.backup_name,
            "cleanup_manifest_identity": (
                None
                if self.cleanup_manifest_identity is None
                else self.cleanup_manifest_identity.as_json()
            ),
            "cleanup_manifest_sha256": self.cleanup_manifest_sha256,
            "phase": self.phase.value,
        }


@dataclass
class _Journal:
    path: Path
    identity: _Identity
    record: _Record
    entry_sha256: str
    valid_bytes: int


@dataclass(frozen=True)
class _Parent:
    path: Path
    identity: _Identity
    descriptor: int | None

    def entry_stat(self, name: str) -> os.stat_result | None:
        try:
            if self.descriptor is None:
                return (self.path / name).stat(follow_symlinks=False)
            return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GuardedTreePublicationError(
                "entry_inspection_failed",
                "could not inspect a guarded publication entry",
                paths=(name,),
            ) from exc

    def entry_identity(self, name: str, *, value: os.stat_result) -> _Identity:
        if self.descriptor is None:
            return _capture_stable_identity(value, path=self.path / name)
        return _capture_stable_identity(
            value,
            dir_fd=self.descriptor,
            name=name,
        )

    def assert_unchanged(self) -> None:
        try:
            _reject_link_components(self.path)
            current = self.path.stat(follow_symlinks=False)
        except (GuardedTreePublicationError, OSError) as exc:
            raise GuardedTreePublicationError(
                "parent_changed",
                "publication parent changed during the guarded operation",
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or _capture_stable_identity(current, path=self.path) != self.identity
        ):
            raise GuardedTreePublicationError(
                "parent_changed",
                "publication parent changed during the guarded operation",
            )
        if self.descriptor is not None:
            opened = os.fstat(self.descriptor)
            if _capture_stable_identity(opened, descriptor=self.descriptor) != self.identity:
                raise GuardedTreePublicationError(
                    "parent_changed",
                    "publication parent descriptor no longer matches its path",
                )

    def sync(self) -> None:
        self.assert_unchanged()
        if self.descriptor is not None:
            os.fsync(self.descriptor)
            return
        _sync_windows_path(self.path, directory=True)


def publish_guarded_tree(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    policy: DestinationPolicy,
    populate: Callable[[GuardedTreeStage], None],
    predecessor_request_digest: str | None = None,
    settle_active_operation: bool = False,
) -> GuardedTreePublicationResult:
    """Populate one tree, optionally settling recoverable active work first."""

    return _publish_guarded_tree(
        destination,
        consumer=consumer,
        request_digest=request_digest,
        policy=policy,
        populate=populate,
        predecessor_request_digest=predecessor_request_digest,
        bind_current_replacement=False,
        settle_active_operation=settle_active_operation,
    )


def replace_guarded_tree(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    populate: Callable[[GuardedTreeStage], None],
) -> GuardedTreePublicationResult:
    """Replace one tree while atomically binding its current terminal receipt."""

    return _publish_guarded_tree(
        destination,
        consumer=consumer,
        request_digest=request_digest,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=populate,
        predecessor_request_digest=None,
        bind_current_replacement=True,
        settle_active_operation=False,
    )


def _publish_guarded_tree(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    policy: DestinationPolicy,
    populate: Callable[[GuardedTreeStage], None],
    predecessor_request_digest: str | None,
    bind_current_replacement: bool,
    settle_active_operation: bool,
) -> GuardedTreePublicationResult:
    """Run one guarded publication through the shared destination owner."""

    destination = Path(destination)
    _validate_publication_input(
        destination,
        consumer=consumer,
        request_digest=request_digest,
        policy=policy,
        populate=populate,
        predecessor_request_digest=predecessor_request_digest,
        settle_active_operation=settle_active_operation,
    )
    parent_path = destination.parent
    expected_parent = _capture_parent(parent_path)
    result: GuardedTreePublicationResult | None = None
    try:
        with (
            cooperative_path_lock(
                parent_path,
                destination.name,
                lock_directory_name="cayu-tree-publication-locks-v1",
            ),
            _pinned_parent(parent_path, expected=expected_parent) as parent,
        ):
            metadata_stem = _resolve_destination_metadata_stem(parent, destination.name)
            if bind_current_replacement:
                predecessor_request_digest = _current_replacement_predecessor(
                    destination,
                    consumer=consumer,
                    request_digest=request_digest,
                    parent=parent,
                    metadata_stem=metadata_stem,
                )
            result = _publish_guarded_tree_owned(
                destination,
                consumer=consumer,
                request_digest=request_digest,
                policy=policy,
                populate=populate,
                predecessor_request_digest=predecessor_request_digest,
                settle_active_operation=settle_active_operation,
                parent=parent,
                metadata_stem=metadata_stem,
            )
    except BaseException as boundary_cleanup_error:
        if result is None:
            raise
        if not isinstance(boundary_cleanup_error, Exception):
            raise
        raise GuardedTreePublicationError(
            "boundary_cleanup_failed",
            "publication committed, but its ownership boundary did not settle cleanly; "
            "an exact retry is safe",
            paths=(destination.name,),
        ) from boundary_cleanup_error
    if result is None:
        raise AssertionError("guarded publication exited without a result")
    return result


def _current_replacement_predecessor(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    parent: _Parent,
    metadata_stem: str,
) -> str | None:
    """Resolve exact replay or successor authority while the destination is locked."""

    journal_path = parent.path / f".cayu-tree-publication-{metadata_stem}.jsonl"
    receipt_path = parent.path / (f".cayu-tree-publication-{metadata_stem}-receipt.jsonl")
    if parent.entry_stat(journal_path.name) is not None:
        journal = _load_journal(journal_path, parent=parent)
        if journal.record.destination_name != destination.name:
            raise _invalid_journal("publication journal does not belong to this destination")
        _require_parent_namespace_mutable(parent)
        outcome = (
            "rolled_back"
            if _retire_stale_settled_publication_if_safe(journal, parent=parent)
            else _recover(journal, parent=parent)
        )
        if outcome == "published":
            _promote_terminal_receipt(journal, receipt_path=receipt_path, parent=parent)

    if parent.entry_stat(receipt_path.name) is None:
        return None
    receipt = _load_journal(receipt_path, parent=parent)
    if receipt.record.destination_name != destination.name:
        raise _invalid_journal("publication receipt does not belong to this destination")
    if receipt.record.phase is not _Phase.SETTLED:
        raise _invalid_journal("publication receipt is not terminal")
    if _retire_stale_settled_publication_if_safe(receipt, parent=parent):
        return None
    if (
        receipt.record.consumer == consumer
        and receipt.record.request_digest == request_digest
        and receipt.record.policy is DestinationPolicy.REPLACE_DIRECTORY
    ):
        return receipt.record.predecessor_request_digest
    return receipt.record.request_digest


def _publish_guarded_tree_owned(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    policy: DestinationPolicy,
    populate: Callable[[GuardedTreeStage], None],
    predecessor_request_digest: str | None,
    settle_active_operation: bool,
    parent: _Parent,
    metadata_stem: str,
) -> GuardedTreePublicationResult:
    _reject_case_alias(parent, destination.name)
    journal_path = parent.path / f".cayu-tree-publication-{metadata_stem}.jsonl"
    receipt_path = parent.path / f".cayu-tree-publication-{metadata_stem}-receipt.jsonl"
    predecessor_receipt_identity: _Identity | None = None
    predecessor_receipt_sha256: str | None = None
    if parent.entry_stat(journal_path.name) is not None:
        journal = _load_journal(journal_path, parent=parent)
        if journal.record.destination_name != destination.name:
            raise _invalid_journal("publication journal does not belong to this destination")
        same_request = (
            journal.record.consumer == consumer
            and journal.record.request_digest == request_digest
            and journal.record.predecessor_request_digest == predecessor_request_digest
            and journal.record.policy is policy
        )
        _require_parent_namespace_mutable(parent)
        retired = _retire_stale_settled_publication_if_safe(journal, parent=parent)
        if not retired and not same_request and not settle_active_operation:
            raise GuardedTreePublicationError(
                "publication_request_conflict",
                "an active publication belongs to a different exact request",
                paths=(destination.name,),
            )
        outcome = "rolled_back" if retired else _recover(journal, parent=parent)
        if outcome == "published":
            _promote_terminal_receipt(journal, receipt_path=receipt_path, parent=parent)

    if parent.entry_stat(receipt_path.name) is not None:
        receipt = _load_journal(receipt_path, parent=parent)
        if receipt.record.destination_name != destination.name:
            raise _invalid_journal("publication receipt does not belong to this destination")
        if receipt.record.phase is not _Phase.SETTLED:
            raise _invalid_journal("publication receipt is not terminal")
        same_request = (
            receipt.record.consumer == consumer
            and receipt.record.request_digest == request_digest
            and receipt.record.predecessor_request_digest == predecessor_request_digest
            and receipt.record.policy is policy
        )
        retired = _retire_stale_settled_publication_if_safe(receipt, parent=parent)
        receipt_outcome = "rolled_back" if retired else _recover(receipt, parent=parent)
        if receipt_outcome == "published":
            if same_request:
                if _reuse_or_retire_exact_receipt(receipt, parent=parent):
                    return GuardedTreePublicationResult(destination=destination, recovered=True)
            elif not (
                policy is DestinationPolicy.REPLACE_DIRECTORY
                and predecessor_request_digest == receipt.record.request_digest
            ):
                raise GuardedTreePublicationError(
                    "publication_request_conflict",
                    "the terminal publication receipt does not authorize this successor request",
                    paths=(destination.name,),
                )
            else:
                _require_record_authority(receipt, parent=parent)
                predecessor_receipt_identity = receipt.identity
                predecessor_receipt_sha256 = receipt.entry_sha256
        else:
            if predecessor_request_digest is not None:
                raise GuardedTreePublicationError(
                    "publication_request_conflict",
                    "the requested predecessor publication is no longer current",
                    paths=(destination.name,),
                )
    elif predecessor_request_digest is not None:
        raise GuardedTreePublicationError(
            "publication_request_conflict",
            "the requested predecessor publication receipt is missing",
            paths=(destination.name,),
        )

    original = _validate_destination(parent, destination.name, policy=policy)
    original_identity = (
        None if original is None else parent.entry_identity(destination.name, value=original)
    )
    original_destination_mode = None if original is None else stat.S_IMODE(original.st_mode)
    if original_identity is None:
        original_sha256 = None
        original_entries: tuple[_CleanupEntry, ...] = ()
    else:
        original_sha256, original_entries = _capture_tree_authority(
            parent.path / destination.name,
            expected=original_identity,
            require_cleanup_access=True,
        )
    token = secrets.token_hex(16)
    record = _Record(
        consumer=consumer,
        request_digest=request_digest,
        predecessor_request_digest=predecessor_request_digest,
        predecessor_receipt_identity=predecessor_receipt_identity,
        predecessor_receipt_sha256=predecessor_receipt_sha256,
        token=token,
        destination_name=destination.name,
        policy=policy,
        parent_identity=parent.identity,
        original_identity=original_identity,
        original_sha256=original_sha256,
        stage_name=f".cayu-tree-stage-{token}",
        stage_identity=None,
        stage_sha256=None,
        backup_name=f".cayu-tree-backup-{token}",
        cleanup_manifest_identity=None,
        cleanup_manifest_sha256=None,
        phase=_Phase.PREPARED,
    )
    if original_identity is not None and original_sha256 is not None:
        _cleanup_manifest_content(
            record,
            cleanup_name=_cleanup_name(record),
            root_identity=original_identity,
            root_sha256=original_sha256,
            entries=original_entries,
        )
    _require_expected_destination(record, parent=parent)
    try:
        journal = _create_journal(journal_path, record=record, parent=parent)
    except BaseException as error:
        try:
            if parent.entry_stat(journal_path.name) is not None:
                created = _load_journal(journal_path, parent=parent)
                _outcome, cleanup_error = _settle_after_failure(
                    created,
                    parent=parent,
                    error=error,
                )
                if cleanup_error is not None:
                    _raise_primary_with_settlement_failure(error, cleanup_error)
        except BaseException as cleanup_error:
            if cleanup_error is error:
                raise
            _record_settlement_failure(error, cleanup_error)
            _raise_primary_with_settlement_failure(error, cleanup_error)
        raise
    try:
        result = _execute_publication(
            journal,
            parent=parent,
            populate=populate,
            original_destination_mode=original_destination_mode,
        )
        _promote_terminal_receipt(journal, receipt_path=receipt_path, parent=parent)
        return result
    except BaseException as error:
        outcome, cleanup_error = _settle_after_failure(
            journal,
            parent=parent,
            error=error,
        )
        if cleanup_error is not None:
            _raise_primary_with_settlement_failure(error, cleanup_error)
        if outcome == "published" and isinstance(error, Exception):
            _promote_terminal_receipt(journal, receipt_path=receipt_path, parent=parent)
            return GuardedTreePublicationResult(destination=destination, recovered=True)
        raise


def recover_guarded_tree(destination: Path) -> str | None:
    """Recover one exact destination-scoped operation without starting new work."""

    destination = Path(destination)
    _validate_destination_name(destination)
    parent_path = destination.parent
    expected_parent = _capture_parent(parent_path)
    completed = False
    outcome: str | None = None
    try:
        with (
            cooperative_path_lock(
                parent_path,
                destination.name,
                lock_directory_name="cayu-tree-publication-locks-v1",
            ),
            _pinned_parent(parent_path, expected=expected_parent) as parent,
        ):
            metadata_stem = _resolve_destination_metadata_stem(parent, destination.name)
            journal_path = parent_path / f".cayu-tree-publication-{metadata_stem}.jsonl"
            receipt_path = parent_path / f".cayu-tree-publication-{metadata_stem}-receipt.jsonl"
            _reject_case_alias(parent, destination.name)
            if parent.entry_stat(journal_path.name) is not None:
                journal = _load_journal(journal_path, parent=parent)
                if journal.record.destination_name != destination.name:
                    raise _invalid_journal(
                        "publication journal does not belong to this destination"
                    )
                _require_parent_namespace_mutable(parent)
                outcome = _recover(journal, parent=parent)
                if outcome == "published":
                    _promote_terminal_receipt(
                        journal,
                        receipt_path=receipt_path,
                        parent=parent,
                    )
            elif parent.entry_stat(receipt_path.name) is not None:
                receipt = _load_journal(receipt_path, parent=parent)
                if receipt.record.destination_name != destination.name:
                    raise _invalid_journal(
                        "publication receipt does not belong to this destination"
                    )
                if receipt.record.phase is not _Phase.SETTLED:
                    raise _invalid_journal("publication receipt is not terminal")
                outcome = _recover(receipt, parent=parent)
            completed = True
    except BaseException as boundary_cleanup_error:
        if not completed:
            raise
        if not isinstance(boundary_cleanup_error, Exception):
            raise
        raise GuardedTreePublicationError(
            "boundary_cleanup_failed",
            "recovery reached a durable outcome, but its ownership boundary did not settle "
            "cleanly; an exact retry is safe",
            paths=(destination.name,),
        ) from boundary_cleanup_error
    return outcome


def _execute_publication(
    journal: _Journal,
    *,
    parent: _Parent,
    populate: Callable[[GuardedTreeStage], None],
    original_destination_mode: int | None,
) -> GuardedTreePublicationResult:
    record = journal.record
    stage = parent.path / record.stage_name
    _create_private_stage(stage, token=record.token, parent=parent)
    stage_identity = _require_directory_identity(parent, record.stage_name, label="staging")
    _append_journal(
        journal,
        replace(record, phase=_Phase.STAGING, stage_identity=stage_identity),
        parent=parent,
    )
    _remove_owner_marker(
        parent,
        record.stage_name,
        expected=stage_identity,
        token=record.token,
    )
    _publication_fault("stage_created")
    populate(
        GuardedTreeStage._owned(
            stage,
            expected=stage_identity,
            original_destination_mode=original_destination_mode,
        )
    )
    stage_sha256, stage_entries = _capture_tree_authority(
        stage,
        expected=stage_identity,
        require_cleanup_access=True,
    )
    _cleanup_manifest_content(
        journal.record,
        cleanup_name=_cleanup_name(journal.record),
        root_identity=stage_identity,
        root_sha256=stage_sha256,
        entries=stage_entries,
    )
    _append_journal(
        journal,
        replace(journal.record, phase=_Phase.STAGED, stage_sha256=stage_sha256),
        parent=parent,
    )
    _publication_fault("stage_synced")

    _require_record_authority(journal, parent=parent)
    _require_expected_destination(journal.record, parent=parent)
    _require_exact_stage(journal.record, parent=parent)
    _append_journal(
        journal,
        replace(journal.record, phase=_Phase.COMMIT_INTENT),
        parent=parent,
    )
    _publication_fault("commit_intent_synced")

    if journal.record.original_identity is not None:
        _rename_no_replace(
            parent,
            journal.record.destination_name,
            journal.record.backup_name,
            expected=journal.record.original_identity,
            label="original destination",
        )
        parent.sync()
        _require_identity(
            parent,
            journal.record.backup_name,
            journal.record.original_identity,
            label="backup",
        )
        _append_journal(
            journal,
            replace(journal.record, phase=_Phase.ORIGINAL_BACKED_UP),
            parent=parent,
        )
        _publication_fault("original_backed_up")

    if journal.record.stage_identity is None:
        raise _conflict(journal.record, "staging authority is incomplete")
    _rename_no_replace(
        parent,
        journal.record.stage_name,
        journal.record.destination_name,
        expected=journal.record.stage_identity,
        label="staging",
    )
    parent.sync()
    _require_published_stage(journal.record, parent=parent)
    _publication_fault("tree_renamed")

    destination = parent.path / journal.record.destination_name
    _finalize_published_tree(
        parent,
        journal.record.destination_name,
        expected=journal.record.stage_identity,
    )
    _require_published_stage(journal.record, parent=parent)
    _append_journal(
        journal,
        replace(journal.record, phase=_Phase.PUBLISHED),
        parent=parent,
    )
    _publication_fault("published")
    if journal.record.original_identity is not None:
        _require_unchanged_original_backup(journal.record, parent=parent)
        _append_journal(
            journal,
            replace(journal.record, phase=_Phase.CLEANUP_OWNED),
            parent=parent,
        )
        _publication_fault("cleanup_owned")
        _remove_exact_tree(
            parent,
            journal.record.backup_name,
            journal.record.original_identity,
            expected_sha256=journal.record.original_sha256,
            cleanup_name=_cleanup_name(journal.record),
            journal=journal,
            sealed_phase=_Phase.CLEANUP_SEALED,
        )
        parent.sync()
    _append_journal(
        journal,
        replace(journal.record, phase=_Phase.SETTLED),
        parent=parent,
    )
    _publication_fault("settled")
    return GuardedTreePublicationResult(destination=destination)


def _recover(
    journal: _Journal,
    *,
    parent: _Parent,
) -> str:
    record = journal.record
    _require_record_authority(journal, parent=parent)
    if record.parent_identity != parent.identity:
        raise _conflict(record, "publication parent does not match its durable authority")

    destination = parent.entry_stat(record.destination_name)
    stage = parent.entry_stat(record.stage_name)
    backup = parent.entry_stat(record.backup_name)
    cleanup_name = _cleanup_name(record)
    cleanup = parent.entry_stat(cleanup_name)
    destination_is_stage = _matches(
        parent,
        record.destination_name,
        record.stage_identity,
        destination,
    )
    destination_is_original = _matches(
        parent,
        record.destination_name,
        record.original_identity,
        destination,
    )
    stage_is_stage = _matches(parent, record.stage_name, record.stage_identity, stage)
    backup_is_original = _matches(
        parent,
        record.backup_name,
        record.original_identity,
        backup,
    )
    cleanup_is_original = _matches(
        parent,
        cleanup_name,
        record.original_identity,
        cleanup,
    )
    cleanup_is_stage = _matches(
        parent,
        cleanup_name,
        record.stage_identity,
        cleanup,
    )

    if destination_is_stage:
        if stage is not None:
            raise _conflict(record, "staging and published identities both exist")
        if backup is not None and not backup_is_original:
            raise _conflict(record, "backup identity conflicts with the publication record")
        if cleanup is not None and not cleanup_is_original:
            raise _conflict(record, "cleanup identity conflicts with the publication record")
        if backup is not None and cleanup is not None:
            raise _conflict(record, "backup and claimed cleanup identities both exist")
        if record.phase in {
            _Phase.PUBLISHED,
            _Phase.CLEANUP_OWNED,
            _Phase.CLEANUP_SEALED,
            _Phase.SETTLED,
        }:
            _require_published_identity(record, parent=parent)
        else:
            if record.stage_identity is None:
                raise _conflict(record, "published staging authority is incomplete")
            _finalize_published_tree(
                parent,
                record.destination_name,
                expected=record.stage_identity,
            )
            _require_published_stage(record, parent=parent)
            _append_journal(
                journal,
                replace(journal.record, phase=_Phase.PUBLISHED),
                parent=parent,
            )
        if backup_is_original or cleanup_is_original:
            if record.phase not in {_Phase.CLEANUP_OWNED, _Phase.CLEANUP_SEALED}:
                if cleanup_is_original:
                    raise _conflict(record, "cleanup claim lacks durable ownership authority")
                _require_unchanged_original_backup(record, parent=parent)
                _append_journal(
                    journal,
                    replace(record, phase=_Phase.CLEANUP_OWNED),
                    parent=parent,
                )
            if journal.record.phase is _Phase.CLEANUP_SEALED and backup_is_original:
                raise _conflict(
                    record,
                    "sealed cleanup authority did not retain its claimed namespace",
                )
            _remove_exact_tree(
                parent,
                record.backup_name,
                record.original_identity,
                expected_sha256=record.original_sha256,
                cleanup_name=cleanup_name,
                journal=journal,
                sealed_phase=_Phase.CLEANUP_SEALED,
            )
            parent.sync()
        elif record.original_identity is not None and record.phase not in {
            _Phase.CLEANUP_SEALED,
            _Phase.SETTLED,
        }:
            raise _conflict(record, "original cleanup evidence disappeared before ownership")
        if journal.record.phase is _Phase.CLEANUP_SEALED:
            # Absence is not durable cleanup evidence until the parent namespace
            # has itself synchronized successfully.
            parent.sync()
            _remove_cleanup_manifest_if_owned(journal, parent=parent)
        if journal.record.phase is not _Phase.SETTLED:
            _append_journal(
                journal,
                replace(journal.record, phase=_Phase.SETTLED),
                parent=parent,
            )
        parent.sync()
        return "published"

    old_is_restored = (
        record.original_identity is None and destination is None
    ) or destination_is_original
    if old_is_restored and backup is None:
        if stage is None and cleanup is None and record.phase is _Phase.ROLLBACK_CLEANUP_OWNED:
            raise _conflict(
                record,
                "rollback cleanup evidence disappeared before it was durably sealed",
            )
        if stage is not None and cleanup is not None:
            raise _conflict(record, "staging and claimed cleanup identities both exist")
        prepared_stage_name = record.stage_name if stage is not None else cleanup_name
        prepared_stage_identity = (
            _owned_prepared_stage_identity(
                parent,
                prepared_stage_name,
                token=record.token,
            )
            if record.stage_identity is None and (stage is not None or cleanup is not None)
            else None
        )
        if cleanup is not None and not (cleanup_is_stage or prepared_stage_identity is not None):
            raise _conflict(record, "claimed cleanup identity conflicts with rollback authority")
        if stage is not None or cleanup is not None:
            if not (stage_is_stage or cleanup_is_stage or prepared_stage_identity is not None):
                raise _conflict(record, "staging identity conflicts with the publication record")
            if record.phase not in {
                _Phase.ROLLBACK_CLEANUP_OWNED,
                _Phase.ROLLBACK_CLEANUP_SEALED,
            }:
                if cleanup is not None:
                    raise _conflict(record, "cleanup claim lacks durable rollback authority")
                _append_journal(
                    journal,
                    replace(
                        journal.record,
                        phase=_Phase.ROLLBACK_CLEANUP_OWNED,
                        stage_identity=(journal.record.stage_identity or prepared_stage_identity),
                    ),
                    parent=parent,
                )
            if stage_is_stage:
                _remove_exact_tree(
                    parent,
                    record.stage_name,
                    record.stage_identity,
                    expected_sha256=record.stage_sha256,
                    cleanup_name=cleanup_name,
                    journal=journal,
                    sealed_phase=_Phase.ROLLBACK_CLEANUP_SEALED,
                )
            elif prepared_stage_identity is not None:
                _remove_exact_tree(
                    parent,
                    record.stage_name,
                    prepared_stage_identity,
                    expected_sha256=None,
                    cleanup_name=cleanup_name,
                    journal=journal,
                    sealed_phase=_Phase.ROLLBACK_CLEANUP_SEALED,
                )
            elif cleanup_is_stage:
                _remove_exact_tree(
                    parent,
                    record.stage_name,
                    record.stage_identity,
                    expected_sha256=record.stage_sha256,
                    cleanup_name=cleanup_name,
                    journal=journal,
                    sealed_phase=_Phase.ROLLBACK_CLEANUP_SEALED,
                )
            parent.sync()
        if journal.record.phase is _Phase.ROLLBACK_CLEANUP_SEALED:
            _remove_cleanup_manifest_if_owned(journal, parent=parent)
        _remove_journal(journal, parent=parent)
        return "rolled_back"

    if destination is None and backup_is_original:
        _require_unchanged_original_backup(record, parent=parent)
        if cleanup is not None:
            raise _conflict(record, "cleanup claim exists before rollback ownership")
        if stage is not None and not stage_is_stage:
            raise _conflict(record, "staging identity conflicts with the publication record")
        if record.original_identity is None:
            raise _conflict(record, "original backup authority is incomplete")
        _rename_no_replace(
            parent,
            record.backup_name,
            record.destination_name,
            expected=record.original_identity,
            label="backup",
        )
        parent.sync()
        _require_identity(
            parent,
            record.destination_name,
            record.original_identity,
            label="restored destination",
        )
        if stage_is_stage:
            _append_journal(
                journal,
                replace(journal.record, phase=_Phase.ROLLBACK_CLEANUP_OWNED),
                parent=parent,
            )
            _remove_exact_tree(
                parent,
                record.stage_name,
                record.stage_identity,
                expected_sha256=record.stage_sha256,
                cleanup_name=cleanup_name,
                journal=journal,
                sealed_phase=_Phase.ROLLBACK_CLEANUP_SEALED,
            )
            parent.sync()
        _remove_journal(journal, parent=parent)
        return "rolled_back"

    raise _conflict(record, "filesystem state is ambiguous for guarded publication recovery")


def _settle_after_failure(
    journal: _Journal,
    *,
    parent: _Parent,
    error: BaseException,
) -> tuple[str | None, BaseException | None]:
    try:
        try:
            current = _load_journal(journal.path, parent=parent)
        except FileNotFoundError:
            return _reconcile_missing_journal(journal.record, parent=parent), None
        if current.record.phase in {
            _Phase.ROLLBACK_CLEANUP_OWNED,
            _Phase.ROLLBACK_CLEANUP_SEALED,
        }:
            return _recover(current, parent=parent), None
        if current.record.phase not in {
            _Phase.PUBLISHED,
            _Phase.CLEANUP_OWNED,
            _Phase.CLEANUP_SEALED,
            _Phase.SETTLED,
        }:
            return _rollback_before_publication(current, parent=parent), None
        return (
            _recover(
                current,
                parent=parent,
            ),
            None,
        )
    except BaseException as cleanup_error:
        if cleanup_error.__cause__ is None and cleanup_error.__context__ is error:
            cleanup_error.__context__ = None
        _record_settlement_failure(error, cleanup_error)
        return None, cleanup_error


def _record_settlement_failure(error: BaseException, cleanup_error: BaseException) -> None:
    paths = getattr(cleanup_error, "paths", ())
    path_evidence = ""
    if paths:
        path_evidence = "; affected paths: " + ", ".join(repr(path) for path in paths)
    error.add_note(
        f"{_SETTLEMENT_RECOVERY_NOTE_PREFIX}{type(cleanup_error).__name__}{path_evidence}"
    )


def _raise_primary_with_settlement_failure(
    primary: BaseException,
    settlement_failure: BaseException,
) -> NoReturn:
    process_signal = _first_process_control_signal(primary)
    signal_source = primary
    other_source = settlement_failure
    signal_precedes_other = True
    if process_signal is None:
        process_signal = _first_process_control_signal(settlement_failure)
        signal_source = settlement_failure
        other_source = primary
        signal_precedes_other = False
    if process_signal is not None:
        remaining_signal_source = _without_failure_identity(
            signal_source,
            omitted=process_signal,
        )
        evidence = (
            [*(() if remaining_signal_source is None else (remaining_signal_source,)), other_source]
            if signal_precedes_other
            else [
                other_source,
                *(() if remaining_signal_source is None else (remaining_signal_source,)),
            ]
        )
        carried = exception_cause(process_signal)
        if carried is None and not exception_suppresses_context(process_signal):
            carried = exception_context(process_signal)
        if carried is not None and not _failure_trees_overlap(
            carried,
            (process_signal, *evidence),
        ):
            evidence.append(carried)
        cause = (
            evidence[0]
            if len(evidence) == 1
            else BaseExceptionGroup(
                "Guarded publication operation and settlement failures.",
                evidence,
            )
        )
        raise process_signal from cause
    _raise_primary_with_secondary_failure(
        primary,
        settlement_failure,
        group_message="Guarded publication operation and settlement failures.",
    )


def _first_process_control_signal(error: BaseException) -> BaseException | None:
    return next(
        (
            candidate
            for candidate in iter_exception_tree(error)
            if isinstance(candidate, _PROCESS_CONTROL_SIGNALS)
        ),
        None,
    )


def _without_failure_identity(
    error: BaseException,
    *,
    omitted: BaseException,
) -> BaseException | None:
    """Rebuild grouped evidence without duplicating the authoritative signal."""

    rebuilt: dict[int, BaseException | None] = {id(omitted): None}
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    pending: list[tuple[BaseException, bool]] = [(error, False)]
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in rebuilt:
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            rebuilt[candidate_id] = candidate
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            retained = tuple(
                rebuilt_child
                for child in children
                if (rebuilt_child := rebuilt.get(id(child))) is not None
            )
            rebuilt[candidate_id] = (
                None
                if not retained
                else BaseExceptionGroup(
                    "Guarded publication settlement retained additional grouped failures.",
                    retained,
                )
            )
            continue
        children = exception_group_children(candidate)
        if children is None:
            rebuilt[candidate_id] = RuntimeError(
                "Guarded publication settlement returned invalid grouped evidence."
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))
    return rebuilt.get(id(error))


def _failure_trees_overlap(
    candidate: BaseException,
    evidence: tuple[BaseException, ...],
) -> bool:
    candidate_ids = {id(item) for item in iter_exception_tree(candidate)}
    return any(id(item) in candidate_ids for root in evidence for item in iter_exception_tree(root))


def _raise_primary_with_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
    *,
    group_message: str,
) -> NoReturn:
    previous_evidence = primary.__cause__
    if (
        previous_evidence is None
        and primary.__context__ is not None
        and not primary.__suppress_context__
    ):
        previous_evidence = primary.__context__
    evidence = [
        *(() if previous_evidence is None else (previous_evidence,)),
        secondary,
    ]
    cause = (
        evidence[0]
        if len(evidence) == 1
        else BaseExceptionGroup(
            group_message,
            evidence,
        )
    )
    raise primary from cause


def _close_descriptor(descriptor: int) -> None:
    """Close one descriptor without replacing an in-flight authoritative failure."""

    primary = sys.exception()
    try:
        os.close(descriptor)
    except BaseException as close_error:
        if primary is None:
            raise
        if close_error.__cause__ is None and close_error.__context__ is primary:
            close_error.__context__ = None
        _raise_primary_with_secondary_failure(
            primary,
            close_error,
            group_message="Guarded publication operation and descriptor cleanup failures.",
        )


def _rollback_before_publication(journal: _Journal, *, parent: _Parent) -> str:
    record = journal.record
    _require_record_authority(journal, parent=parent)
    if record.parent_identity != parent.identity:
        raise _conflict(record, "publication parent does not match its durable authority")

    destination = parent.entry_stat(record.destination_name)
    stage = parent.entry_stat(record.stage_name)
    backup = parent.entry_stat(record.backup_name)
    cleanup_name = _cleanup_name(record)
    if parent.entry_stat(cleanup_name) is not None:
        raise _conflict(record, "cleanup claim exists before durable rollback ownership")
    if _matches(parent, record.destination_name, record.stage_identity, destination):
        if stage is not None:
            raise _conflict(record, "staging and published identities both exist")
        _require_published_stage(record, parent=parent)
        if record.stage_identity is None:
            raise _conflict(record, "published staging authority is incomplete")
        _rename_no_replace(
            parent,
            record.destination_name,
            record.stage_name,
            expected=record.stage_identity,
            label="published destination",
        )
        parent.sync()
        destination = parent.entry_stat(record.destination_name)
        stage = parent.entry_stat(record.stage_name)

    if record.original_identity is None:
        if destination is not None or backup is not None:
            raise _conflict(record, "filesystem state conflicts with rollback authority")
    elif (
        _matches(parent, record.destination_name, record.original_identity, destination)
        and backup is None
    ):
        pass
    elif destination is None and _matches(
        parent,
        record.backup_name,
        record.original_identity,
        backup,
    ):
        _require_unchanged_original_backup(record, parent=parent)
        if record.original_identity is None:
            raise _conflict(record, "original backup authority is incomplete")
        _rename_no_replace(
            parent,
            record.backup_name,
            record.destination_name,
            expected=record.original_identity,
            label="backup",
        )
        parent.sync()
        _require_identity(
            parent,
            record.destination_name,
            record.original_identity,
            label="restored destination",
        )
    else:
        raise _conflict(record, "filesystem state conflicts with rollback authority")

    if stage is not None:
        prepared_stage_identity = (
            _owned_prepared_stage_identity(
                parent,
                record.stage_name,
                token=record.token,
            )
            if record.stage_identity is None
            else None
        )
        cleanup_identity = (
            record.stage_identity
            if _matches(parent, record.stage_name, record.stage_identity, stage)
            else prepared_stage_identity
        )
        if cleanup_identity is None:
            raise _conflict(record, "staging identity conflicts with rollback authority")
        _append_journal(
            journal,
            replace(
                journal.record,
                phase=_Phase.ROLLBACK_CLEANUP_OWNED,
                stage_identity=cleanup_identity,
            ),
            parent=parent,
        )
        _remove_exact_tree(
            parent,
            record.stage_name,
            cleanup_identity,
            expected_sha256=(
                record.stage_sha256 if cleanup_identity == record.stage_identity else None
            ),
            cleanup_name=cleanup_name,
            journal=journal,
            sealed_phase=_Phase.ROLLBACK_CLEANUP_SEALED,
        )
        parent.sync()
    _remove_journal(journal, parent=parent)
    return "rolled_back"


def _reconcile_missing_journal(record: _Record, *, parent: _Parent) -> str:
    if record.parent_identity != parent.identity:
        raise _conflict(record, "publication parent does not match its durable authority")
    destination = parent.entry_stat(record.destination_name)
    stage = parent.entry_stat(record.stage_name)
    backup = parent.entry_stat(record.backup_name)
    if (
        _matches(parent, record.destination_name, record.stage_identity, destination)
        and stage is None
        and backup is None
    ):
        if record.phase in {
            _Phase.PUBLISHED,
            _Phase.CLEANUP_OWNED,
            _Phase.CLEANUP_SEALED,
            _Phase.SETTLED,
        }:
            _require_published_identity(record, parent=parent)
        else:
            _require_published_stage(record, parent=parent)
        return "published"
    if (
        backup is None
        and stage is None
        and (
            (record.original_identity is None and destination is None)
            or _matches(
                parent,
                record.destination_name,
                record.original_identity,
                destination,
            )
        )
    ):
        return "rolled_back"
    raise _conflict(record, "journal disappeared before filesystem settlement was proven")


def _validate_publication_input(
    destination: Path,
    *,
    consumer: str,
    request_digest: str,
    policy: DestinationPolicy,
    populate: Callable[[GuardedTreeStage], None],
    predecessor_request_digest: str | None,
    settle_active_operation: bool,
) -> None:
    _validate_destination_name(destination)
    if not isinstance(policy, DestinationPolicy):
        raise GuardedTreePublicationError(
            "invalid_policy",
            "publication destination policy is invalid",
        )
    if not callable(populate):
        raise GuardedTreePublicationError(
            "invalid_callback",
            "publication callbacks must be callable",
        )
    if type(settle_active_operation) is not bool:
        raise GuardedTreePublicationError(
            "invalid_recovery_policy",
            "publication active-operation settlement policy is invalid",
        )
    try:
        consumer_size = len(consumer.encode("utf-8"))
    except (AttributeError, UnicodeEncodeError) as exc:
        raise GuardedTreePublicationError(
            "invalid_consumer",
            "invalid publication consumer",
        ) from exc
    if (
        not consumer
        or consumer_size > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in consumer)
    ):
        raise GuardedTreePublicationError("invalid_consumer", "invalid publication consumer")
    if not _is_sha256(request_digest):
        raise GuardedTreePublicationError(
            "invalid_request_digest",
            "publication request digest must be a canonical SHA-256 value",
        )
    if predecessor_request_digest is not None and not _is_sha256(predecessor_request_digest):
        raise GuardedTreePublicationError(
            "invalid_predecessor_request_digest",
            "publication predecessor digest must be a canonical SHA-256 value",
        )
    if predecessor_request_digest is not None and policy is not DestinationPolicy.REPLACE_DIRECTORY:
        raise GuardedTreePublicationError(
            "invalid_predecessor_request_digest",
            "only replacement publication can name a predecessor receipt",
        )


def _validate_destination_name(destination: Path) -> None:
    if destination.parent == destination or not destination.name or destination.name in {".", ".."}:
        raise GuardedTreePublicationError(
            "invalid_destination",
            "guarded publication requires a non-root destination",
        )
    try:
        destination_size = len(destination.name.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise GuardedTreePublicationError(
            "invalid_destination",
            "publication destination name is invalid",
        ) from exc
    if destination_size > 255:
        raise GuardedTreePublicationError(
            "invalid_destination",
            "publication destination name is too long",
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in destination.name) or (
        os.name == "nt"
        and ("\\" in destination.name or _is_unsafe_windows_component(destination.name))
    ):
        raise GuardedTreePublicationError(
            "invalid_destination",
            "publication destination name is invalid",
        )


def _is_unsafe_windows_component(component: str) -> bool:
    """Reject spellings that Win32 aliases, reserves, or cannot represent."""

    if component.endswith((" ", ".")) or component.strip(" ") in {".", ".."}:
        return True
    if any(
        ord(character) < 32
        or ord(character) == 127
        or character in _WINDOWS_INVALID_FILENAME_CHARACTERS
        for character in component
    ):
        return True
    stem = component.partition(".")[0].rstrip(" ").casefold()
    return stem in _WINDOWS_RESERVED_FILENAME_STEMS


def _prepare_stage_files(
    contents: Mapping[str, bytes],
    *,
    directories: Iterable[str],
    file_modes: Mapping[str, int] | None,
    file_mode: int | None,
    directory_mode: int | None,
) -> tuple[tuple[_PreparedStageFile, ...], tuple[_PreparedStageDirectory, ...]]:
    if file_mode is not None:
        _validate_stage_mode(file_mode, directory=False)
    if directory_mode is not None:
        _validate_stage_mode(directory_mode, directory=True)
    try:
        raw_items = tuple(contents.items())
    except (AttributeError, RuntimeError) as exc:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage contents must be a stable mapping",
        ) from exc
    try:
        raw_directories = tuple(directories)
    except (TypeError, RuntimeError) as exc:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage directories must be a stable iterable",
        ) from exc
    try:
        raw_file_modes = () if file_modes is None else tuple(file_modes.items())
    except (AttributeError, RuntimeError) as exc:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage file modes must be a stable mapping",
        ) from exc
    prepared_file_modes: dict[str, int] = {}
    for relative, mode in raw_file_modes:
        if type(relative) is not str or relative in prepared_file_modes:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage file modes contain an invalid path",
            )
        _validate_stage_mode(mode, directory=False)
        prepared_file_modes[relative] = mode

    files: list[_PreparedStageFile] = []
    file_parts: set[tuple[str, ...]] = set()
    for relative, content in raw_items:
        if type(relative) is not str or type(content) is not bytes:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage contents must map relative strings to bytes",
            )
        path = _prepare_stage_relative_path(relative)
        parts = path.parts
        if parts in file_parts:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage contents contain a duplicate file path",
                paths=(relative,),
            )
        file_parts.add(parts)
        files.append(
            _PreparedStageFile(
                path=path,
                content=content,
                mode=prepared_file_modes.pop(relative, file_mode),
            )
        )
    if prepared_file_modes:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage file modes name files absent from the staged contents",
            paths=(min(prepared_file_modes),),
        )

    directory_modes: dict[tuple[str, ...], int | None] = {}
    for relative in raw_directories:
        if type(relative) is not str:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage directories must contain relative strings",
            )
        path = _prepare_stage_relative_path(relative)
        directory_modes.setdefault(path.parts, directory_mode)
    explicit_directory_parts = set(directory_modes)
    normalized_entries: dict[tuple[str, ...], tuple[str, ...]] = {}
    for parts in (*file_parts, *explicit_directory_parts):
        for depth in range(1, len(parts) + 1):
            candidate = parts[:depth]
            normalized_candidate = tuple(_normalized_name(part) for part in candidate)
            previous_candidate = normalized_entries.get(normalized_candidate)
            if previous_candidate is not None and previous_candidate != candidate:
                raise GuardedTreePublicationError(
                    "tree_case_alias",
                    "guarded stage contents contain conflicting case or Unicode aliases",
                    paths=(
                        PurePosixPath(*previous_candidate).as_posix(),
                        PurePosixPath(*candidate).as_posix(),
                    ),
                )
            normalized_entries[normalized_candidate] = candidate
    for parts in file_parts:
        for depth in range(1, len(parts)):
            prefix = parts[:depth]
            if prefix in file_parts:
                raise GuardedTreePublicationError(
                    "invalid_population",
                    "guarded stage contents contain a file/directory topology conflict",
                    paths=(PurePosixPath(*prefix).as_posix(), PurePosixPath(*parts).as_posix()),
                )
            directory_modes.setdefault(prefix, directory_mode)
    for parts in tuple(directory_modes):
        if parts in file_parts:
            raise GuardedTreePublicationError(
                "invalid_population",
                "guarded stage contents contain a file/directory topology conflict",
                paths=(PurePosixPath(*parts).as_posix(),),
            )
        for depth in range(1, len(parts)):
            prefix = parts[:depth]
            if prefix in file_parts:
                raise GuardedTreePublicationError(
                    "invalid_population",
                    "guarded stage contents contain a file/directory topology conflict",
                    paths=(PurePosixPath(*prefix).as_posix(), PurePosixPath(*parts).as_posix()),
                )
            directory_modes.setdefault(prefix, directory_mode)
    prepared_directories = [
        _PreparedStageDirectory(path=PurePosixPath(*parts), mode=mode)
        for parts, mode in directory_modes.items()
    ]
    if len(files) + len(prepared_directories) > _TREE_ENTRY_LIMIT:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree exceeds its bounded entry limit",
        )
    _require_population_cleanup_capacity(
        files=files,
        directories=prepared_directories,
    )
    return (
        tuple(sorted(files, key=lambda item: item.path.as_posix())),
        tuple(
            sorted(
                prepared_directories,
                key=lambda item: (len(item.path.parts), item.path.as_posix()),
            )
        ),
    )


def _prepare_stage_relative_path(relative: str) -> PurePosixPath:
    try:
        relative_size = len(relative.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage contents contain an invalid relative path",
        ) from exc
    path = PurePosixPath(relative)
    parts = path.parts
    if (
        not relative
        or "\\" in relative
        or "\x00" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parts)
        or (os.name == "nt" and any(_is_unsafe_windows_component(part) for part in parts))
        or relative_size > 4096
    ):
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage contents contain an unsafe relative path",
            paths=(relative[:256],),
        )
    if len(parts) > _TREE_DEPTH_LIMIT:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree exceeds its bounded depth limit",
            paths=(relative[:256],),
        )
    return path


def _validate_stage_mode(mode: object, *, directory: bool) -> None:
    required = stat.S_IRUSR | (stat.S_IWUSR | stat.S_IXUSR if directory else 0)
    if os.name == "nt" and not directory:
        required |= stat.S_IWUSR
    if type(mode) is not int or not 0 <= mode <= 0o7777 or mode & required != required:
        raise GuardedTreePublicationError(
            "invalid_population",
            "guarded stage modes must be canonical and retain required owner access",
        )


def _require_population_cleanup_capacity(
    *,
    files: list[_PreparedStageFile],
    directories: list[_PreparedStageDirectory],
) -> None:
    # Use deliberately wide platform identities so every accepted population
    # is guaranteed to fit the exact cleanup manifest produced after writing.
    wide_identity = _Identity(
        device=(1 << 128) - 1,
        inode=(1 << 128) - 1,
        kind=stat.S_IFDIR,
        incarnation=(1 << 256) - 1,
    )
    entries: list[dict[str, object]] = []
    for item in directories:
        entries.append(
            {
                "path": item.path.as_posix(),
                "identity": wide_identity.as_json(),
                "mode": 0o7777 if item.mode is None else item.mode,
                "size": None,
                "content_sha256": None,
            }
        )
    file_identity = replace(wide_identity, kind=stat.S_IFREG)
    for item in files:
        entries.append(
            {
                "path": item.path.as_posix(),
                "identity": file_identity.as_json(),
                "mode": 0o7777 if item.mode is None else item.mode,
                "size": max(len(item.content), (1 << 63) - 1),
                "content_sha256": f"sha256:{'0' * 64}",
            }
        )
    payload = {
        "schema_version": _CLEANUP_MANIFEST_SCHEMA_VERSION,
        "token": "0" * 32,
        "cleanup_name": f".cayu-tree-cleanup-{'0' * 32}",
        "root_identity": wide_identity.as_json(),
        "root_sha256": f"sha256:{'0' * 64}",
        "entries": sorted(entries, key=lambda entry: cast("str", entry["path"])),
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    if len(encoded) > _CLEANUP_MANIFEST_LIMIT_BYTES:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication cleanup authority exceeds its bounded size limit",
        )


def _capture_stage_default_directory_mode(path: Path, *, expected: _Identity) -> int:
    """Observe directory creation mode inside the already owned private stage."""

    probe_name = ".cayu-tree-directory-mode-probe"
    if os.name == "nt":
        with _windows_directory_namespace_fence(path):
            current = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(current.st_mode)
                or _capture_stable_identity(current, path=path) != expected
            ):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory changed before mode observation",
                    paths=(path.name,),
                )
            probe = path / probe_name
            try:
                probe.mkdir(mode=0o777)
            except FileExistsError as exc:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory acquired unexpected content",
                    paths=(probe_name,),
                ) from exc
            probe_value = probe.stat(follow_symlinks=False)
            _require_plain_directory(
                probe_value,
                label="directory mode probe",
                path=probe_name,
            )
            probe_identity = _capture_stable_identity(probe_value, path=probe)
            mode = stat.S_IMODE(probe_value.st_mode)
            _delete_windows_entry_by_handle(probe, expected=probe_identity)
            after = path.stat(follow_symlinks=False)
            if _capture_stable_identity(after, path=path) != expected:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory changed during mode observation",
                    paths=(path.name,),
                )
    else:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            root = os.open(path, flags)
        except OSError as exc:
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed before mode observation",
                paths=(path.name,),
            ) from exc
        try:
            current = os.fstat(root)
            if (
                not stat.S_ISDIR(current.st_mode)
                or _capture_stable_identity(current, descriptor=root) != expected
            ):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory changed before mode observation",
                    paths=(path.name,),
                )
            try:
                os.mkdir(probe_name, mode=0o777, dir_fd=root)
            except FileExistsError as exc:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory acquired unexpected content",
                    paths=(probe_name,),
                ) from exc
            probe = os.open(probe_name, flags, dir_fd=root)
            try:
                probe_value = os.fstat(probe)
                if not stat.S_ISDIR(probe_value.st_mode):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication directory mode probe changed type",
                        paths=(probe_name,),
                    )
                probe_identity = _capture_stable_identity(probe_value, descriptor=probe)
                mode = stat.S_IMODE(probe_value.st_mode)
            finally:
                _close_descriptor(probe)
            current_probe = os.stat(probe_name, dir_fd=root, follow_symlinks=False)
            if (
                _capture_stable_identity(
                    current_probe,
                    dir_fd=root,
                    name=probe_name,
                )
                != probe_identity
            ):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication directory mode probe changed before cleanup",
                    paths=(probe_name,),
                )
            os.rmdir(probe_name, dir_fd=root)
            after = os.fstat(root)
            if _capture_stable_identity(after, descriptor=root) != expected:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory changed during mode observation",
                    paths=(path.name,),
                )
        finally:
            _close_descriptor(root)
    _validate_stage_mode(mode, directory=True)
    return mode


def _write_stage_files(
    path: Path,
    *,
    expected: _Identity,
    files: tuple[_PreparedStageFile, ...],
    directories: tuple[_PreparedStageDirectory, ...],
    root_mode: int,
) -> None:
    if os.name == "nt":
        _write_stage_files_on_windows(
            path,
            expected=expected,
            files=files,
            directories=directories,
            root_mode=root_mode,
        )
        return
    _write_stage_files_from_fd(
        path,
        expected=expected,
        files=files,
        directories=directories,
        root_mode=root_mode,
    )


def _write_stage_files_from_fd(
    path: Path,
    *,
    expected: _Identity,
    files: tuple[_PreparedStageFile, ...],
    directories: tuple[_PreparedStageDirectory, ...],
    root_mode: int,
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root = os.open(path, flags)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "staging_changed",
            "publication staging directory changed before population",
            paths=(path.name,),
        ) from exc
    try:
        root_identity = os.fstat(root)
        if (
            not stat.S_ISDIR(root_identity.st_mode)
            or _capture_stable_identity(root_identity, descriptor=root) != expected
        ):
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed before population",
                paths=(path.name,),
            )
        with os.scandir(root) as entries:
            if next(entries, None) is not None:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory was not empty before population",
                    paths=(path.name,),
                )
        directory_identities: dict[tuple[str, ...], _Identity] = {
            (): _capture_stable_identity(root_identity, descriptor=root)
        }
        for item in directories:
            _create_stage_directory_from_fd(
                root,
                root_path=path,
                item=item,
                directory_flags=flags,
                directory_identities=directory_identities,
            )
        for item in files:
            _write_stage_file_from_fd(
                root,
                root_path=path,
                item=item,
                directory_flags=flags,
                directory_identities=directory_identities,
            )
        os.fchmod(root, root_mode)
    finally:
        _close_descriptor(root)


def _create_stage_directory_from_fd(
    root: int,
    *,
    root_path: Path,
    item: _PreparedStageDirectory,
    directory_flags: int,
    directory_identities: dict[tuple[str, ...], _Identity],
) -> None:
    descriptor = os.dup(root)
    prefix: tuple[str, ...] = ()
    try:
        for component in item.path.parts:
            prefix = (*prefix, component)
            expected = directory_identities.get(prefix)
            if expected is None:
                try:
                    os.mkdir(
                        component,
                        mode=(0o777 if item.mode is None else 0o700),
                        dir_fd=descriptor,
                    )
                except FileExistsError as exc:
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication staging directory acquired unexpected content",
                        paths=(PurePosixPath(*prefix).as_posix(),),
                    ) from exc
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                current = os.fstat(child)
                current_identity = _capture_stable_identity(current, descriptor=child)
                if not stat.S_ISDIR(current.st_mode) or (
                    expected is not None and current_identity != expected
                ):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication staging directory changed during population",
                        paths=(PurePosixPath(*prefix).as_posix(),),
                    )
                if expected is None:
                    directory_identities[prefix] = current_identity
                if prefix == item.path.parts and item.mode is not None:
                    os.fchmod(child, item.mode)
            except BaseException:
                _close_descriptor(child)
                raise
            previous = descriptor
            descriptor = child
            _close_descriptor(previous)
    finally:
        _close_descriptor(descriptor)


def _write_stage_file_from_fd(
    root: int,
    *,
    root_path: Path,
    item: _PreparedStageFile,
    directory_flags: int,
    directory_identities: dict[tuple[str, ...], _Identity],
) -> None:
    descriptor = os.dup(root)
    prefix: tuple[str, ...] = ()
    try:
        for component in item.path.parts[:-1]:
            prefix = (*prefix, component)
            expected = directory_identities.get(prefix)
            if expected is None:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory lost prepared parent authority",
                    paths=(root_path.joinpath(*prefix).name,),
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                current = os.fstat(child)
                if not stat.S_ISDIR(current.st_mode) or (
                    expected is not None
                    and _capture_stable_identity(current, descriptor=child) != expected
                ):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication staging directory changed during population",
                        paths=(PurePosixPath(*prefix).as_posix(),),
                    )
            except BaseException:
                _close_descriptor(child)
                raise
            previous = descriptor
            descriptor = child
            _close_descriptor(previous)

        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            output = os.open(
                item.path.name,
                file_flags,
                0o666 if item.mode is None else 0o600,
                dir_fd=descriptor,
            )
        except FileExistsError as exc:
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory acquired unexpected content",
                paths=(item.path.as_posix(),),
            ) from exc
        try:
            _write_all(output, item.content)
            if item.mode is not None:
                os.fchmod(output, item.mode)
        finally:
            _close_descriptor(output)
    finally:
        _close_descriptor(descriptor)


def _write_stage_files_on_windows(
    path: Path,
    *,
    expected: _Identity,
    files: tuple[_PreparedStageFile, ...],
    directories: tuple[_PreparedStageDirectory, ...],
    root_mode: int,
) -> None:
    with ExitStack() as fences:
        fences.enter_context(_windows_directory_namespace_fence(path))
        current = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _capture_stable_identity(current, path=path) != expected
        ):
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed before population",
                paths=(path.name,),
            )
        with os.scandir(path) as entries:
            staging_has_entries = next(entries, None) is not None
        if staging_has_entries:
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory was not empty before population",
                paths=(path.name,),
            )
        directory_identities: dict[tuple[str, ...], _Identity] = {(): expected}
        for item in directories:
            directory = path
            prefix: tuple[str, ...] = ()
            for component in item.path.parts:
                prefix = (*prefix, component)
                directory /= component
                expected_directory = directory_identities.get(prefix)
                if expected_directory is None:
                    try:
                        directory.mkdir()
                    except FileExistsError as exc:
                        raise GuardedTreePublicationError(
                            "staging_changed",
                            "publication staging directory acquired unexpected content",
                            paths=(PurePosixPath(*prefix).as_posix(),),
                        ) from exc
                    fences.enter_context(_windows_directory_namespace_fence(directory))
                    identity = directory.stat(follow_symlinks=False)
                    _require_plain_directory(
                        identity,
                        label="staging directory",
                        path=PurePosixPath(*prefix).as_posix(),
                    )
                    directory_identities[prefix] = _capture_stable_identity(
                        identity,
                        path=directory,
                    )
                else:
                    identity = directory.stat(follow_symlinks=False)
                    if _capture_stable_identity(identity, path=directory) != expected_directory:
                        raise GuardedTreePublicationError(
                            "staging_changed",
                            "publication staging directory changed during population",
                            paths=(PurePosixPath(*prefix).as_posix(),),
                        )
                if prefix == item.path.parts and item.mode is not None:
                    directory.chmod(item.mode)
        for item in files:
            directory = path
            prefix = ()
            for component in item.path.parts[:-1]:
                prefix = (*prefix, component)
                directory /= component
                expected_directory = directory_identities.get(prefix)
                if expected_directory is None:
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication staging directory lost prepared parent authority",
                        paths=(PurePosixPath(*prefix).as_posix(),),
                    )
                identity = directory.stat(follow_symlinks=False)
                if _capture_stable_identity(identity, path=directory) != expected_directory:
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "publication staging directory changed during population",
                        paths=(PurePosixPath(*prefix).as_posix(),),
                    )
            target = directory / item.path.name
            try:
                with target.open("xb") as output:
                    output.write(item.content)
                if item.mode is not None:
                    target.chmod(item.mode)
            except FileExistsError as exc:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication staging directory acquired unexpected content",
                    paths=(item.path.as_posix(),),
                ) from exc
        path.chmod(root_mode)
        after = path.stat(follow_symlinks=False)
        if _capture_stable_identity(after, path=path) != expected:
            raise GuardedTreePublicationError(
                "staging_changed",
                "publication staging directory changed during population",
                paths=(path.name,),
            )


def _validate_destination(
    parent: _Parent,
    name: str,
    *,
    policy: DestinationPolicy,
) -> os.stat_result | None:
    current = parent.entry_stat(name)
    if current is None:
        return None
    _require_plain_directory(current, label="destination", path=name)
    if policy is DestinationPolicy.ABSENT_OR_EMPTY and not _directory_is_empty(
        parent,
        name,
        expected=parent.entry_identity(name, value=current),
    ):
        raise GuardedTreePublicationError(
            "destination_not_empty",
            "publication destination must be absent or empty",
            paths=(name,),
        )
    return current


def _require_expected_destination(record: _Record, *, parent: _Parent) -> None:
    current = parent.entry_stat(record.destination_name)
    if record.original_identity is None:
        if current is not None:
            raise _conflict(record, "destination appeared before publication")
        return
    if (
        current is None
        or parent.entry_identity(record.destination_name, value=current) != record.original_identity
    ):
        raise _conflict(record, "destination identity changed before publication")
    _require_plain_directory(current, label="destination", path=record.destination_name)
    actual = _seal_tree(
        parent.path / record.destination_name,
        expected=record.original_identity,
    )
    if actual != record.original_sha256:
        raise _conflict(record, "destination content changed before publication")
    if record.policy is DestinationPolicy.ABSENT_OR_EMPTY and not _directory_is_empty(
        parent,
        record.destination_name,
        expected=record.original_identity,
    ):
        raise _conflict(record, "destination became non-empty before publication")


def _require_unchanged_original_backup(record: _Record, *, parent: _Parent) -> None:
    if record.original_identity is None or record.original_sha256 is None:
        raise _conflict(record, "original backup authority is incomplete")
    actual = _seal_tree(
        parent.path / record.backup_name,
        expected=record.original_identity,
    )
    if actual != record.original_sha256:
        raise _conflict(record, "original destination changed after it was backed up")


def _create_private_stage(stage: Path, *, token: str, parent: _Parent) -> None:
    parent.assert_unchanged()
    if os.name == "nt":
        security_descriptor_error = _create_private_windows_directory(stage)
        _publication_fault("stage_directory_created")
        identity = _require_directory_identity(parent, stage.name, label="staging")
        with _windows_directory_namespace_fence(stage):
            marker = stage / _OWNER_MARKER
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(marker, flags, 0o600)
            try:
                _write_all(descriptor, token.encode("ascii"))
                os.fsync(descriptor)
            finally:
                _close_descriptor(descriptor)
            _sync_windows_path(stage, directory=True)
        _assert_windows_directory_dacl_is_protected(stage)
        if security_descriptor_error is not None:
            raise security_descriptor_error
    else:
        if parent.descriptor is None:
            raise AssertionError("POSIX publication parent has no descriptor")
        os.mkdir(stage.name, mode=0o700, dir_fd=parent.descriptor)
        _publication_fault("stage_directory_created")
        identity = _require_directory_identity(parent, stage.name, label="staging")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        stage_descriptor = os.open(
            stage.name,
            directory_flags,
            dir_fd=parent.descriptor,
        )
        try:
            opened_stage = os.fstat(stage_descriptor)
            if _capture_stable_identity(opened_stage, descriptor=stage_descriptor) != identity:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging directory changed during ownership publication",
                )
            marker_flags = (
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            marker_descriptor = os.open(
                _OWNER_MARKER,
                marker_flags,
                0o600,
                dir_fd=stage_descriptor,
            )
            try:
                _write_all(marker_descriptor, token.encode("ascii"))
                os.fsync(marker_descriptor)
            finally:
                _close_descriptor(marker_descriptor)
            os.fsync(stage_descriptor)
        finally:
            _close_descriptor(stage_descriptor)
    parent.sync()


def _remove_owner_marker(
    parent: _Parent,
    stage_name: str,
    *,
    expected: _Identity,
    token: str,
) -> None:
    if parent.descriptor is None:
        stage = parent.path / stage_name
        with _windows_directory_namespace_fence(stage):
            current_stage = stage.stat(follow_symlinks=False)
            if _capture_stable_identity(current_stage, path=stage) != expected:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging directory changed before ownership marker removal",
                )
            marker = stage / _OWNER_MARKER
            marker_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            marker_descriptor = os.open(marker, marker_flags)
            try:
                marker_identity = os.fstat(marker_descriptor)
                if not stat.S_ISREG(marker_identity.st_mode):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging ownership marker changed type",
                    )
                content = _read_bounded(marker_descriptor, limit=128)
                if content != token.encode("ascii"):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging ownership marker changed",
                    )
            finally:
                _close_descriptor(marker_descriptor)
            if not _Identity.capture(marker_identity).matches(marker.stat(follow_symlinks=False)):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging ownership marker was replaced",
                )
            marker.unlink()
            _sync_windows_path(stage, directory=True)
        return

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    stage_descriptor = os.open(stage_name, directory_flags, dir_fd=parent.descriptor)
    try:
        opened_stage = os.fstat(stage_descriptor)
        if _capture_stable_identity(opened_stage, descriptor=stage_descriptor) != expected:
            raise GuardedTreePublicationError(
                "staging_changed",
                "staging directory changed before ownership marker removal",
            )
        marker_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        marker_descriptor = os.open(
            _OWNER_MARKER,
            marker_flags,
            dir_fd=stage_descriptor,
        )
        try:
            marker_identity = os.fstat(marker_descriptor)
            if not stat.S_ISREG(marker_identity.st_mode):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging ownership marker changed type",
                )
            content = _read_bounded(marker_descriptor, limit=128)
            if content != token.encode("ascii"):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging ownership marker changed",
                )
        finally:
            _close_descriptor(marker_descriptor)
        current = os.stat(
            _OWNER_MARKER,
            dir_fd=stage_descriptor,
            follow_symlinks=False,
        )
        if not _Identity.capture(marker_identity).matches(current):
            raise GuardedTreePublicationError(
                "staging_changed",
                "staging ownership marker was replaced",
            )
        os.unlink(_OWNER_MARKER, dir_fd=stage_descriptor)
        os.fsync(stage_descriptor)
    finally:
        _close_descriptor(stage_descriptor)


def _owned_prepared_stage_identity(
    parent: _Parent,
    stage_name: str,
    *,
    token: str,
) -> _Identity | None:
    if parent.descriptor is None:
        stage = parent.path / stage_name
        try:
            with _windows_directory_namespace_fence(stage):
                current = stage.stat(follow_symlinks=False)
                _require_plain_directory(current, label="staging", path=stage_name)
                _assert_windows_directory_dacl_is_protected(stage)
                marker = stage / _OWNER_MARKER
                marker_flags = (
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(marker, marker_flags)
                except FileNotFoundError:
                    return None
                try:
                    marker_identity = os.fstat(descriptor)
                    if not stat.S_ISREG(marker_identity.st_mode) or _read_bounded(
                        descriptor,
                        limit=128,
                    ) != token.encode("ascii"):
                        return None
                finally:
                    _close_descriptor(descriptor)
                after = stage.stat(follow_symlinks=False)
                if not os.path.samestat(current, after):
                    return None
                return _capture_stable_identity(current, path=stage)
        except (GuardedTreePublicationError, OSError, UnicodeError):
            return None

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        stage_descriptor = os.open(
            stage_name,
            directory_flags,
            dir_fd=parent.descriptor,
        )
    except (OSError, UnicodeError):
        return None
    try:
        current = os.fstat(stage_descriptor)
        if not stat.S_ISDIR(current.st_mode) or stat.S_IMODE(current.st_mode) != 0o700:
            return None
        marker_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            marker_descriptor = os.open(
                _OWNER_MARKER,
                marker_flags,
                dir_fd=stage_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            marker_identity = os.fstat(marker_descriptor)
            if not stat.S_ISREG(marker_identity.st_mode) or _read_bounded(
                marker_descriptor,
                limit=128,
            ) != token.encode("ascii"):
                return None
        finally:
            _close_descriptor(marker_descriptor)
        return _capture_stable_identity(current, descriptor=stage_descriptor)
    finally:
        _close_descriptor(stage_descriptor)


def _linux_mount_points() -> frozenset[str]:
    if not sys.platform.startswith("linux"):
        return frozenset()
    mountinfo = Path("/proc/self/mountinfo")
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GuardedTreePublicationError(
            "mount_inspection_failed",
            "could not inspect Linux mount ownership boundaries",
        ) from exc
    mount_points: set[str] = set()
    for line in lines:
        fields = line.split(" ")
        if len(fields) < 6:
            raise GuardedTreePublicationError(
                "mount_inspection_failed",
                "Linux mount ownership evidence is malformed",
            )
        mount_path = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            fields[4],
        )
        mount_points.add(os.path.normpath(mount_path))
    return frozenset(mount_points)


def _path_is_mount_boundary(
    path: Path,
    *,
    parent_device: int | None = None,
    linux_mount_points: frozenset[str] | None = None,
) -> bool:
    if os.name == "nt":
        return False
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "mount_inspection_failed",
            "could not inspect a guarded publication mount boundary",
            paths=(path.name,),
        ) from exc
    if parent_device is not None and current.st_dev != parent_device:
        return True
    if os.path.ismount(path):
        return True
    if not sys.platform.startswith("linux"):
        return False
    if linux_mount_points is None:
        linux_mount_points = _linux_mount_points()
    absolute = os.path.normpath(os.path.abspath(path))
    return absolute in linux_mount_points


def _seal_tree(path: Path, *, expected: _Identity) -> str:
    digest, _entries = _capture_tree_authority(path, expected=expected)
    return digest


def _cleanup_access_error(path: str) -> GuardedTreePublicationError:
    return GuardedTreePublicationError(
        "cleanup_unavailable",
        "guarded publication cannot prove that the tree remains traversable and removable",
        paths=(path,),
    )


def _require_posix_directory_cleanup_access(path: Path, *, relative: str) -> None:
    access_kwargs: dict[str, bool] = {}
    if os.access in os.supports_effective_ids:
        access_kwargs["effective_ids"] = True
    if os.access in os.supports_follow_symlinks:
        access_kwargs["follow_symlinks"] = False
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK, **access_kwargs):
        raise _cleanup_access_error(relative)


def _posix_cleanup_blocking_flags(value: os.stat_result) -> int:
    blocking = 0
    for name in (
        "UF_IMMUTABLE",
        "UF_APPEND",
        "UF_NOUNLINK",
        "SF_IMMUTABLE",
        "SF_APPEND",
        "SF_NOUNLINK",
    ):
        blocking |= getattr(stat, name, 0)
    return getattr(value, "st_flags", 0) & blocking


def _linux_file_flags(descriptor: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    import array
    import fcntl

    flags = array.array("L", [0])
    ioctl_request = (
        (_LINUX_IOCTL_READ << 30)
        | (flags.itemsize << 16)
        | (_LINUX_FS_IOC_GETFLAGS_TYPE << 8)
        | _LINUX_FS_IOC_GETFLAGS_NUMBER
    )
    try:
        fcntl.ioctl(descriptor, ioctl_request, flags, True)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTTY, errno.EOPNOTSUPP}:
            return None
        raise
    return flags[0]


def _linux_cleanup_blocking_flags(descriptor: int) -> int:
    flags = _linux_file_flags(descriptor)
    if flags is None:
        return 0
    return flags & (_LINUX_FS_IMMUTABLE_FL | _LINUX_FS_APPEND_FL)


def _require_posix_cleanup_flags(
    value: os.stat_result,
    *,
    descriptor: int,
    relative: str,
) -> None:
    try:
        blocked = _posix_cleanup_blocking_flags(value) or _linux_cleanup_blocking_flags(descriptor)
    except OSError as exc:
        raise _cleanup_access_error(relative) from exc
    if blocked:
        raise _cleanup_access_error(relative)


def _require_parent_namespace_mutable(parent: _Parent) -> None:
    if parent.descriptor is None:
        return
    value = os.fstat(parent.descriptor)
    if _capture_stable_identity(value, descriptor=parent.descriptor) != parent.identity:
        raise GuardedTreePublicationError(
            "parent_changed",
            "publication parent changed before its namespace was mutated",
        )
    _require_posix_cleanup_flags(
        value,
        descriptor=parent.descriptor,
        relative=parent.path.name,
    )


def _require_windows_cleanup_access(
    path: Path,
    *,
    expected: _Identity,
    value: os.stat_result,
) -> None:
    if stat.S_ISREG(value.st_mode) and (
        getattr(value, "st_file_attributes", 0) & _WINDOWS_FILE_ATTRIBUTE_READONLY
    ):
        raise _cleanup_access_error(path.name)
    try:
        with _windows_deletion_handle(path):
            current = path.stat(follow_symlinks=False)
            if _capture_stable_identity(current, path=path) != expected:
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "publication tree changed while cleanup access was checked",
                    paths=(path.name,),
                )
    except OSError as exc:
        raise _cleanup_access_error(path.name) from exc


def _capture_tree_authority(
    path: Path,
    *,
    expected: _Identity,
    require_cleanup_access: bool = False,
    entry_limit: int | None = None,
) -> tuple[str, tuple[_CleanupEntry, ...]]:
    if entry_limit is None:
        entry_limit = _TREE_ENTRY_LIMIT
    if type(entry_limit) is not int or entry_limit <= 0:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree has an invalid entry limit",
        )
    if os.name == "nt":
        return _seal_tree_on_windows(
            path,
            expected=expected,
            require_cleanup_access=require_cleanup_access,
            entry_limit=entry_limit,
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except PermissionError as exc:
        if require_cleanup_access:
            raise _cleanup_access_error(path.name) from exc
        raise
    try:
        opened_root = os.fstat(descriptor)
        if _capture_stable_identity(opened_root, descriptor=descriptor) != expected:
            raise GuardedTreePublicationError("staging_changed", "staging root changed")
        digest = hashlib.sha256()
        root_value = opened_root
        if require_cleanup_access:
            _require_posix_cleanup_flags(
                root_value,
                descriptor=descriptor,
                relative=path.name,
            )
        linux_mount_points = _linux_mount_points()
        if _path_is_mount_boundary(
            path,
            parent_device=path.parent.stat().st_dev,
            linux_mount_points=linux_mount_points,
        ):
            raise GuardedTreePublicationError(
                "mount_boundary",
                "guarded publication cannot own a mounted filesystem root",
                paths=(path.name,),
            )
        digest.update(
            f"r\0{expected.device}\0{expected.inode}\0{expected.kind}\0"
            f"{expected.incarnation}\0"
            f"{stat.S_IMODE(root_value.st_mode):o}\0".encode()
        )
        entries: list[_CleanupEntry] = []
        entry_budget = _TreeEntryBudget(entry_limit)
        _seal_directory_from_fd(
            descriptor,
            root_path=path,
            prefix=PurePosixPath(),
            digest=digest,
            flags=flags,
            entries=entries,
            entry_budget=entry_budget,
            linux_mount_points=linux_mount_points,
            require_cleanup_access=require_cleanup_access,
        )
        os.fsync(descriptor)
        final_root = os.fstat(descriptor)
        if _capture_stable_identity(final_root, descriptor=descriptor) != expected:
            raise GuardedTreePublicationError("staging_changed", "staging root changed")
        return f"sha256:{digest.hexdigest()}", tuple(sorted(entries, key=lambda entry: entry.path))
    finally:
        _close_descriptor(descriptor)


def _seal_directory_from_fd(
    descriptor: int,
    *,
    root_path: Path,
    prefix: PurePosixPath,
    digest: Any,
    flags: int,
    entries: list[_CleanupEntry],
    entry_budget: _TreeEntryBudget,
    linux_mount_points: frozenset[str],
    require_cleanup_access: bool,
) -> None:
    try:
        with os.scandir(descriptor) as scanned_entries:
            names = _bounded_sealing_entry_names(
                scanned_entries,
                entry_budget=entry_budget,
            )
    except PermissionError as exc:
        if require_cleanup_access:
            raise _cleanup_access_error(prefix.as_posix() or root_path.name) from exc
        raise
    if require_cleanup_access and names:
        _require_posix_directory_cleanup_access(
            root_path.joinpath(*prefix.parts),
            relative=prefix.as_posix() or root_path.name,
        )
    _reject_normalized_aliases(names)
    for name in names:
        relative = prefix / name
        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        _require_supported_tree_entry(value, path=relative.as_posix())
        mode = stat.S_IMODE(value.st_mode)
        if stat.S_ISDIR(value.st_mode):
            if _path_is_mount_boundary(
                root_path.joinpath(*relative.parts),
                parent_device=os.fstat(descriptor).st_dev,
                linux_mount_points=linux_mount_points,
            ):
                raise GuardedTreePublicationError(
                    "mount_boundary",
                    "guarded publication cannot traverse a mounted descendant",
                    paths=(relative.as_posix(),),
                )
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except PermissionError as exc:
                if require_cleanup_access:
                    raise _cleanup_access_error(relative.as_posix()) from exc
                raise
            try:
                opened = os.fstat(child)
                if not _Identity.capture(value).matches(opened):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging directory changed while it was sealed",
                        paths=(relative.as_posix(),),
                    )
                if require_cleanup_access:
                    _require_posix_cleanup_flags(
                        opened,
                        descriptor=child,
                        relative=relative.as_posix(),
                    )
                opened_identity = _capture_stable_identity(
                    opened,
                    descriptor=child,
                )
                digest.update(
                    f"d\0{relative.as_posix()}\0{opened_identity.device}\0"
                    f"{opened_identity.inode}\0{opened_identity.kind}\0"
                    f"{opened_identity.incarnation}\0{mode:o}\0".encode()
                )
                _append_cleanup_entry(
                    entries,
                    _CleanupEntry(
                        path=relative.as_posix(),
                        identity=opened_identity,
                        mode=mode,
                        size=None,
                        content_sha256=None,
                    ),
                )
                _seal_directory_from_fd(
                    child,
                    root_path=root_path,
                    prefix=relative,
                    digest=digest,
                    flags=flags,
                    entries=entries,
                    entry_budget=entry_budget,
                    linux_mount_points=linux_mount_points,
                    require_cleanup_access=require_cleanup_access,
                )
                os.fsync(child)
                if not _Identity.capture(opened).matches(os.fstat(child)):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging directory changed while it was sealed",
                        paths=(relative.as_posix(),),
                    )
            finally:
                _close_descriptor(child)
            continue
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            child = os.open(name, file_flags, dir_fd=descriptor)
        except PermissionError as exc:
            if require_cleanup_access:
                raise _cleanup_access_error(relative.as_posix()) from exc
            raise
        try:
            opened = os.fstat(child)
            if not _Identity.capture(value).matches(opened):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging file changed while it was sealed",
                    paths=(relative.as_posix(),),
                )
            if require_cleanup_access:
                _require_posix_cleanup_flags(
                    opened,
                    descriptor=child,
                    relative=relative.as_posix(),
                )
            opened_observation = _FileMutationObservation.capture(opened)
            content_digest = hashlib.sha256()
            content_size = 0
            while chunk := os.read(child, 1024 * 1024):
                content_size += len(chunk)
                content_digest.update(chunk)
            os.fsync(child)
            after = os.fstat(child)
            if (
                not _Identity.capture(opened).matches(after)
                or _FileMutationObservation.capture(after) != opened_observation
                or content_size != opened_observation.size
            ):
                raise GuardedTreePublicationError(
                    "staging_changed",
                    "staging file changed while it was sealed",
                    paths=(relative.as_posix(),),
                )
            opened_identity = _capture_stable_identity(
                opened,
                descriptor=child,
            )
        finally:
            _close_descriptor(child)
        digest.update(
            f"f\0{relative.as_posix()}\0{opened_identity.device}\0"
            f"{opened_identity.inode}\0{opened_identity.kind}\0"
            f"{opened_identity.incarnation}\0{opened_observation.mode:o}\0"
            f"{opened_observation.size}\0".encode()
        )
        digest.update(content_digest.digest())
        _append_cleanup_entry(
            entries,
            _CleanupEntry(
                path=relative.as_posix(),
                identity=opened_identity,
                mode=opened_observation.mode,
                size=opened_observation.size,
                content_sha256=f"sha256:{content_digest.hexdigest()}",
            ),
        )


def _append_cleanup_entry(entries: list[_CleanupEntry], entry: _CleanupEntry) -> None:
    if len(entries) >= _TREE_ENTRY_LIMIT:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree exceeds its bounded entry limit",
        )
    entry_path_size = _utf8_size(entry.path)
    if entry_path_size is None:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree contains an invalid relative path",
        )
    if entry_path_size > 4096:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree contains an overlong relative path",
        )
    if len(PurePosixPath(entry.path).parts) > _TREE_DEPTH_LIMIT:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication tree exceeds its bounded depth limit",
            paths=(entry.path[:256],),
        )
    entries.append(entry)


def _bounded_sealing_entry_names(
    directory_entries: Iterator[Any],
    *,
    entry_budget: _TreeEntryBudget,
) -> list[str]:
    names: list[str] = []
    for entry in directory_entries:
        entry_budget.reserve()
        names.append(entry.name)
    names.sort(key=_normalized_name)
    return names


def _seal_tree_on_windows(
    path: Path,
    *,
    expected: _Identity,
    require_cleanup_access: bool,
    entry_limit: int,
) -> tuple[str, tuple[_CleanupEntry, ...]]:
    digest = hashlib.sha256()
    entries: list[_CleanupEntry] = []
    entry_budget = _TreeEntryBudget(entry_limit)
    if require_cleanup_access:
        before = path.stat(follow_symlinks=False)
        if _capture_stable_identity(before, path=path) != expected:
            raise GuardedTreePublicationError("staging_changed", "staging root changed")
        _require_windows_cleanup_access(path, expected=expected, value=before)
    with _windows_directory_namespace_fence(path):
        current = path.stat(follow_symlinks=False)
        if _capture_stable_identity(current, path=path) != expected:
            raise GuardedTreePublicationError("staging_changed", "staging root changed")
        digest.update(
            f"r\0{expected.device}\0{expected.inode}\0{expected.kind}\0"
            f"{expected.incarnation}\0"
            f"{stat.S_IMODE(current.st_mode):o}\0".encode()
        )
        _seal_windows_directory(
            path,
            prefix=PurePosixPath(),
            digest=digest,
            entries=entries,
            entry_budget=entry_budget,
            require_cleanup_access=require_cleanup_access,
        )
        _sync_windows_path(path, directory=True)
        after = path.stat(follow_symlinks=False)
        if _capture_stable_identity(after, path=path) != expected:
            raise GuardedTreePublicationError("staging_changed", "staging root changed")
    return f"sha256:{digest.hexdigest()}", tuple(sorted(entries, key=lambda entry: entry.path))


def _seal_windows_directory(
    path: Path,
    *,
    prefix: PurePosixPath,
    digest: Any,
    entries: list[_CleanupEntry],
    entry_budget: _TreeEntryBudget,
    require_cleanup_access: bool,
) -> None:
    with os.scandir(path) as scanned_entries:
        names = _bounded_sealing_entry_names(
            scanned_entries,
            entry_budget=entry_budget,
        )
    _reject_normalized_aliases(names)
    for name in names:
        child = path / name
        relative = prefix / name
        before = child.stat(follow_symlinks=False)
        _require_supported_tree_entry(before, path=relative.as_posix())
        before_identity = _capture_stable_identity(before, path=child)
        if require_cleanup_access:
            _require_windows_cleanup_access(child, expected=before_identity, value=before)
        mode = stat.S_IMODE(before.st_mode)
        file_observation: _FileMutationObservation | None = None
        if stat.S_ISDIR(before.st_mode):
            with _windows_directory_namespace_fence(child):
                digest.update(
                    f"d\0{relative.as_posix()}\0{before_identity.device}\0"
                    f"{before_identity.inode}\0{before_identity.kind}\0"
                    f"{before_identity.incarnation}\0{mode:o}\0".encode()
                )
                _append_cleanup_entry(
                    entries,
                    _CleanupEntry(
                        path=relative.as_posix(),
                        identity=before_identity,
                        mode=mode,
                        size=None,
                        content_sha256=None,
                    ),
                )
                _seal_windows_directory(
                    child,
                    prefix=relative,
                    digest=digest,
                    entries=entries,
                    entry_budget=entry_budget,
                    require_cleanup_access=require_cleanup_access,
                )
                _sync_windows_path(child, directory=True)
        else:
            content_digest = hashlib.sha256()
            content_size = 0
            with child.open("rb") as source:
                opened = os.fstat(source.fileno())
                if _capture_stable_identity(opened, descriptor=source.fileno()) != before_identity:
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging file changed while it was sealed",
                        paths=(relative.as_posix(),),
                    )
                file_observation = _capture_windows_file_mutation_observation(
                    opened,
                    descriptor=source.fileno(),
                )
                while content := source.read(1024 * 1024):
                    content_size += len(content)
                    content_digest.update(content)
                after_read = os.fstat(source.fileno())
                if (
                    _capture_stable_identity(after_read, descriptor=source.fileno())
                    != before_identity
                    or _capture_windows_file_mutation_observation(
                        after_read,
                        descriptor=source.fileno(),
                    )
                    != file_observation
                    or content_size != file_observation.size
                ):
                    raise GuardedTreePublicationError(
                        "staging_changed",
                        "staging file changed while it was sealed",
                        paths=(relative.as_posix(),),
                    )
            _sync_windows_path(child, directory=False)
            digest.update(
                f"f\0{relative.as_posix()}\0{before_identity.device}\0"
                f"{before_identity.inode}\0{before_identity.kind}\0"
                f"{before_identity.incarnation}\0{file_observation.mode:o}\0"
                f"{content_size}\0".encode()
            )
            digest.update(content_digest.digest())
            _append_cleanup_entry(
                entries,
                _CleanupEntry(
                    path=relative.as_posix(),
                    identity=before_identity,
                    mode=file_observation.mode,
                    size=content_size,
                    content_sha256=f"sha256:{content_digest.hexdigest()}",
                ),
            )
        after = child.stat(follow_symlinks=False)
        if _capture_stable_identity(after, path=child) != before_identity or (
            file_observation is not None
            and _capture_windows_file_mutation_observation(after, path=child) != file_observation
        ):
            raise GuardedTreePublicationError(
                "staging_changed",
                "staging entry changed while it was sealed",
                paths=(relative.as_posix(),),
            )


def _require_exact_stage(record: _Record, *, parent: _Parent) -> None:
    parent.assert_unchanged()
    if record.stage_identity is None or record.stage_sha256 is None:
        raise _conflict(record, "staging authority is incomplete")
    _require_identity(parent, record.stage_name, record.stage_identity, label="staging")
    actual = _seal_tree(parent.path / record.stage_name, expected=record.stage_identity)
    if actual != record.stage_sha256:
        raise _conflict(record, "staging content changed after it was sealed")


def _require_published_stage(record: _Record, *, parent: _Parent) -> None:
    if not _published_stage_matches(record, parent=parent):
        raise _conflict(record, "published tree changed before settlement")


def _published_stage_matches(record: _Record, *, parent: _Parent) -> bool:
    parent.assert_unchanged()
    if record.stage_identity is None or record.stage_sha256 is None:
        raise _conflict(record, "published staging authority is incomplete")
    _require_identity(
        parent,
        record.destination_name,
        record.stage_identity,
        label="published destination",
    )
    actual = _seal_tree(parent.path / record.destination_name, expected=record.stage_identity)
    return actual == record.stage_sha256


def _reuse_or_retire_exact_receipt(receipt: _Journal, *, parent: _Parent) -> bool:
    """Reuse only current content; retire a stale absent-or-empty publication."""

    record = receipt.record
    if _published_stage_matches(record, parent=parent):
        return True
    if record.policy is not DestinationPolicy.ABSENT_OR_EMPTY:
        raise _conflict(record, "published tree changed after settlement")
    if record.stage_identity is None:
        raise _conflict(record, "published staging authority is incomplete")
    if not _directory_is_empty(
        parent,
        record.destination_name,
        expected=record.stage_identity,
    ):
        raise GuardedTreePublicationError(
            "destination_not_empty",
            "publication destination must be absent or empty",
            paths=(record.destination_name,),
        )
    _require_parent_namespace_mutable(parent)
    _remove_journal(receipt, parent=parent)
    return False


def _retire_stale_settled_publication_if_safe(
    journal: _Journal,
    *,
    parent: _Parent,
) -> bool:
    """Retire terminal history only when no current or private owner remains."""

    record = journal.record
    if record.phase is not _Phase.SETTLED:
        return False
    _require_record_authority(journal, parent=parent)
    if record.parent_identity != parent.identity:
        raise _conflict(record, "publication parent does not match its durable authority")
    current = parent.entry_stat(record.destination_name)
    if current is not None:
        _require_plain_directory(
            current,
            label="destination",
            path=record.destination_name,
        )
        current_identity = parent.entry_identity(record.destination_name, value=current)
        if current_identity == record.stage_identity and _published_stage_matches(
            record,
            parent=parent,
        ):
            return False
        if not _directory_is_empty(
            parent,
            record.destination_name,
            expected=current_identity,
        ):
            return False
    private_names = (
        record.stage_name,
        record.backup_name,
        _cleanup_name(record),
        _cleanup_manifest_name(record),
    )
    if any(parent.entry_stat(name) is not None for name in private_names):
        return False
    if _pending_metadata_candidates(parent, _cleanup_manifest_name(record)):
        return False
    _require_parent_namespace_mutable(parent)
    _remove_journal(journal, parent=parent)
    return True


def _require_published_identity(record: _Record, *, parent: _Parent) -> None:
    if record.stage_identity is None:
        raise _conflict(record, "published staging authority is incomplete")
    current = parent.entry_stat(record.destination_name)
    if (
        current is None
        or parent.entry_identity(record.destination_name, value=current) != record.stage_identity
    ):
        raise _conflict(record, "published destination identity changed")
    _require_plain_directory(
        current,
        label="published destination",
        path=record.destination_name,
    )


def _create_journal(path: Path, *, record: _Record, parent: _Parent) -> _Journal:
    entry, entry_sha256 = _journal_entry(record, previous_sha256=None)
    temporary_name = _pending_metadata_name(path.name, entry)
    identity = _publish_new_regular_file(
        parent,
        final_name=path.name,
        temporary_name=temporary_name,
        content=entry,
        created_fault_phase="journal_temp_created",
        written_fault_phase="journal_temp_written",
        fault_phase="journal_temp_synced",
    )
    return _Journal(
        path=path,
        identity=identity,
        record=record,
        entry_sha256=entry_sha256,
        valid_bytes=len(entry),
    )


def _pending_metadata_name(final_name: str, content: bytes) -> str:
    return f"{final_name}.pending-{hashlib.sha256(content).hexdigest()}"


def _pending_metadata_candidates(parent: _Parent, final_name: str) -> tuple[str, ...]:
    prefix = f"{final_name}.pending-"
    candidates: list[str] = []
    try:
        with os.scandir(parent.path if parent.descriptor is None else parent.descriptor) as entries:
            for entry in _bounded_parent_directory_entries(
                entries,
                overflow=lambda: _invalid_journal(
                    "pending publication metadata discovery exceeds its bounded parent-entry limit"
                ),
            ):
                if not entry.name.startswith(prefix):
                    continue
                candidates.append(entry.name)
                if len(candidates) > 1:
                    raise _invalid_journal("multiple pending publication metadata files exist")
    except OSError as exc:
        raise _invalid_journal("pending publication metadata could not be inspected") from exc
    return tuple(candidates)


def _read_pending_metadata(
    parent: _Parent,
    name: str,
    *,
    limit: int,
) -> tuple[_Identity, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        if parent.descriptor is None:
            descriptor = os.open(parent.path / name, flags)
        else:
            descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    except OSError as exc:
        raise _invalid_journal("pending publication metadata could not be opened") from exc
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or (os.name != "nt" and stat.S_IMODE(value.st_mode) & 0o077)
            or value.st_nlink != 1
        ):
            raise _invalid_journal("pending publication metadata is not a private regular file")
        identity = _capture_stable_identity(value, descriptor=descriptor)
        content = _read_bounded(descriptor, limit=limit)
        after = os.fstat(descriptor)
        if _capture_stable_identity(
            after, descriptor=descriptor
        ) != identity or after.st_size != len(content):
            raise _invalid_journal("pending publication metadata changed while it was read")
    finally:
        _close_descriptor(descriptor)
    if len(content) > limit:
        raise _invalid_journal("pending publication metadata exceeds its size limit")
    current = parent.entry_stat(name)
    if current is None or parent.entry_identity(name, value=current) != identity:
        raise _invalid_journal("pending publication metadata was replaced while it was read")
    return identity, content


def _remove_exact_pending_metadata(parent: _Parent, name: str, *, expected: _Identity) -> None:
    current = parent.entry_stat(name)
    if (
        current is None
        or parent.entry_identity(name, value=current) != expected
        or not stat.S_ISREG(current.st_mode)
    ):
        raise _invalid_journal("pending publication metadata changed before cleanup")
    if parent.descriptor is None:
        _delete_windows_entry_by_handle(parent.path / name, expected=expected)
    else:
        os.unlink(name, dir_fd=parent.descriptor)
    parent.sync()


def _sync_exact_pending_metadata(parent: _Parent, name: str, *, expected: _Identity) -> None:
    access = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    flags = access | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        if parent.descriptor is None:
            descriptor = os.open(parent.path / name, flags)
        else:
            descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    except OSError as exc:
        raise _invalid_journal("pending publication metadata could not be synchronized") from exc
    try:
        current = os.fstat(descriptor)
        if _capture_stable_identity(current, descriptor=descriptor) != expected or not stat.S_ISREG(
            current.st_mode
        ):
            raise _invalid_journal("pending publication metadata changed before synchronization")
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise _invalid_journal(
                "pending publication metadata could not be synchronized"
            ) from exc
        opened = os.fstat(descriptor)
        if _capture_stable_identity(opened, descriptor=descriptor) != expected:
            raise _invalid_journal("pending publication metadata changed during synchronization")
    finally:
        _close_descriptor(descriptor)
    current = parent.entry_stat(name)
    if current is None or parent.entry_identity(name, value=current) != expected:
        raise _invalid_journal("pending publication metadata was replaced during synchronization")


def _publish_new_regular_file(
    parent: _Parent,
    *,
    final_name: str,
    temporary_name: str,
    content: bytes,
    created_fault_phase: str | None = None,
    written_fault_phase: str | None = None,
    fault_phase: str | None = None,
) -> _Identity:
    _require_parent_namespace_mutable(parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    temporary_identity: _Identity | None = None
    published = False
    operation_error: BaseException | None = None
    try:
        if parent.descriptor is None:
            descriptor = os.open(parent.path / temporary_name, flags, 0o600)
        else:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent.descriptor)
        temporary_value = os.fstat(descriptor)
        temporary_identity = _capture_stable_identity(
            temporary_value,
            descriptor=descriptor,
        )
        if created_fault_phase is not None:
            _publication_fault(created_fault_phase)
        _write_all(descriptor, content)
        if written_fault_phase is not None:
            _publication_fault(written_fault_phase)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if _capture_stable_identity(after, descriptor=descriptor) != temporary_identity:
            raise _invalid_journal("temporary metadata identity changed while it was written")
        opened_descriptor = descriptor
        descriptor = None
        _close_descriptor(opened_descriptor)
        if fault_phase is not None:
            _publication_fault(fault_phase)
        _rename_name_no_replace(parent, temporary_name, final_name)
        current = parent.entry_stat(final_name)
        if (
            current is None
            or parent.entry_identity(final_name, value=current) != temporary_identity
            or not stat.S_ISREG(current.st_mode)
        ):
            raise _invalid_journal("atomically published metadata identity changed")
        published = True
        parent.sync()
        return temporary_identity
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if descriptor is not None:
            _close_descriptor(descriptor)
        if not published and temporary_identity is not None:
            current = parent.entry_stat(temporary_name)
            if (
                current is not None
                and parent.entry_identity(temporary_name, value=current) == temporary_identity
            ):
                try:
                    if parent.descriptor is None:
                        _delete_windows_entry_by_handle(
                            parent.path / temporary_name,
                            expected=temporary_identity,
                        )
                    else:
                        os.unlink(temporary_name, dir_fd=parent.descriptor)
                    parent.sync()
                except BaseException as cleanup_error:
                    if operation_error is None:
                        raise
                    _raise_primary_with_secondary_failure(
                        operation_error,
                        cleanup_error,
                        group_message=(
                            "Guarded publication metadata creation and temporary cleanup failures."
                        ),
                    )


def _load_journal(path: Path, *, parent: _Parent) -> _Journal:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if parent.descriptor is None:
        descriptor = os.open(path, flags)
    else:
        descriptor = os.open(path.name, flags, dir_fd=parent.descriptor)
    try:
        identity_value = os.fstat(descriptor)
        if not stat.S_ISREG(identity_value.st_mode):
            raise _invalid_journal("publication journal is not a regular file")
        identity = _capture_stable_identity(identity_value, descriptor=descriptor)
        content = _read_bounded(descriptor, limit=_JOURNAL_LIMIT_BYTES)
    finally:
        _close_descriptor(descriptor)
    if len(content) > _JOURNAL_LIMIT_BYTES:
        raise _invalid_journal("publication journal exceeds its size limit")
    offset = 0
    previous: str | None = None
    record: _Record | None = None
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        record, entry_sha256 = _parse_journal_entry(
            line,
            previous_sha256=previous,
            expected_sequence=0 if record is None else record.sequence + 1,
            immutable=record,
        )
        previous = entry_sha256
        offset += len(line)
    if record is None or previous is None:
        raise _invalid_journal("publication journal has no complete entry")
    current = parent.entry_stat(path.name)
    if current is None or parent.entry_identity(path.name, value=current) != identity:
        raise _invalid_journal("publication journal was replaced while it was read")
    return _Journal(
        path=path,
        identity=identity,
        record=record,
        entry_sha256=previous,
        valid_bytes=offset,
    )


def _append_journal(journal: _Journal, record: _Record, *, parent: _Parent) -> None:
    record = replace(record, sequence=journal.record.sequence + 1)
    _validate_record(record)
    _require_record_successor(journal.record, record)
    entry, entry_sha256 = _journal_entry(record, previous_sha256=journal.entry_sha256)
    if journal.valid_bytes + len(entry) > _JOURNAL_LIMIT_BYTES:
        raise GuardedTreePublicationError(
            "journal_limit",
            "publication journal exhausted its bounded transition budget",
        )
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if parent.descriptor is None:
        descriptor = os.open(journal.path, flags)
    else:
        descriptor = os.open(journal.path.name, flags, dir_fd=parent.descriptor)
    try:
        opened = os.fstat(descriptor)
        if _capture_stable_identity(opened, descriptor=descriptor) != journal.identity:
            raise _invalid_journal("publication journal identity changed")
        os.ftruncate(descriptor, journal.valid_bytes)
        os.lseek(descriptor, journal.valid_bytes, os.SEEK_SET)
        _write_all(descriptor, entry)
        os.fsync(descriptor)
    finally:
        _close_descriptor(descriptor)
    current = parent.entry_stat(journal.path.name)
    if (
        current is None
        or parent.entry_identity(journal.path.name, value=current) != journal.identity
    ):
        raise _invalid_journal("publication journal was replaced during update")
    journal.record = record
    journal.entry_sha256 = entry_sha256
    journal.valid_bytes += len(entry)


def _promote_terminal_receipt(
    journal: _Journal,
    *,
    receipt_path: Path,
    parent: _Parent,
) -> None:
    """Publish the latest settled journal as the sole terminal receipt."""

    if journal.record.phase is not _Phase.SETTLED:
        raise _invalid_journal("only a settled publication can become a terminal receipt")
    if journal.path == receipt_path:
        _require_record_authority(journal, parent=parent)
        parent.sync()
        return

    active = parent.entry_stat(journal.path.name)
    receipt = parent.entry_stat(receipt_path.name)
    if active is None:
        if (
            receipt is None
            or parent.entry_identity(receipt_path.name, value=receipt) != journal.identity
        ):
            raise _conflict(
                journal.record,
                "terminal receipt publication acknowledgement is ambiguous",
            )
        journal.path = receipt_path
        parent.sync()
        return
    if parent.entry_identity(
        journal.path.name, value=active
    ) != journal.identity or not stat.S_ISREG(active.st_mode):
        raise _invalid_journal("active publication journal identity changed")

    if receipt is not None:
        previous = _load_journal(receipt_path, parent=parent)
        if (
            previous.record.phase is not _Phase.SETTLED
            or journal.record.predecessor_request_digest is None
            or journal.record.predecessor_receipt_identity is None
            or journal.record.predecessor_receipt_sha256 is None
            or journal.record.policy is not DestinationPolicy.REPLACE_DIRECTORY
            or previous.record.request_digest != journal.record.predecessor_request_digest
            or previous.identity != journal.record.predecessor_receipt_identity
            or previous.entry_sha256 != journal.record.predecessor_receipt_sha256
        ):
            raise _conflict(
                journal.record,
                "existing terminal receipt does not match successor authority",
            )
        _remove_journal(previous, parent=parent)

    _rename_exact_regular_file_no_replace(
        parent,
        journal.path.name,
        receipt_path.name,
        expected=journal.identity,
    )
    journal.path = receipt_path
    parent.sync()


def _journal_entry(record: _Record, *, previous_sha256: str | None) -> tuple[bytes, str]:
    payload = record.payload(previous_sha256=previous_sha256)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()
    encoded = (
        json.dumps(
            {**payload, "entry_sha256": digest},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    return encoded, digest


def _parse_journal_entry(
    line: bytes,
    *,
    previous_sha256: str | None,
    expected_sequence: int,
    immutable: _Record | None,
) -> tuple[_Record, str]:
    try:
        value = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise _invalid_journal("publication journal contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise _invalid_journal("publication journal entry is not an object")
    expected_keys = {
        "schema_version",
        "sequence",
        "previous_sha256",
        "consumer",
        "request_digest",
        "predecessor_request_digest",
        "predecessor_receipt_identity",
        "predecessor_receipt_sha256",
        "token",
        "destination_name",
        "policy",
        "parent_identity",
        "original_identity",
        "original_sha256",
        "stage_name",
        "stage_identity",
        "stage_sha256",
        "backup_name",
        "cleanup_manifest_identity",
        "cleanup_manifest_sha256",
        "phase",
        "entry_sha256",
    }
    if set(value) != expected_keys:
        raise _invalid_journal("publication journal entry has unexpected fields")
    entry_sha256 = value.pop("entry_sha256")
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if not isinstance(entry_sha256, str) or entry_sha256 != hashlib.sha256(canonical).hexdigest():
        raise _invalid_journal("publication journal entry digest does not match")
    if type(value["schema_version"]) is not int or value["schema_version"] != (
        _JOURNAL_SCHEMA_VERSION
    ):
        raise _invalid_journal("unsupported publication journal schema")
    if (
        type(value["sequence"]) is not int
        or value["sequence"] != expected_sequence
        or value["previous_sha256"] != previous_sha256
    ):
        raise _invalid_journal("publication journal chain is not contiguous")
    try:
        policy = DestinationPolicy(value["policy"])
        phase = _Phase(value["phase"])
    except (TypeError, ValueError) as exc:
        raise _invalid_journal("publication journal contains an unknown enum") from exc
    parent_identity = _Identity.from_json(value["parent_identity"], field="parent")
    original_raw = value["original_identity"]
    original_identity = (
        None if original_raw is None else _Identity.from_json(original_raw, field="original")
    )
    original_sha256 = value["original_sha256"]
    if original_sha256 is not None and not _is_sha256(original_sha256):
        raise _invalid_journal("publication journal contains an invalid original digest")
    stage_raw = value["stage_identity"]
    stage_identity = None if stage_raw is None else _Identity.from_json(stage_raw, field="stage")
    strings = {
        key: value[key]
        for key in (
            "consumer",
            "request_digest",
            "token",
            "destination_name",
            "stage_name",
            "backup_name",
        )
    }
    if any(not isinstance(item, str) for item in strings.values()):
        raise _invalid_journal("publication journal contains a non-string identity field")
    predecessor_raw = value["predecessor_request_digest"]
    if predecessor_raw is not None and not isinstance(predecessor_raw, str):
        raise _invalid_journal("publication journal contains an invalid predecessor digest")
    predecessor_request_digest = cast("str | None", predecessor_raw)
    predecessor_receipt_raw = value["predecessor_receipt_identity"]
    predecessor_receipt_identity = (
        None
        if predecessor_receipt_raw is None
        else _Identity.from_json(predecessor_receipt_raw, field="predecessor receipt")
    )
    predecessor_receipt_sha256 = value["predecessor_receipt_sha256"]
    if predecessor_receipt_sha256 is not None and (
        not isinstance(predecessor_receipt_sha256, str)
        or len(predecessor_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in predecessor_receipt_sha256)
    ):
        raise _invalid_journal("publication journal contains an invalid predecessor receipt digest")
    stage_sha256 = value["stage_sha256"]
    if stage_sha256 is not None and not _is_sha256(stage_sha256):
        raise _invalid_journal("publication journal contains an invalid stage digest")
    cleanup_manifest_raw = value["cleanup_manifest_identity"]
    cleanup_manifest_identity = (
        None
        if cleanup_manifest_raw is None
        else _Identity.from_json(cleanup_manifest_raw, field="cleanup manifest")
    )
    cleanup_manifest_sha256 = value["cleanup_manifest_sha256"]
    if cleanup_manifest_sha256 is not None and not _is_sha256(cleanup_manifest_sha256):
        raise _invalid_journal("publication journal contains an invalid cleanup manifest digest")
    record = _Record(
        consumer=strings["consumer"],
        request_digest=strings["request_digest"],
        predecessor_request_digest=predecessor_request_digest,
        predecessor_receipt_identity=predecessor_receipt_identity,
        predecessor_receipt_sha256=predecessor_receipt_sha256,
        token=strings["token"],
        destination_name=strings["destination_name"],
        policy=policy,
        parent_identity=parent_identity,
        original_identity=original_identity,
        original_sha256=original_sha256,
        stage_name=strings["stage_name"],
        stage_identity=stage_identity,
        stage_sha256=stage_sha256,
        backup_name=strings["backup_name"],
        cleanup_manifest_identity=cleanup_manifest_identity,
        cleanup_manifest_sha256=cleanup_manifest_sha256,
        phase=phase,
        sequence=expected_sequence,
    )
    _validate_record(record)
    if immutable is not None:
        _require_record_successor(immutable, record)
    return record, entry_sha256


def _require_record_successor(previous: _Record, record: _Record) -> None:
    if (
        record.consumer,
        record.request_digest,
        record.predecessor_request_digest,
        record.predecessor_receipt_identity,
        record.predecessor_receipt_sha256,
        record.token,
        record.destination_name,
        record.policy,
        record.parent_identity,
        record.original_identity,
        record.original_sha256,
        record.stage_name,
        record.backup_name,
    ) != (
        previous.consumer,
        previous.request_digest,
        previous.predecessor_request_digest,
        previous.predecessor_receipt_identity,
        previous.predecessor_receipt_sha256,
        previous.token,
        previous.destination_name,
        previous.policy,
        previous.parent_identity,
        previous.original_identity,
        previous.original_sha256,
        previous.stage_name,
        previous.backup_name,
    ):
        raise _invalid_journal("publication journal immutable authority changed")
    if record.sequence != previous.sequence + 1:
        raise _invalid_journal("publication journal sequence transition is invalid")
    if record.phase not in _ALLOWED_PHASE_TRANSITIONS[previous.phase]:
        raise _invalid_journal("publication journal phase transition is invalid")
    if previous.stage_identity is not None and (record.stage_identity != previous.stage_identity):
        raise _invalid_journal("publication journal staging identity changed")
    if previous.stage_sha256 is not None and (record.stage_sha256 != previous.stage_sha256):
        raise _invalid_journal("publication journal staging digest changed")
    if previous.cleanup_manifest_identity is not None and (
        record.cleanup_manifest_identity != previous.cleanup_manifest_identity
        or record.cleanup_manifest_sha256 != previous.cleanup_manifest_sha256
    ):
        raise _invalid_journal("publication journal cleanup manifest authority changed")
    if (
        previous.cleanup_manifest_identity is None
        and record.cleanup_manifest_identity is not None
        and record.phase not in {_Phase.CLEANUP_SEALED, _Phase.ROLLBACK_CLEANUP_SEALED}
    ):
        raise _invalid_journal("publication journal acquired cleanup authority unexpectedly")


def _validate_record(record: _Record) -> None:
    consumer_size = _utf8_size(record.consumer)
    destination_size = _utf8_size(record.destination_name)
    if (
        not record.consumer
        or consumer_size is None
        or consumer_size > 128
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in record.consumer
        )
        or not _is_sha256(record.request_digest)
        or (
            record.predecessor_request_digest is not None
            and not _is_sha256(record.predecessor_request_digest)
        )
        or (
            record.predecessor_request_digest is not None
            and record.policy is not DestinationPolicy.REPLACE_DIRECTORY
        )
        or (
            record.predecessor_receipt_identity is not None
            and record.predecessor_receipt_identity.kind != stat.S_IFREG
        )
        or len(record.token) != 32
        or any(character not in "0123456789abcdef" for character in record.token)
        or destination_size is None
        or destination_size > 255
        or Path(record.destination_name).name != record.destination_name
        or record.destination_name in {".", ".."}
        or any(
            ord(character) < 32 or ord(character) == 127 for character in record.destination_name
        )
        or (
            os.name == "nt"
            and (
                "\\" in record.destination_name
                or _is_unsafe_windows_component(record.destination_name)
            )
        )
        or record.stage_name != f".cayu-tree-stage-{record.token}"
        or record.backup_name != f".cayu-tree-backup-{record.token}"
        or record.parent_identity.kind != stat.S_IFDIR
        or (record.original_identity is not None and record.original_identity.kind != stat.S_IFDIR)
        or (record.stage_identity is not None and record.stage_identity.kind != stat.S_IFDIR)
        or (
            record.cleanup_manifest_identity is not None
            and record.cleanup_manifest_identity.kind != stat.S_IFREG
        )
    ):
        raise _invalid_journal("publication journal identity is invalid")
    if (record.original_identity is None) != (record.original_sha256 is None):
        raise _invalid_journal("publication journal original authority is incomplete")
    if not (
        (record.predecessor_request_digest is None)
        == (record.predecessor_receipt_identity is None)
        == (record.predecessor_receipt_sha256 is None)
    ):
        raise _invalid_journal("publication journal predecessor authority is incomplete")
    if (record.cleanup_manifest_identity is None) != (record.cleanup_manifest_sha256 is None):
        raise _invalid_journal("publication journal cleanup manifest authority is incomplete")
    if record.sequence == 0 and (
        record.phase is not _Phase.PREPARED
        or record.stage_identity is not None
        or record.stage_sha256 is not None
        or record.cleanup_manifest_identity is not None
    ):
        raise _invalid_journal("publication journal initial authority is invalid")
    if record.phase is _Phase.STAGING and (
        record.stage_identity is None or record.stage_sha256 is not None
    ):
        raise _invalid_journal("publication journal staging phase is invalid")
    if record.phase in {
        _Phase.STAGED,
        _Phase.COMMIT_INTENT,
        _Phase.ORIGINAL_BACKED_UP,
        _Phase.PUBLISHED,
        _Phase.CLEANUP_OWNED,
        _Phase.CLEANUP_SEALED,
        _Phase.SETTLED,
    } and (record.stage_identity is None or record.stage_sha256 is None):
        raise _invalid_journal("publication journal phase lacks sealed staging authority")
    if record.phase in {
        _Phase.ORIGINAL_BACKED_UP,
        _Phase.CLEANUP_OWNED,
        _Phase.CLEANUP_SEALED,
    } and (record.original_identity is None):
        raise _invalid_journal("publication journal phase lacks original authority")
    if record.phase in {_Phase.CLEANUP_SEALED, _Phase.ROLLBACK_CLEANUP_SEALED} and (
        record.cleanup_manifest_identity is None
    ):
        raise _invalid_journal("publication journal phase lacks cleanup manifest authority")


def _remove_journal(journal: _Journal, *, parent: _Parent) -> None:
    _require_record_authority(journal, parent=parent)
    if parent.descriptor is None:
        _delete_windows_entry_by_handle(journal.path, expected=journal.identity)
    else:
        os.unlink(journal.path.name, dir_fd=parent.descriptor)
    parent.sync()


def _require_record_authority(journal: _Journal, *, parent: _Parent) -> None:
    current = parent.entry_stat(journal.path.name)
    if (
        current is None
        or parent.entry_identity(journal.path.name, value=current) != journal.identity
        or not stat.S_ISREG(current.st_mode)
    ):
        raise _invalid_journal("publication journal identity changed")


def _cleanup_name(record: _Record) -> str:
    return f".cayu-tree-cleanup-{record.token}"


def _cleanup_manifest_name(record: _Record) -> str:
    return f".cayu-tree-authority-{record.token}.json"


def _cleanup_manifest_pending_name(record: _Record, content: bytes) -> str:
    return _pending_metadata_name(_cleanup_manifest_name(record), content)


def _cleanup_manifest_content(
    record: _Record,
    *,
    cleanup_name: str,
    root_identity: _Identity,
    root_sha256: str,
    entries: tuple[_CleanupEntry, ...],
) -> bytes:
    payload = {
        "schema_version": _CLEANUP_MANIFEST_SCHEMA_VERSION,
        "token": record.token,
        "cleanup_name": cleanup_name,
        "root_identity": root_identity.as_json(),
        "root_sha256": root_sha256,
        "entries": [entry.as_json() for entry in entries],
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    if len(encoded) > _CLEANUP_MANIFEST_LIMIT_BYTES:
        raise GuardedTreePublicationError(
            "tree_limit",
            "guarded publication cleanup authority exceeds its bounded size limit",
        )
    return encoded


def _read_cleanup_manifest(
    parent: _Parent,
    record: _Record,
    *,
    cleanup_name: str,
    expected_root: _Identity,
    expected_identity: _Identity | None,
    expected_sha256: str | None,
    manifest_name: str | None = None,
) -> _CleanupManifest:
    if manifest_name is None:
        manifest_name = _cleanup_manifest_name(record)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if parent.descriptor is None:
        descriptor = os.open(parent.path / manifest_name, flags)
    else:
        descriptor = os.open(manifest_name, flags, dir_fd=parent.descriptor)
    try:
        opened = os.fstat(descriptor)
        identity = _capture_stable_identity(opened, descriptor=descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest is not a regular file",
                paths=(manifest_name,),
            )
        if expected_identity is not None and identity != expected_identity:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest identity changed",
                paths=(manifest_name,),
            )
        content = _read_bounded(descriptor, limit=_CLEANUP_MANIFEST_LIMIT_BYTES)
    finally:
        _close_descriptor(descriptor)
    if len(content) > _CLEANUP_MANIFEST_LIMIT_BYTES:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest exceeds its bounded size limit",
            paths=(manifest_name,),
        )
    content_sha256 = f"sha256:{hashlib.sha256(content).hexdigest()}"
    if expected_sha256 is not None and content_sha256 != expected_sha256:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest content changed",
            paths=(manifest_name,),
        )
    try:
        value = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest contains invalid JSON",
            paths=(manifest_name,),
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "token",
        "cleanup_name",
        "root_identity",
        "root_sha256",
        "entries",
    }:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest has an invalid shape",
            paths=(manifest_name,),
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != _CLEANUP_MANIFEST_SCHEMA_VERSION
        or value["token"] != record.token
        or value["cleanup_name"] != cleanup_name
        or value["root_sha256"] is None
        or not _is_sha256(value["root_sha256"])
    ):
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest authority does not match the publication",
            paths=(manifest_name,),
        )
    root_identity = _Identity.from_json(value["root_identity"], field="cleanup root")
    if root_identity != expected_root:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest root identity does not match the publication",
            paths=(manifest_name,),
        )
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > _TREE_ENTRY_LIMIT:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest entry collection is invalid",
            paths=(manifest_name,),
        )
    entries: list[_CleanupEntry] = []
    previous_path: str | None = None
    directory_paths: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "identity",
            "mode",
            "size",
            "content_sha256",
        }:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest contains an invalid entry",
                paths=(manifest_name,),
            )
        raw_entry = cast("dict[str, object]", raw_entry)
        relative = raw_entry["path"]
        if not isinstance(relative, str):
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest contains an invalid path",
                paths=(manifest_name,),
            )
        relative_size = _utf8_size(relative)
        path_value = PurePosixPath(relative)
        if (
            not relative
            or path_value.is_absolute()
            or path_value.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path_value.parts)
            or len(path_value.parts) > _TREE_DEPTH_LIMIT
            or relative_size is None
            or relative_size > 4096
            or (previous_path is not None and relative <= previous_path)
        ):
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest path ordering is invalid",
                paths=(manifest_name,),
            )
        if len(path_value.parts) > 1 and path_value.parent.as_posix() not in directory_paths:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest path topology is invalid",
                paths=(manifest_name,),
            )
        entry_identity = _Identity.from_json(
            raw_entry["identity"],
            field=f"cleanup entry {index}",
        )
        mode = raw_entry["mode"]
        size = raw_entry["size"]
        entry_sha256 = raw_entry["content_sha256"]
        if type(mode) is not int or not 0 <= mode <= 0o7777:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest contains an invalid mode",
                paths=(manifest_name,),
            )
        if entry_identity.kind == stat.S_IFDIR:
            if size is not None or entry_sha256 is not None:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "cleanup directory authority is invalid",
                    paths=(manifest_name,),
                )
            directory_paths.add(relative)
        elif entry_identity.kind == stat.S_IFREG:
            if (
                type(size) is not int
                or size < 0
                or not isinstance(entry_sha256, str)
                or not _is_sha256(entry_sha256)
            ):
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "cleanup file authority is invalid",
                    paths=(manifest_name,),
                )
        else:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup manifest contains an unsupported entry type",
                paths=(manifest_name,),
            )
        entries.append(
            _CleanupEntry(
                path=relative,
                identity=entry_identity,
                mode=mode,
                size=size,
                content_sha256=entry_sha256,
            )
        )
        previous_path = relative
    return _CleanupManifest(
        path=parent.path / manifest_name,
        identity=identity,
        content_sha256=content_sha256,
        token=record.token,
        cleanup_name=cleanup_name,
        root_identity=root_identity,
        root_sha256=cast("str", value["root_sha256"]),
        entries=tuple(entries),
    )


def _prepare_cleanup_manifest(
    journal: _Journal,
    parent: _Parent,
    *,
    cleanup_name: str,
    root_identity: _Identity,
    expected_root_sha256: str | None,
) -> _CleanupManifest:
    path = parent.path / cleanup_name
    actual_sha256, entries = _capture_tree_authority(path, expected=root_identity)
    if expected_root_sha256 is not None and actual_sha256 != expected_root_sha256:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "claimed cleanup descendants do not match their sealed authority",
            paths=(cleanup_name,),
        )
    content = _cleanup_manifest_content(
        journal.record,
        cleanup_name=cleanup_name,
        root_identity=root_identity,
        root_sha256=actual_sha256,
        entries=entries,
    )
    manifest_name = _cleanup_manifest_name(journal.record)
    pending_name = _cleanup_manifest_pending_name(journal.record, content)
    pending_candidates = _pending_metadata_candidates(parent, manifest_name)
    if pending_candidates and pending_candidates != (pending_name,):
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "a pending cleanup manifest belongs to different authority",
            paths=pending_candidates,
        )
    manifest_entry = parent.entry_stat(manifest_name)
    pending_entry = parent.entry_stat(pending_name)
    if manifest_entry is not None and pending_entry is not None:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "active and pending cleanup manifests both exist",
            paths=(manifest_name, pending_name),
        )
    if manifest_entry is None:
        if pending_entry is not None:
            pending_identity, pending_content = _read_pending_metadata(
                parent,
                pending_name,
                limit=_CLEANUP_MANIFEST_LIMIT_BYTES,
            )
            if pending_content != content:
                _remove_exact_pending_metadata(
                    parent,
                    pending_name,
                    expected=pending_identity,
                )
                pending_entry = None
        if pending_entry is None:
            _publish_new_regular_file(
                parent,
                final_name=manifest_name,
                temporary_name=pending_name,
                content=content,
                created_fault_phase="cleanup_manifest_temp_created",
                written_fault_phase="cleanup_manifest_temp_written",
                fault_phase="cleanup_manifest_temp_synced",
            )
        else:
            pending = _read_cleanup_manifest(
                parent,
                journal.record,
                cleanup_name=cleanup_name,
                expected_root=root_identity,
                expected_identity=None,
                expected_sha256=f"sha256:{hashlib.sha256(content).hexdigest()}",
                manifest_name=pending_name,
            )
            _sync_exact_pending_metadata(
                parent,
                pending_name,
                expected=pending.identity,
            )
            _rename_exact_regular_file_no_replace(
                parent,
                pending_name,
                manifest_name,
                expected=pending.identity,
            )
            parent.sync()
    manifest = _read_cleanup_manifest(
        parent,
        journal.record,
        cleanup_name=cleanup_name,
        expected_root=root_identity,
        expected_identity=None,
        expected_sha256=None,
    )
    if manifest.content_sha256 != f"sha256:{hashlib.sha256(content).hexdigest()}":
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "existing cleanup manifest conflicts with the claimed tree",
            paths=(manifest_name,),
        )
    verified_sha256, verified_entries = _capture_tree_authority(path, expected=root_identity)
    if verified_sha256 != manifest.root_sha256 or verified_entries != manifest.entries:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "claimed cleanup tree changed before manifest authority was durable",
            paths=(cleanup_name, manifest_name),
        )
    return manifest


def _load_sealed_cleanup_manifest(
    journal: _Journal,
    parent: _Parent,
    *,
    cleanup_name: str,
    root_identity: _Identity,
) -> _CleanupManifest:
    if (
        journal.record.cleanup_manifest_identity is None
        or journal.record.cleanup_manifest_sha256 is None
    ):
        raise _invalid_journal("sealed cleanup lacks manifest authority")
    return _read_cleanup_manifest(
        parent,
        journal.record,
        cleanup_name=cleanup_name,
        expected_root=root_identity,
        expected_identity=journal.record.cleanup_manifest_identity,
        expected_sha256=journal.record.cleanup_manifest_sha256,
    )


def _mark_cleanup_sealed(
    journal: _Journal,
    manifest: _CleanupManifest,
    *,
    phase: _Phase,
    parent: _Parent,
) -> None:
    _append_journal(
        journal,
        replace(
            journal.record,
            phase=phase,
            cleanup_manifest_identity=manifest.identity,
            cleanup_manifest_sha256=manifest.content_sha256,
        ),
        parent=parent,
    )
    _publication_fault(phase.value)


def _remove_exact_tree(
    parent: _Parent,
    name: str,
    expected: _Identity | None,
    *,
    expected_sha256: str | None,
    cleanup_name: str,
    journal: _Journal,
    sealed_phase: _Phase,
) -> None:
    if expected is None:
        raise GuardedTreePublicationError(
            "cleanup_authority_missing",
            "tree cleanup lacks exact identity authority",
            paths=(name,),
        )
    source_name = name
    current = parent.entry_stat(source_name)
    claimed = parent.entry_stat(cleanup_name)
    if current is not None and claimed is not None:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "owned and claimed cleanup names both exist",
            paths=(source_name, cleanup_name),
        )
    if claimed is None:
        if current is None:
            if journal.record.phase is sealed_phase:
                _remove_cleanup_manifest_if_owned(journal, parent=parent)
                return
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned tree disappeared before cleanup was durably sealed",
                paths=(source_name, cleanup_name),
            )
        if parent.entry_identity(source_name, value=current) != expected:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned tree identity changed before cleanup",
                paths=(source_name, cleanup_name),
            )
        _require_plain_directory(current, label="cleanup tree", path=name)
        _rename_no_replace(
            parent,
            source_name,
            cleanup_name,
            expected=expected,
            label="cleanup tree",
        )
        parent.sync()
        claimed = parent.entry_stat(cleanup_name)
    if claimed is None or parent.entry_identity(cleanup_name, value=claimed) != expected:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "claimed cleanup tree identity changed",
            paths=(source_name, cleanup_name),
        )
    _require_plain_directory(claimed, label="cleanup tree", path=cleanup_name)
    name = cleanup_name
    path = parent.path / cleanup_name
    if journal.record.phase is sealed_phase:
        manifest = _load_sealed_cleanup_manifest(
            journal,
            parent,
            cleanup_name=cleanup_name,
            root_identity=expected,
        )
    else:
        manifest = _prepare_cleanup_manifest(
            journal,
            parent,
            cleanup_name=cleanup_name,
            root_identity=expected,
            expected_root_sha256=expected_sha256,
        )
        _mark_cleanup_sealed(
            journal,
            manifest,
            phase=sealed_phase,
            parent=parent,
        )
    authority = {entry.path: entry for entry in manifest.entries}
    _require_current_tree_within_cleanup_manifest(path, expected=expected, manifest=manifest)
    if os.name == "nt":
        _delete_windows_entry_by_handle(
            path,
            expected=expected,
            authority=authority,
            prefix=PurePosixPath(),
        )
        parent.sync()
        _remove_cleanup_manifest_if_owned(journal, parent=parent, manifest=manifest)
        _require_cleanup_source_absent(parent, source_name, cleanup_name)
        return
    if parent.descriptor is None:
        raise AssertionError("POSIX publication parent has no descriptor")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    try:
        opened = os.fstat(descriptor)
        if _capture_stable_identity(opened, descriptor=descriptor) != expected:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned tree changed while cleanup was pinned",
                paths=(name,),
            )
        _remove_directory_contents_from_fd(
            descriptor,
            path=path,
            flags=flags,
            authority=authority,
            prefix=PurePosixPath(),
        )
    finally:
        _close_descriptor(descriptor)
    current = parent.entry_stat(name)
    if current is None or parent.entry_identity(name, value=current) != expected:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "owned tree changed before final cleanup",
            paths=(name,),
        )
    os.rmdir(name, dir_fd=parent.descriptor)
    parent.sync()
    _remove_cleanup_manifest_if_owned(journal, parent=parent, manifest=manifest)
    _require_cleanup_source_absent(parent, source_name, cleanup_name)


def _require_current_tree_within_cleanup_manifest(
    path: Path,
    *,
    expected: _Identity,
    manifest: _CleanupManifest,
) -> None:
    _current_sha256, current_entries = _capture_tree_authority(path, expected=expected)
    authority = {entry.path: entry for entry in manifest.entries}
    for current in current_entries:
        if authority.get(current.path) != current:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "claimed cleanup tree contains replaced or unowned content",
                paths=(current.path,),
            )


def _remove_cleanup_manifest_if_owned(
    journal: _Journal,
    *,
    parent: _Parent,
    manifest: _CleanupManifest | None = None,
) -> None:
    manifest_name = _cleanup_manifest_name(journal.record)
    current = parent.entry_stat(manifest_name)
    if current is None:
        return
    expected = (
        manifest.identity if manifest is not None else journal.record.cleanup_manifest_identity
    )
    if (
        expected is None
        or parent.entry_identity(manifest_name, value=current) != expected
        or not stat.S_ISREG(current.st_mode)
    ):
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "cleanup manifest identity changed before removal",
            paths=(manifest_name,),
        )
    if parent.descriptor is None:
        _delete_windows_entry_by_handle(parent.path / manifest_name, expected=expected)
    else:
        os.unlink(manifest_name, dir_fd=parent.descriptor)
    parent.sync()


def _require_cleanup_source_absent(
    parent: _Parent,
    source_name: str,
    cleanup_name: str,
) -> None:
    if parent.entry_stat(source_name) is not None:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "a conflicting tree appeared after the exact cleanup claim",
            paths=(source_name, cleanup_name),
        )


def _remove_directory_contents_from_fd(
    descriptor: int,
    *,
    path: Path,
    flags: int,
    authority: dict[str, _CleanupEntry] | None = None,
    prefix: PurePosixPath | None = None,
    linux_mount_points: frozenset[str] | None = None,
) -> None:
    if prefix is None:
        prefix = PurePosixPath()
    if linux_mount_points is None:
        linux_mount_points = _linux_mount_points()
    with os.scandir(descriptor) as entries:
        names = _bounded_cleanup_entry_names(
            entries,
            authority=authority,
            prefix=prefix,
        )
    for name in names:
        child_path = path / name
        relative = prefix / name
        try:
            identity = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        _require_supported_tree_entry(identity, path=str(child_path))
        expected = _capture_stable_identity(
            identity,
            dir_fd=descriptor,
            name=name,
        )
        expected_entry = None if authority is None else authority.get(relative.as_posix())
        captured_entry: _CapturedCleanupEntry | None = None
        if authority is not None and expected_entry is None:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup encountered content absent from its durable authority",
                paths=(relative.as_posix(),),
            )
        if expected_entry is not None:
            captured_entry = _capture_posix_cleanup_entry(
                descriptor,
                name,
                relative=relative,
                value=identity,
            )
            if captured_entry.entry != expected_entry:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "owned cleanup entry changed before deletion",
                    paths=(relative.as_posix(),),
                )
        if stat.S_ISDIR(identity.st_mode):
            if _path_is_mount_boundary(
                child_path,
                parent_device=os.fstat(descriptor).st_dev,
                linux_mount_points=linux_mount_points,
            ):
                raise GuardedTreePublicationError(
                    "mount_boundary",
                    "guarded cleanup cannot traverse a mounted descendant",
                    paths=(relative.as_posix(),),
                )
            child = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if _capture_stable_identity(opened, descriptor=child) != expected:
                    raise GuardedTreePublicationError(
                        "cleanup_conflict",
                        "owned directory changed during cleanup",
                        paths=(name,),
                    )
                _remove_directory_contents_from_fd(
                    child,
                    path=child_path,
                    flags=flags,
                    authority=authority,
                    prefix=relative,
                    linux_mount_points=linux_mount_points,
                )
            finally:
                _close_descriptor(child)
            try:
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if _capture_stable_identity(current, dir_fd=descriptor, name=name) != expected:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "owned directory changed during cleanup",
                    paths=(name,),
                )
            try:
                os.rmdir(name, dir_fd=descriptor)
            except OSError as exc:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "owned cleanup directory acquired conflicting content",
                    paths=(relative.as_posix(),),
                ) from exc
            _publication_fault("cleanup_entry_removed")
        else:
            try:
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if _capture_stable_identity(current, dir_fd=descriptor, name=name) != expected or (
                captured_entry is not None
                and (
                    captured_entry.file_observation is None
                    or _FileMutationObservation.capture(current) != captured_entry.file_observation
                )
            ):
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "owned file changed during cleanup",
                    paths=(name,),
                )
            os.unlink(name, dir_fd=descriptor)
            _publication_fault("cleanup_entry_removed")


def _bounded_cleanup_entry_names(
    directory_entries: Iterator[Any],
    *,
    authority: dict[str, _CleanupEntry] | None,
    prefix: PurePosixPath,
) -> list[str]:
    names: list[str] = []
    for entry in directory_entries:
        name = entry.name
        relative = (prefix / name).as_posix()
        if authority is not None and relative not in authority:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "cleanup encountered content absent from its durable authority",
                paths=(relative,),
            )
        if len(names) >= _TREE_ENTRY_LIMIT:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup tree exceeds its bounded entry authority",
                paths=(prefix.as_posix(),),
            )
        names.append(name)
    return names


def _capture_posix_cleanup_entry(
    descriptor: int,
    name: str,
    *,
    relative: PurePosixPath,
    value: os.stat_result,
) -> _CapturedCleanupEntry:
    identity = _capture_stable_identity(
        value,
        dir_fd=descriptor,
        name=name,
    )
    mode = stat.S_IMODE(value.st_mode)
    if stat.S_ISDIR(value.st_mode):
        return _CapturedCleanupEntry(
            entry=_CleanupEntry(
                path=relative.as_posix(),
                identity=identity,
                mode=mode,
                size=None,
                content_sha256=None,
            ),
            file_observation=None,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    child = os.open(name, flags, dir_fd=descriptor)
    try:
        opened = os.fstat(child)
        if _capture_stable_identity(opened, descriptor=child) != identity:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup file changed while it was inspected",
                paths=(relative.as_posix(),),
            )
        opened_observation = _FileMutationObservation.capture(opened)
        content_digest = hashlib.sha256()
        content_size = 0
        while chunk := os.read(child, 1024 * 1024):
            content_size += len(chunk)
            content_digest.update(chunk)
        after = os.fstat(child)
        if (
            _capture_stable_identity(after, descriptor=child) != identity
            or _FileMutationObservation.capture(after) != opened_observation
            or content_size != opened_observation.size
        ):
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup file changed while it was inspected",
                paths=(relative.as_posix(),),
            )
    finally:
        _close_descriptor(child)
    return _CapturedCleanupEntry(
        entry=_CleanupEntry(
            path=relative.as_posix(),
            identity=identity,
            mode=opened_observation.mode,
            size=opened_observation.size,
            content_sha256=f"sha256:{content_digest.hexdigest()}",
        ),
        file_observation=opened_observation,
    )


def _rename_exact_regular_file_no_replace(
    parent: _Parent,
    source: str,
    destination: str,
    *,
    expected: _Identity,
) -> None:
    parent.assert_unchanged()
    current = parent.entry_stat(source)
    if (
        current is None
        or parent.entry_identity(source, value=current) != expected
        or not stat.S_ISREG(current.st_mode)
    ):
        raise _invalid_journal("active publication journal changed before receipt promotion")

    pinned_descriptor: int | None = None
    if parent.descriptor is not None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        pinned_descriptor = os.open(source, flags, dir_fd=parent.descriptor)
        pinned = os.fstat(pinned_descriptor)
        if _capture_stable_identity(pinned, descriptor=pinned_descriptor) != expected:
            conflict = _invalid_journal(
                "active publication journal changed while receipt promotion was pinned"
            )
            try:
                raise conflict
            finally:
                _close_descriptor(pinned_descriptor)
    try:
        _rename_name_no_replace(parent, source, destination)
        moved = parent.entry_stat(destination)
        pinned_matches = pinned_descriptor is None or (
            _capture_stable_identity(
                os.fstat(pinned_descriptor),
                descriptor=pinned_descriptor,
            )
            == expected
        )
        if (
            moved is not None
            and parent.entry_identity(destination, value=moved) == expected
            and pinned_matches
        ):
            return
        conflict = _invalid_journal(
            "active publication journal changed during terminal receipt promotion"
        )
        _restore_unexpected_rename(
            parent,
            source=source,
            destination=destination,
            moved=moved,
            error=conflict,
        )
        raise conflict
    finally:
        if pinned_descriptor is not None:
            _close_descriptor(pinned_descriptor)


def _rename_no_replace(
    parent: _Parent,
    source: str,
    destination: str,
    *,
    expected: _Identity,
    label: str,
) -> None:
    """Move one exact directory without replacing the destination namespace."""

    parent.assert_unchanged()
    current = parent.entry_stat(source)
    if current is None or parent.entry_identity(source, value=current) != expected:
        raise GuardedTreePublicationError(
            f"{label.replace(' ', '_')}_changed",
            f"{label} identity changed at the guarded rename boundary",
            paths=(source, destination),
        )
    _require_plain_directory(current, label=label, path=source)

    pinned_descriptor: int | None = None
    if parent.descriptor is not None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        pinned_descriptor = os.open(source, flags, dir_fd=parent.descriptor)
        pinned = os.fstat(pinned_descriptor)
        if _capture_stable_identity(pinned, descriptor=pinned_descriptor) != expected:
            conflict = GuardedTreePublicationError(
                f"{label.replace(' ', '_')}_changed",
                f"{label} identity changed while the rename source was pinned",
                paths=(source, destination),
            )
            try:
                raise conflict
            finally:
                _close_descriptor(pinned_descriptor)
    try:
        _rename_name_no_replace(parent, source, destination)
        moved = parent.entry_stat(destination)
        pinned_matches = pinned_descriptor is None or (
            _capture_stable_identity(
                os.fstat(pinned_descriptor),
                descriptor=pinned_descriptor,
            )
            == expected
        )
        if (
            moved is not None
            and parent.entry_identity(destination, value=moved) == expected
            and pinned_matches
        ):
            return
        conflict = GuardedTreePublicationError(
            f"{label.replace(' ', '_')}_changed",
            f"{label} identity changed during the guarded rename",
            paths=(source, destination),
        )
        _restore_unexpected_rename(
            parent,
            source=source,
            destination=destination,
            moved=moved,
            error=conflict,
        )
        raise conflict
    finally:
        if pinned_descriptor is not None:
            _close_descriptor(pinned_descriptor)


def _restore_unexpected_rename(
    parent: _Parent,
    *,
    source: str,
    destination: str,
    moved: os.stat_result | None,
    error: GuardedTreePublicationError,
) -> None:
    if moved is None or parent.entry_stat(source) is not None:
        return
    moved_identity = parent.entry_identity(destination, value=moved)
    try:
        _rename_name_no_replace(parent, destination, source)
        restored = parent.entry_stat(source)
        if restored is None or parent.entry_identity(source, value=restored) != moved_identity:
            raise GuardedTreePublicationError(
                "rename_restoration_conflict",
                "the unexpected rename result changed before it could be restored",
                paths=(source, destination),
            )
        parent.sync()
    except BaseException as restoration_error:
        if restoration_error.__cause__ is None and restoration_error.__context__ is error:
            restoration_error.__context__ = None
        _record_settlement_failure(error, restoration_error)
        error.__cause__ = restoration_error


def _rename_name_no_replace(parent: _Parent, source: str, destination: str) -> None:
    """Perform only the platform namespace operation; callers own source authority."""

    parent.assert_unchanged()
    if parent.entry_stat(destination) is not None:
        raise FileExistsError(errno.EEXIST, "publication destination exists", destination)
    if os.name == "nt":
        os.rename(parent.path / source, parent.path / destination)
        return
    if parent.descriptor is None:
        raise AssertionError("POSIX publication parent has no descriptor")
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise GuardedTreePublicationError(
                "no_replace_unavailable",
                "this platform lacks atomic no-replace directory publication",
            )
        result = renameat2(
            parent.descriptor,
            os.fsencode(source),
            parent.descriptor,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = libc.renameatx_np
        result = renameatx_np(
            parent.descriptor,
            os.fsencode(source),
            parent.descriptor,
            os.fsencode(destination),
            0x00000004,
        )
    else:
        raise GuardedTreePublicationError(
            "no_replace_unavailable",
            "this platform lacks atomic no-replace directory publication",
        )
    if result != 0:
        error_code = ctypes.get_errno()
        raise OSError(error_code, os.strerror(error_code), destination)


def _directory_is_empty(parent: _Parent, name: str, *, expected: _Identity) -> bool:
    path = parent.path / name
    if parent.descriptor is None:
        with _windows_directory_namespace_fence(path):
            current = path.stat(follow_symlinks=False)
            if _capture_stable_identity(current, path=path) != expected:
                raise GuardedTreePublicationError(
                    "destination_changed",
                    "destination changed while emptiness was checked",
                )
            with os.scandir(path) as entries:
                return next(entries, None) is None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    try:
        opened = os.fstat(descriptor)
        if _capture_stable_identity(opened, descriptor=descriptor) != expected:
            raise GuardedTreePublicationError(
                "destination_changed",
                "destination changed while emptiness was checked",
            )
        with os.scandir(descriptor) as entries:
            return next(entries, None) is None
    finally:
        _close_descriptor(descriptor)


def _require_identity(
    parent: _Parent,
    name: str,
    expected: _Identity | None,
    *,
    label: str,
) -> None:
    current = parent.entry_stat(name)
    if (
        expected is None
        or current is None
        or parent.entry_identity(name, value=current) != expected
    ):
        raise GuardedTreePublicationError(
            f"{label.replace(' ', '_')}_changed",
            f"{label} identity does not match guarded publication authority",
            paths=(name,),
        )


def _require_directory_identity(parent: _Parent, name: str, *, label: str) -> _Identity:
    current = parent.entry_stat(name)
    if current is None:
        raise GuardedTreePublicationError(
            f"{label}_missing",
            f"{label} directory is missing",
            paths=(name,),
        )
    _require_plain_directory(current, label=label, path=name)
    return parent.entry_identity(name, value=current)


def _require_plain_directory(value: os.stat_result, *, label: str, path: str) -> None:
    if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
        raise GuardedTreePublicationError(
            "unsafe_entry",
            f"{label} must not be a symbolic link, junction, or reparse point",
            paths=(path,),
        )
    if not stat.S_ISDIR(value.st_mode):
        raise GuardedTreePublicationError(
            "unsupported_entry",
            f"{label} must be an ordinary directory",
            paths=(path,),
        )


def _require_supported_tree_entry(value: os.stat_result, *, path: str) -> None:
    if stat.S_ISLNK(value.st_mode) or _is_windows_reparse_point(value):
        raise GuardedTreePublicationError(
            "unsafe_tree_entry",
            "publication tree contains a link or reparse point",
            paths=(path,),
        )
    if not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)):
        raise GuardedTreePublicationError(
            "unsupported_tree_entry",
            "publication tree contains an unsupported entry type",
            paths=(path,),
        )


def _reject_case_alias(parent: _Parent, destination_name: str) -> None:
    expected = _normalized_name(destination_name)
    destination_present = False
    aliases: list[str] = []
    try:
        with os.scandir(parent.path if parent.descriptor is None else parent.descriptor) as entries:
            for entry in _bounded_parent_directory_entries(
                entries,
                overflow=lambda: GuardedTreePublicationError(
                    "parent_inspection_failed",
                    "publication parent alias discovery exceeds its bounded entry limit",
                ),
            ):
                name = entry.name
                if name == destination_name:
                    destination_present = True
                    continue
                if _normalized_name(name) != expected:
                    continue
                aliases.append(name)
                aliases.sort()
                if len(aliases) > 8:
                    aliases.pop()
    except OSError as exc:
        raise GuardedTreePublicationError(
            "parent_inspection_failed",
            "could not inspect publication parent aliases",
        ) from exc
    if not aliases:
        return
    if os.name != "nt" and (destination_present or parent.entry_stat(destination_name) is None):
        # On a case-sensitive POSIX filesystem these are distinct native names.
        # If the requested spelling is absent from the directory listing but a
        # lookup resolves, the filesystem itself aliases one of the spellings.
        return
    raise GuardedTreePublicationError(
        "destination_case_alias",
        "publication destination has a conflicting case or Unicode alias",
        paths=tuple(aliases),
    )


def _reject_normalized_aliases(names: list[str]) -> None:
    seen: dict[str, str] = {}
    for name in names:
        normalized = _normalized_name(name)
        previous = seen.get(normalized)
        if previous is not None and previous != name:
            raise GuardedTreePublicationError(
                "tree_case_alias",
                "publication tree contains conflicting case or Unicode aliases",
                paths=(previous, name),
            )
        seen[normalized] = name


def _capture_parent(path: Path) -> _Identity:
    _reject_link_components(path)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "parent_unavailable",
            "publication parent must be an existing ordinary directory",
        ) from exc
    _require_plain_directory(current, label="publication parent", path=path.name)
    return _capture_stable_identity(current, path=path)


@contextmanager
def _pinned_parent(path: Path, *, expected: _Identity) -> Iterator[_Parent]:
    _reject_link_components(path)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "parent_unavailable",
            "publication parent must be an existing ordinary directory",
        ) from exc
    _require_plain_directory(current, label="publication parent", path=path.name)
    if _capture_stable_identity(current, path=path) != expected:
        raise GuardedTreePublicationError(
            "parent_changed",
            "publication parent changed before the guarded operation acquired ownership",
        )
    if os.name == "nt":
        with _windows_directory_namespace_fence(path):
            parent = _Parent(path=path, identity=expected, descriptor=None)
            parent.assert_unchanged()
            yield parent
        return
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _capture_stable_identity(opened, descriptor=descriptor) != expected:
            raise GuardedTreePublicationError(
                "parent_changed",
                "publication parent changed while it was pinned",
            )
        parent = _Parent(path=path, identity=expected, descriptor=descriptor)
        yield parent
    finally:
        _close_descriptor(descriptor)


def _reject_link_components(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            identity = component.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GuardedTreePublicationError(
                "path_inspection_failed",
                "could not inspect publication path components",
            ) from exc
        if stat.S_ISLNK(identity.st_mode) or _is_windows_reparse_point(identity):
            raise GuardedTreePublicationError(
                "unsafe_path",
                "publication path must not traverse a link or reparse point",
                paths=(component.name,),
            )


def _assert_windows_directory_dacl_is_protected(path: Path) -> None:
    dacl_present, dacl_protected = _windows_directory_dacl_state(path)
    if not dacl_present or not dacl_protected:
        raise GuardedTreePublicationError(
            "unsafe_windows_permissions",
            "private staging directory lacks its protected DACL",
            paths=(path.name,),
        )


def _assert_windows_directory_dacl_is_inherited(path: Path) -> None:
    dacl_present, dacl_protected = _windows_directory_dacl_state(path)
    if not dacl_present or dacl_protected:
        raise GuardedTreePublicationError(
            "unsafe_windows_permissions",
            "published directory did not inherit parent permissions",
            paths=(path.name,),
        )


def _windows_directory_dacl_state(path: Path) -> tuple[bool, bool]:
    import ctypes
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    get_named_security_info = advapi32.GetNamedSecurityInfoW
    get_named_security_info.argtypes = (
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    )
    get_named_security_info.restype = wintypes.DWORD
    get_security_descriptor_control = advapi32.GetSecurityDescriptorControl
    get_security_descriptor_control.argtypes = (
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    get_security_descriptor_control.restype = wintypes.BOOL
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    local_free = kernel32.LocalFree
    local_free.argtypes = (wintypes.LPVOID,)
    local_free.restype = wintypes.LPVOID

    dacl = wintypes.LPVOID()
    security_descriptor = wintypes.LPVOID()
    error_code = get_named_security_info(
        ctypes.create_unicode_buffer(str(path)),
        1,
        0x4,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(security_descriptor),
    )
    if error_code:
        raise OSError(
            error_code,
            f"could not inspect staging-directory DACL for {path.name[:256]}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    inspection_error: BaseException | None = None
    try:
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not get_security_descriptor_control(
            security_descriptor,
            ctypes.byref(control),
            ctypes.byref(revision),
        ):
            error_code = windows_ctypes.get_last_error()
            raise OSError(
                error_code,
                "could not inspect staging-directory DACL control for "
                f"{path.name[:256]}: "
                f"{windows_ctypes.FormatError(error_code)}",
            )
        return bool(dacl), bool(control.value & 0x1000)
    except BaseException as exc:
        inspection_error = exc
        raise
    finally:
        if local_free(security_descriptor):
            free_error = OSError(
                f"could not release staging-directory DACL metadata for {path.name[:256]}"
            )
            if inspection_error is not None:
                _raise_primary_with_secondary_failure(
                    inspection_error,
                    free_error,
                    group_message=(
                        "Guarded publication DACL inspection and metadata cleanup failures."
                    ),
                )
            else:
                raise free_error


def _restore_windows_directory_inheritance(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    class _Acl(ctypes.Structure):
        _fields_ = (
            ("AclRevision", wintypes.BYTE),
            ("Sbz1", wintypes.BYTE),
            ("AclSize", wintypes.WORD),
            ("AceCount", wintypes.WORD),
            ("Sbz2", wintypes.WORD),
        )

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    initialize_acl = advapi32.InitializeAcl
    initialize_acl.argtypes = (
        ctypes.POINTER(_Acl),
        wintypes.DWORD,
        wintypes.DWORD,
    )
    initialize_acl.restype = wintypes.BOOL
    set_named_security_info = advapi32.SetNamedSecurityInfoW
    set_named_security_info.argtypes = (
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    )
    set_named_security_info.restype = wintypes.DWORD

    empty_dacl = _Acl()
    if not initialize_acl(ctypes.byref(empty_dacl), ctypes.sizeof(empty_dacl), 2):
        error_code = windows_ctypes.get_last_error()
        raise OSError(
            error_code,
            "could not initialize the published directory DACL for "
            f"{path.name[:256]}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    error_code = set_named_security_info(
        ctypes.create_unicode_buffer(str(path)),
        1,
        0x4 | 0x20000000,
        None,
        None,
        ctypes.byref(empty_dacl),
        None,
    )
    if error_code:
        raise OSError(
            error_code,
            f"could not restore inherited permissions on {path.name[:256]}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    _assert_windows_directory_dacl_is_inherited(path)


def _finalize_published_tree(
    parent: _Parent,
    name: str,
    *,
    expected: _Identity,
) -> None:
    parent.assert_unchanged()
    _require_identity(parent, name, expected, label="published destination")
    if os.name != "nt":
        return
    path = parent.path / name
    with _windows_directory_namespace_fence(path):
        _require_identity(parent, name, expected, label="published destination")
        _restore_windows_directory_inheritance(path)
        _require_identity(parent, name, expected, label="published destination")


def _create_private_windows_directory(path: Path) -> OSError | None:
    from ctypes import wintypes

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = (
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        )

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    )
    convert.restype = wintypes.BOOL
    create_directory = kernel32.CreateDirectoryW
    create_directory.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(_SecurityAttributes),
    )
    create_directory.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = (wintypes.LPVOID,)
    local_free.restype = wintypes.LPVOID
    descriptor = wintypes.LPVOID()
    if not convert(_WINDOWS_PRIVATE_DIRECTORY_SDDL, 1, ctypes.byref(descriptor), None):
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
    creation_error: BaseException | None = None
    release_error: OSError | None = None
    try:
        if not create_directory(str(path), ctypes.byref(attributes)):
            raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    except BaseException as exc:
        creation_error = exc
        raise
    finally:
        if local_free(descriptor):
            release_error = OSError("could not release the staging-directory security descriptor")
            if creation_error is not None:
                _raise_primary_with_secondary_failure(
                    creation_error,
                    release_error,
                    group_message=(
                        "Guarded publication directory creation and security metadata cleanup "
                        "failures."
                    ),
                )
    return release_error


@contextmanager
def _windows_directory_namespace_fence(path: Path) -> Iterator[None]:
    if os.name != "nt":
        yield
        return
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        0x80,
        0x1 | 0x2,
        None,
        0x3,
        0x00200000 | 0x02000000,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    fence_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        fence_error = exc
        raise
    finally:
        if not close_handle(handle):
            close_error = windows_ctypes.WinError(windows_ctypes.get_last_error())
            if fence_error is not None:
                _raise_primary_with_secondary_failure(
                    fence_error,
                    close_error,
                    group_message="Guarded publication operation and namespace-fence failures.",
                )
            else:
                raise close_error


def _sync_windows_path(path: Path, *, directory: bool) -> None:
    if os.name != "nt":
        return
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush = kernel32.FlushFileBuffers
    flush.argtypes = (wintypes.HANDLE,)
    flush.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    flags = 0x02000000 if directory else 0
    handle = create_file(str(path), 0x40000000, 0x1 | 0x2, None, 0x3, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    sync_error: BaseException | None = None
    try:
        if not flush(handle):
            raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    except BaseException as exc:
        sync_error = exc
        raise
    finally:
        if not close(handle):
            close_error = windows_ctypes.WinError(windows_ctypes.get_last_error())
            if sync_error is not None:
                _raise_primary_with_secondary_failure(
                    sync_error,
                    close_error,
                    group_message="Guarded publication sync and handle cleanup failures.",
                )
            else:
                raise close_error


def _delete_windows_entry_by_handle(
    path: Path,
    *,
    expected: _Identity,
    authority: dict[str, _CleanupEntry] | None = None,
    prefix: PurePosixPath | None = None,
) -> None:
    if prefix is None:
        prefix = PurePosixPath()
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "owned cleanup entry changed before it was pinned",
            paths=(path.name,),
        ) from exc
    if _capture_stable_identity(current, path=path) != expected:
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "owned cleanup entry identity changed",
            paths=(path.name,),
        )
    _require_supported_tree_entry(current, path=path.name)

    with _windows_deletion_handle(
        path,
        read_content=stat.S_ISREG(current.st_mode),
    ) as (deletion_handle, mark_for_deletion):
        captured_entry: _CapturedCleanupEntry | None = None
        try:
            opened = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup entry changed after it was pinned",
                paths=(path.name,),
            ) from exc
        if _capture_stable_identity(opened, path=path) != expected:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup entry changed after it was pinned",
                paths=(path.name,),
            )
        _require_supported_tree_entry(opened, path=path.name)
        if stat.S_ISDIR(opened.st_mode):
            try:
                with os.scandir(path) as entries:
                    children = _bounded_cleanup_entry_names(
                        entries,
                        authority=authority,
                        prefix=prefix,
                    )
            except OSError as exc:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "could not inspect an owned cleanup directory",
                    paths=(path.name,),
                ) from exc
            for child_name in children:
                child = path / child_name
                relative = prefix / child_name
                try:
                    child_value = child.stat(follow_symlinks=False)
                    child_identity = _capture_stable_identity(child_value, path=child)
                except OSError as exc:
                    raise GuardedTreePublicationError(
                        "cleanup_conflict",
                        "owned cleanup entry changed during traversal",
                        paths=(child_name,),
                    ) from exc
                expected_entry = None if authority is None else authority.get(relative.as_posix())
                if authority is not None and (
                    expected_entry is None or expected_entry.identity != child_identity
                ):
                    raise GuardedTreePublicationError(
                        "cleanup_conflict",
                        "cleanup encountered content absent from its durable authority",
                        paths=(relative.as_posix(),),
                    )
                _delete_windows_entry_by_handle(
                    child,
                    expected=(
                        child_identity if expected_entry is None else expected_entry.identity
                    ),
                    authority=authority,
                    prefix=relative,
                )
        elif authority is not None:
            expected_entry = authority.get(prefix.as_posix())
            if expected_entry is None or expected_entry.identity != expected:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "cleanup encountered content absent from its durable authority",
                    paths=(prefix.as_posix(),),
                )
            captured_entry = _capture_windows_cleanup_entry(
                path,
                relative=prefix,
                value=opened,
                handle=deletion_handle,
            )
            if captured_entry.entry != expected_entry:
                raise GuardedTreePublicationError(
                    "cleanup_conflict",
                    "owned cleanup entry changed before deletion",
                    paths=(prefix.as_posix(),),
                )
        try:
            final = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup entry changed before deletion",
                paths=(path.name,),
            ) from exc
        if _capture_stable_identity(final, path=path) != expected or (
            captured_entry is not None
            and (
                captured_entry.file_observation is None
                or _capture_windows_file_mutation_observation(
                    final,
                    handle=deletion_handle,
                )
                != captured_entry.file_observation
            )
        ):
            raise GuardedTreePublicationError(
                "cleanup_conflict",
                "owned cleanup entry changed before deletion",
                paths=(path.name,),
            )
        _require_supported_tree_entry(final, path=path.name)
        mark_for_deletion()


def _capture_windows_cleanup_entry(
    path: Path,
    *,
    relative: PurePosixPath,
    value: os.stat_result,
    handle: int,
) -> _CapturedCleanupEntry:
    identity = _capture_stable_identity(value, path=path)
    opened_observation = _capture_windows_file_mutation_observation(value, handle=handle)
    content_digest = hashlib.sha256()
    content_size = 0
    for content in _read_windows_file_handle(handle):
        content_size += len(content)
        content_digest.update(content)
    after = path.stat(follow_symlinks=False)
    if (
        _capture_stable_identity(after, path=path) != identity
        or _capture_windows_file_mutation_observation(after, handle=handle) != opened_observation
        or content_size != opened_observation.size
    ):
        raise GuardedTreePublicationError(
            "cleanup_conflict",
            "owned cleanup file changed while it was inspected",
            paths=(relative.as_posix(),),
        )
    return _CapturedCleanupEntry(
        entry=_CleanupEntry(
            path=relative.as_posix(),
            identity=identity,
            mode=opened_observation.mode,
            size=opened_observation.size,
            content_sha256=f"sha256:{content_digest.hexdigest()}",
        ),
        file_observation=opened_observation,
    )


def _read_windows_file_handle(handle: int) -> Iterator[bytes]:
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    read_file = kernel32.ReadFile
    read_file.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    read_file.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(1024 * 1024)
    while True:
        consumed = wintypes.DWORD()
        if not read_file(
            handle,
            buffer,
            len(buffer),
            ctypes.byref(consumed),
            None,
        ):
            error_code = windows_ctypes.get_last_error()
            raise OSError(error_code, "could not read an owned cleanup file")
        if consumed.value == 0:
            return
        yield bytes(buffer.raw[: consumed.value])


@contextmanager
def _windows_deletion_handle(
    path: Path,
    *,
    read_content: bool = False,
) -> Iterator[tuple[int, Callable[[], None]]]:
    if os.name != "nt":
        raise GuardedTreePublicationError(
            "unsupported_platform_operation",
            "Windows cleanup handles require Windows",
        )
    from ctypes import wintypes

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", wintypes.BOOLEAN),)

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_file_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    desired_access = 0x00010000 | 0x80
    if read_content:
        desired_access |= 0x80000000
    handle = create_file(
        str(path),
        desired_access,
        0x1,
        None,
        0x3,
        0x00200000 | 0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, "could not pin an owned entry for cleanup")

    def mark_for_deletion() -> None:
        disposition = _FileDispositionInfo(delete_file=True)
        if not set_file_information(
            handle,
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error_code = windows_ctypes.get_last_error()
            raise OSError(error_code, "could not delete an owned cleanup entry")

    cleanup_error: BaseException | None = None
    try:
        yield handle, mark_for_deletion
    except BaseException as exc:
        cleanup_error = exc
        raise
    finally:
        if not close_handle(handle):
            close_error = OSError(
                windows_ctypes.get_last_error(),
                "could not release an owned cleanup handle",
            )
            if cleanup_error is not None:
                _raise_primary_with_secondary_failure(
                    cleanup_error,
                    close_error,
                    group_message="Guarded publication cleanup and handle-release failures.",
                )
            else:
                raise close_error


def _is_windows_reparse_point(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT)


def _matches(
    parent: _Parent,
    name: str,
    expected: _Identity | None,
    value: os.stat_result | None,
) -> bool:
    if expected is None or value is None or not expected.matches(value):
        return False
    if (
        stat.S_ISLNK(value.st_mode)
        or _is_windows_reparse_point(value)
        or not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode))
    ):
        return False
    return parent.entry_identity(name, value=value) == expected


def _normalized_name(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _destination_name_for_lookup_semantics(
    value: str,
    semantics: _DirectoryLookupSemantics,
) -> str:
    if semantics is _DirectoryLookupSemantics.UNICODE_NORMALIZED:
        return unicodedata.normalize("NFC", value)
    if semantics is not _DirectoryLookupSemantics.CASE_SENSITIVE:
        # Unknown filesystems retain the conservative collision domain rather
        # than risking two durable owners for one native destination.
        return _normalized_name(value)
    return value


def _directory_lookup_semantics(parent: _Parent) -> _DirectoryLookupSemantics:
    if os.name == "nt":
        return _DirectoryLookupSemantics.UNICODE_CASEFOLDED
    if sys.platform == "darwin":
        try:
            root: int | Path = parent.descriptor if parent.descriptor is not None else parent.path
            case_sensitive = os.pathconf(root, _DARWIN_PC_CASE_SENSITIVE)
        except OSError as exc:
            if exc.errno in {
                errno.EINVAL,
                errno.ENOSYS,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
            }:
                return _DirectoryLookupSemantics.UNKNOWN
            raise GuardedTreePublicationError(
                "parent_inspection_failed",
                "could not determine publication parent lookup semantics",
            ) from exc
        if case_sensitive == 1:
            # APFS and HFS+ remain canonically normalization-insensitive even
            # when case-sensitive. Preserve case while binding canonically
            # equivalent spellings to one durable publication owner.
            return _DirectoryLookupSemantics.UNICODE_NORMALIZED
        if case_sensitive == 0:
            return _DirectoryLookupSemantics.UNICODE_CASEFOLDED
        return _DirectoryLookupSemantics.UNKNOWN
    if sys.platform.startswith("linux") and parent.descriptor is not None:
        try:
            flags = _linux_file_flags(parent.descriptor)
        except OSError as exc:
            raise GuardedTreePublicationError(
                "parent_inspection_failed",
                "could not determine publication parent lookup semantics",
            ) from exc
        if flags is None:
            return _DirectoryLookupSemantics.UNKNOWN
        if flags & _LINUX_FS_CASEFOLD_FL:
            return _DirectoryLookupSemantics.UNICODE_CASEFOLDED
        return _DirectoryLookupSemantics.CASE_SENSITIVE
    return _DirectoryLookupSemantics.UNKNOWN


def _destination_metadata_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _publication_metadata_keys(value: str) -> tuple[str, str, str]:
    return (
        _destination_metadata_key(_normalized_name(value)),
        _destination_metadata_key(unicodedata.normalize("NFC", value)),
        _destination_metadata_key(value),
    )


def _publication_metadata_stem(keys: tuple[str, str, str]) -> str:
    return "-".join(keys)


def _publication_metadata_scope_prefix(
    keys: tuple[str, str, str],
    semantics: _DirectoryLookupSemantics,
) -> str:
    collision_key, canonical_key, _raw_key = keys
    prefix = f".cayu-tree-publication-{collision_key}-"
    if semantics is _DirectoryLookupSemantics.UNICODE_NORMALIZED:
        return f"{prefix}{canonical_key}-"
    if semantics is _DirectoryLookupSemantics.CASE_SENSITIVE:
        return f".cayu-tree-publication-{_publication_metadata_stem(keys)}"
    return prefix


def _publication_metadata_candidates(
    parent: _Parent,
    *,
    keys: tuple[str, str, str],
    semantics: _DirectoryLookupSemantics,
) -> dict[str, tuple[str, ...]]:
    names_by_stem: dict[str, list[str]] = {}
    candidate_count = 0
    prefix = _publication_metadata_scope_prefix(keys, semantics)
    try:
        with os.scandir(parent.path if parent.descriptor is None else parent.descriptor) as entries:
            for entry in _bounded_parent_directory_entries(
                entries,
                overflow=lambda: _invalid_journal(
                    "publication metadata owner discovery exceeds its bounded parent-entry limit"
                ),
            ):
                if not entry.name.startswith(prefix):
                    continue
                candidate_count += 1
                if candidate_count > _PUBLICATION_METADATA_CENSUS_LIMIT:
                    raise _invalid_journal(
                        "publication metadata owner discovery exceeds its bounded limit"
                    )
                match = _PUBLICATION_METADATA_NAME_PATTERN.fullmatch(entry.name)
                if match is None:
                    raise _invalid_journal(
                        "publication metadata owner discovery found a malformed candidate"
                    )
                candidate_keys = (
                    match.group("collision_key"),
                    match.group("canonical_key"),
                    match.group("raw_key"),
                )
                if not _metadata_keys_match_lookup_scope(
                    candidate_keys,
                    keys,
                    semantics=semantics,
                ):
                    raise _invalid_journal("publication metadata owner escaped its lookup scope")
                names_by_stem.setdefault(
                    _publication_metadata_stem(candidate_keys),
                    [],
                ).append(entry.name)
    except OSError as exc:
        raise _invalid_journal("publication metadata owners could not be inspected") from exc
    return {stem: tuple(sorted(metadata_names)) for stem, metadata_names in names_by_stem.items()}


def _bounded_parent_directory_entries(
    entries: Iterator[Any],
    *,
    overflow: Callable[[], GuardedTreePublicationError],
) -> Iterator[Any]:
    for index, entry in enumerate(entries):
        if index >= _PARENT_DIRECTORY_CENSUS_LIMIT:
            raise overflow()
        yield entry


def _metadata_keys_match_lookup_scope(
    candidate: tuple[str, str, str],
    expected: tuple[str, str, str],
    *,
    semantics: _DirectoryLookupSemantics,
) -> bool:
    if semantics is _DirectoryLookupSemantics.CASE_SENSITIVE:
        return candidate == expected
    if semantics is _DirectoryLookupSemantics.UNICODE_NORMALIZED:
        return candidate[:2] == expected[:2]
    return candidate[0] == expected[0]


def _metadata_candidate_destination_names(
    parent: _Parent,
    *,
    stem: str,
    metadata_names: tuple[str, ...],
) -> set[str]:
    durable_names: set[str] = set()
    for metadata_name in metadata_names:
        if parent.entry_stat(metadata_name) is None:
            continue
        if ".jsonl.pending-" in metadata_name:
            raise _invalid_journal(
                "pending publication journal lacks independently durable Cayu ownership evidence"
            )
        destination_name = _load_journal(
            parent.path / metadata_name,
            parent=parent,
        ).record.destination_name
        if _publication_metadata_stem(_publication_metadata_keys(destination_name)) != stem:
            raise _invalid_journal(
                "publication metadata filename does not match its destination authority"
            )
        durable_names.add(destination_name)
    if len(durable_names) > 1:
        raise _invalid_journal(
            "one publication metadata key contains conflicting destination authority"
        )
    return durable_names


def _resolve_destination_metadata_stem(parent: _Parent, value: str) -> str:
    semantics = _directory_lookup_semantics(parent)
    selected_name = _destination_name_for_lookup_semantics(value, semantics)
    selected_keys = _publication_metadata_keys(value)
    selected_stem = _publication_metadata_stem(selected_keys)
    matching_existing_stems: list[str] = []
    for stem, metadata_names in _publication_metadata_candidates(
        parent,
        keys=selected_keys,
        semantics=semantics,
    ).items():
        durable_names = _metadata_candidate_destination_names(
            parent,
            stem=stem,
            metadata_names=metadata_names,
        )
        if any(
            _destination_name_for_lookup_semantics(name, semantics) != selected_name
            for name in durable_names
        ):
            raise _invalid_journal(
                "publication metadata lookup key collides with another destination"
            )
        if durable_names:
            matching_existing_stems.append(stem)
    if len(matching_existing_stems) > 1:
        raise _invalid_journal(
            "one publication destination has durable metadata under multiple owner keys"
        )
    if matching_existing_stems:
        return matching_existing_stems[0]
    return selected_stem


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("could not write publication journal")
        offset += written


def _read_bounded(descriptor: int, *, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _conflict(record: _Record, message: str) -> GuardedTreePublicationError:
    return GuardedTreePublicationError(
        "publication_conflict",
        message,
        paths=(
            record.destination_name,
            record.stage_name,
            record.backup_name,
            _cleanup_name(record),
        ),
    )


def _invalid_journal(message: str) -> GuardedTreePublicationError:
    return GuardedTreePublicationError("invalid_publication_journal", message)


def _publication_fault(phase: str) -> None:
    """Private phase seam used by real subprocess fault-injection tests."""

    del phase

"""Isolated POSIX supervisor for contained stdio MCP servers.

This file is executed directly with the current Python interpreter.  Keep it
stdlib-only: isolated startup deliberately does not import the Cayu package.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hmac
import json
import os
import platform
import select
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import NamedTuple, NoReturn

_POLL_INTERVAL_S = 0.005
_MAX_CONTROL_BYTES = 65_536
_MAX_SERVER_ENV_BYTES = 16 * 1024 * 1024
_RENDEZVOUS_ABSTRACT_PREFIX = b"\0cayu-mcp-containment-v1-"
_EXIT_OWNER_GONE = 71
_EXIT_START_FAILED = 72
_EXIT_CLEANUP_FAILED = 73
_ANCHOR_CLEANUP_REQUESTED = False
_ANCHOR_FORCE_REQUESTED = False
_LINUX_CAPABILITY_VERSION_3 = 0x20080522
_LINUX_SECUREBITS_LOCKED_NO_PRIVILEGE = 0x0F
_LINUX_F_ADD_SEALS = 1033
_LINUX_F_GET_SEALS = 1034
_LINUX_MFD_CLOEXEC = 0x0001
_LINUX_MFD_ALLOW_SEALING = 0x0002
_LINUX_REQUIRED_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008

_AUDIT_ARCH_BY_MACHINE = {
    "aarch64": 0xC00000B7,
    "arm64": 0xC00000B7,
    "amd64": 0xC000003E,
    "x86_64": 0xC000003E,
}
_PROCESS_TREE_ESCAPE_SYSCALLS = {
    "aarch64": (117, 154, 157, 424, 438),
    "arm64": (117, 154, 157, 424, 438),
    "amd64": (101, 109, 112, 424, 438),
    "x86_64": (101, 109, 112, 424, 438),
}
_CAPSET_SYSCALLS = {
    "aarch64": 91,
    "arm64": 91,
    "amd64": 126,
    "x86_64": 126,
}
_PARENT_SIGNAL_SYSCALLS = {
    "aarch64": {
        "kill": 129,
        "pidfd_open": 434,
        "process_vm_writev": 271,
        "prlimit64": 261,
        "rt_sigqueueinfo": 138,
        "rt_tgsigqueueinfo": 240,
        "tgkill": 131,
        "tkill": 130,
    },
    "arm64": {
        "kill": 129,
        "pidfd_open": 434,
        "process_vm_writev": 271,
        "prlimit64": 261,
        "rt_sigqueueinfo": 138,
        "rt_tgsigqueueinfo": 240,
        "tgkill": 131,
        "tkill": 130,
    },
    "amd64": {
        "kill": 62,
        "pidfd_open": 434,
        "process_vm_writev": 311,
        "prlimit64": 302,
        "rt_sigqueueinfo": 129,
        "rt_tgsigqueueinfo": 297,
        "tgkill": 234,
        "tkill": 200,
    },
    "x86_64": {
        "kill": 62,
        "pidfd_open": 434,
        "process_vm_writev": 311,
        "prlimit64": 302,
        "rt_sigqueueinfo": 129,
        "rt_tgsigqueueinfo": 297,
        "tgkill": 234,
        "tkill": 200,
    },
}


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


class _LinuxCapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _LinuxCapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


class _LinuxProcessStat(NamedTuple):
    state: str
    parent_pid: int
    process_group: int
    start_tick: int
    proc_inode: int


def _append_argument_denial(
    instructions: list[_SockFilter],
    *,
    syscall_number: int,
    checks: tuple[tuple[int, tuple[int, ...]], ...],
) -> None:
    block: list[_SockFilter] = []
    for offset, denied_values in checks:
        block.append(_SockFilter(0x20, 0, 0, offset))
        for value in denied_values:
            block.extend(
                (
                    _SockFilter(0x15, 0, 1, value & 0xFFFFFFFF),
                    _SockFilter(0x06, 0, 0, 0x00050000 | errno.EPERM),
                )
            )
    block.append(_SockFilter(0x06, 0, 0, 0x7FFF0000))
    instructions.append(_SockFilter(0x15, 0, len(block), syscall_number))
    instructions.extend(block)


def _install_linux_process_tree_filter(
    *,
    protected_process_pids: tuple[int, ...],
    protected_process_groups: tuple[int, ...],
) -> None:
    """Keep the contained tree grouped and its cleanup parent alive."""

    machine = platform.machine().lower()
    audit_arch = _AUDIT_ARCH_BY_MACHINE.get(machine)
    escape_syscalls = _PROCESS_TREE_ESCAPE_SYSCALLS.get(machine)
    parent_signal_syscalls = _PARENT_SIGNAL_SYSCALLS.get(machine)
    capset_syscall = _CAPSET_SYSCALLS.get(machine)
    if (
        sys.platform != "linux"
        or audit_arch is None
        or escape_syscalls is None
        or parent_signal_syscalls is None
        or capset_syscall is None
    ):
        raise RuntimeError("unsupported process-tree containment platform")

    # seccomp_data offsets: nr=0, arch=4. Denying setsid and setpgid keeps every
    # descendant in the isolated group without restricting ordinary process or
    # thread creation. The inherited filter and no_new_privs bit survive every
    # fork/exec boundary.
    instructions = [
        _SockFilter(0x20, 0, 0, 4),
        _SockFilter(0x15, 1, 0, audit_arch),
        _SockFilter(0x06, 0, 0, 0x80000000),
        _SockFilter(0x20, 0, 0, 0),
    ]
    if audit_arch == 0xC000003E:
        # Normalize the x32 ABI syscall bit before matching so a compatible
        # executable cannot bypass the process-group restrictions.
        instructions.append(_SockFilter(0x54, 0, 0, 0xBFFFFFFF))
    for syscall_number in escape_syscalls:
        instructions.extend(
            (
                _SockFilter(0x15, 0, 1, syscall_number),
                _SockFilter(
                    0x06,
                    0,
                    0,
                    0x00050000 | errno.EPERM,
                ),
            )
        )
    protected_pids = tuple(dict.fromkeys(pid & 0xFFFFFFFF for pid in protected_process_pids))
    protected_groups = tuple(
        dict.fromkeys((-pgid) & 0xFFFFFFFF for pgid in protected_process_groups)
    )
    _append_argument_denial(
        instructions,
        syscall_number=parent_signal_syscalls["kill"],
        checks=((16, (*protected_pids, *protected_groups, 0, 0xFFFFFFFF)),),
    )
    for name in (
        "pidfd_open",
        "process_vm_writev",
        "prlimit64",
        "rt_sigqueueinfo",
        "tkill",
    ):
        _append_argument_denial(
            instructions,
            syscall_number=parent_signal_syscalls[name],
            checks=((16, protected_pids),),
        )
    for name in ("rt_tgsigqueueinfo", "tgkill"):
        _append_argument_denial(
            instructions,
            syscall_number=parent_signal_syscalls[name],
            checks=((16, protected_pids), (24, protected_pids)),
        )
    instructions.append(_SockFilter(0x06, 0, 0, 0x7FFF0000))
    instruction_array = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(len(instructions), instruction_array)
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "failed to set no_new_privs")
    # A privileged Cayu owner can otherwise pass CAP_SYS_PTRACE (or broader
    # capabilities) into the server, allowing it to bypass non-dumpable
    # cleanup owners. Lock out root/setuid privilege semantics, clear ambient
    # authority, then irrevocably empty every inheritable/permitted/effective
    # capability word before exec. no_new_privs prevents file capabilities from
    # restoring anything after this point.
    if (
        os.geteuid() == 0
        and prctl(  # PR_SET_SECUREBITS
            28,
            _LINUX_SECUREBITS_LOCKED_NO_PRIVILEGE,
            0,
            0,
            0,
        )
        != 0
    ):
        raise OSError(ctypes.get_errno(), "failed to lock process privilege semantics")
    if prctl(47, 4, 0, 0, 0) != 0:  # PR_CAP_AMBIENT / PR_CAP_AMBIENT_CLEAR_ALL
        raise OSError(ctypes.get_errno(), "failed to clear ambient process capabilities")
    header = _LinuxCapabilityHeader(_LINUX_CAPABILITY_VERSION_3, 0)
    capability_words = (_LinuxCapabilityData * 2)()
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    if syscall(capset_syscall, ctypes.byref(header), ctypes.byref(capability_words)) != 0:
        raise OSError(ctypes.get_errno(), "failed to drop process capabilities")
    status = Path("/proc/self/status").read_text(encoding="ascii")
    remaining_capabilities = {
        name: int(value.strip(), 16)
        for line in status.splitlines()
        for name, separator, value in (line.partition(":"),)
        if separator and name in {"CapEff", "CapPrm", "CapInh", "CapAmb"}
    }
    if any(
        remaining_capabilities.get(name, -1) != 0
        for name in ("CapEff", "CapPrm", "CapInh", "CapAmb")
    ):
        raise RuntimeError("process capabilities remained after containment preparation")
    program_address = ctypes.cast(ctypes.pointer(program), ctypes.c_void_p).value
    if program_address is None or prctl(22, 2, program_address, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "failed to install process-tree filter")


def _set_linux_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "failed to establish child subreaper")


def _set_linux_process_nondumpable() -> None:
    """Prevent same-UID descendants from opening cleanup-owner memory or FDs."""

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), "failed to disable cleanup-owner dumpability")
    if prctl(3, 0, 0, 0, 0) != 0:  # PR_GET_DUMPABLE
        raise RuntimeError("cleanup owner remained dumpable")


def _establish_linux_child_reaping_semantics() -> None:
    """Own the zombie/reaping behavior required by exact PGID settlement."""

    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("child reaping semantics could not be established")


def _read_server_environment(fd: int) -> dict[str, str]:
    """Read one sealed, anonymous server environment from the Cayu owner."""

    import fcntl

    try:
        if (
            fcntl.fcntl(fd, _LINUX_F_GET_SEALS) & _LINUX_REQUIRED_MEMFD_SEALS
            != _LINUX_REQUIRED_MEMFD_SEALS
        ):
            raise RuntimeError("server environment transfer was not sealed")
        size = os.fstat(fd).st_size
        if not 0 <= size <= _MAX_SERVER_ENV_BYTES:
            raise RuntimeError("server environment exceeded its containment transfer limit")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = os.read(fd, min(65_536, size - len(chunks)))
            if not chunk:
                raise RuntimeError("server environment transfer ended early")
            chunks.extend(chunk)
        if os.read(fd, 1):
            raise RuntimeError("server environment transfer exceeded its declared size")
        value = json.loads(chunks)
    finally:
        with suppress(OSError):
            os.close(fd)
    if type(value) is not dict:
        raise RuntimeError("server environment transfer was not an object")
    environment: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) is not str or not key or "=" in key:
            environment.clear()
            raise RuntimeError("server environment transfer contained an invalid entry")
        try:
            key.encode("utf-8", "strict")
            item.encode("utf-8", "strict")
        except UnicodeEncodeError:
            environment.clear()
            raise RuntimeError("server environment transfer contained invalid text") from None
        if "\x00" in key or "\x00" in item:
            environment.clear()
            raise RuntimeError("server environment transfer contained invalid text")
        environment[key] = item
    return environment


def _verify_linux_environment_transfer() -> None:
    """Exercise the anonymous sealed transfer used after secret resolution."""

    import fcntl

    fd = -1
    try:
        memfd_create = getattr(os, "memfd_create", None)
        if not callable(memfd_create):
            raise RuntimeError("anonymous environment transfer is unavailable")
        fd = memfd_create(
            "cayu-mcp-containment-preflight",
            flags=_LINUX_MFD_CLOEXEC | _LINUX_MFD_ALLOW_SEALING,
        )
        if os.write(fd, b"{}") != 2:
            raise RuntimeError("server environment preflight write was incomplete")
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(
            fd,
            _LINUX_F_ADD_SEALS,
            _LINUX_REQUIRED_MEMFD_SEALS,
        )
        if _read_server_environment(fd):
            raise RuntimeError("server environment preflight returned unexpected data")
        fd = -1
    finally:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)


def _verify_linux_waitid_without_reaping() -> None:
    """Exercise the exact non-reaping child observation used by the supervisor."""

    fork = getattr(os, "fork", None)
    waitid = getattr(os, "waitid", None)
    if not callable(fork) or not callable(waitid):
        raise RuntimeError("non-reaping child observation is unavailable")
    child_pid = fork()
    if child_pid == 0:
        _fixed_exit(0)
    reaped = False
    pidfd = -1
    try:
        pidfd = _linux_pidfd_open(child_pid)
        _linux_pidfd_send_signal(pidfd, 0)
        result = waitid(os.P_PID, child_pid, os.WEXITED | os.WNOWAIT)
        if result is None or result.si_pid != child_pid:
            raise RuntimeError("non-reaping child observation returned invalid evidence")
        waited_pid, status = os.waitpid(child_pid, 0)
        reaped = True
        if waited_pid != child_pid or os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError("containment preflight child did not settle successfully")
    finally:
        if pidfd >= 0:
            with suppress(OSError):
                os.close(pidfd)
        if not reaped:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                os.waitpid(child_pid, 0)


def _validate_rendezvous_identity(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError("invalid containment rendezvous identity")
    return value


def _rendezvous_address(identity: str) -> bytes:
    identity = _validate_rendezvous_identity(identity)
    address = (
        _RENDEZVOUS_ABSTRACT_PREFIX
        + b"u"
        + str(os.geteuid()).encode("ascii")
        + b"-"
        + identity.encode("ascii")
    )
    if len(address) > 107:
        raise RuntimeError("containment rendezvous identity exceeded the address limit")
    return address


def _verify_linux_abstract_rendezvous() -> None:
    """Prove exact bind exclusion and release in Linux's pathless namespace."""

    address = b"\0cayu-mcp-containment-preflight-" + os.urandom(16).hex().encode("ascii")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as owner:
        owner.bind(address)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as contender:
            try:
                contender.bind(address)
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
            else:
                raise RuntimeError("containment rendezvous admitted a duplicate owner")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as replacement:
        replacement.bind(address)


def _validated_bound_rendezvous(fd: int, identity: str) -> socket.socket:
    """Own and authenticate one supervisor-produced abstract rendezvous socket."""

    if fd < 0:
        raise RuntimeError("containment rendezvous descriptor was invalid")
    owned = socket.socket(fileno=fd)
    try:
        if (
            owned.family != socket.AF_UNIX
            or (owned.type & socket.SOCK_STREAM) != socket.SOCK_STREAM
            or owned.getsockname() != _rendezvous_address(identity)
        ):
            raise RuntimeError("containment rendezvous descriptor was not authoritative")
    except BaseException:
        owned.close()
        raise
    return owned


def _verify_linux_containment_primitives() -> None:
    """Exercise every host primitive required by the trusted wrapper processes."""

    _establish_linux_child_reaping_semantics()
    _verify_linux_environment_transfer()
    _set_linux_process_nondumpable()
    _set_linux_child_subreaper()
    _verify_linux_waitid_without_reaping()
    _verify_linux_abstract_rendezvous()
    process_pid = os.getpid()
    parent_pid = os.getppid()
    _install_linux_process_tree_filter(
        protected_process_pids=(process_pid, parent_pid),
        protected_process_groups=(os.getpgrp(), os.getpgid(parent_pid)),
    )
    if process_pid not in _linux_process_group_stats(os.getpgrp()):
        raise RuntimeError("process-group evidence was unavailable after containment preparation")


def _request_anchor_cleanup(_signal_number: int, _frame: object) -> None:
    global _ANCHOR_CLEANUP_REQUESTED
    _ANCHOR_CLEANUP_REQUESTED = True


def _request_forced_anchor_cleanup(_signal_number: int, _frame: object) -> None:
    global _ANCHOR_CLEANUP_REQUESTED, _ANCHOR_FORCE_REQUESTED
    _ANCHOR_CLEANUP_REQUESTED = True
    _ANCHOR_FORCE_REQUESTED = True


def _linux_pidfd_open(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    while True:
        ctypes.set_errno(0)
        pidfd = syscall(434, pid, 0)  # pidfd_open
        if pidfd >= 0:
            return int(pidfd)
        error_number = ctypes.get_errno()
        if error_number != errno.EINTR:
            raise OSError(error_number, "pidfd_open failed")


def _linux_pidfd_send_signal(pidfd: int, signal_number: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    while True:
        ctypes.set_errno(0)
        result = syscall(424, pidfd, signal_number, 0, 0)  # pidfd_send_signal
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number != errno.EINTR:
            raise OSError(error_number, "pidfd_send_signal failed")


def _linux_pidfd_signaling_supported() -> bool:
    """Probe the exact process-instance signaling primitives used by cleanup."""

    if sys.platform != "linux":
        return False
    if _linux_process_identity(os.getpid()) is None:
        return False
    pidfd = -1
    try:
        pidfd = _linux_pidfd_open(os.getpid())
        _linux_pidfd_send_signal(pidfd, 0)
    except OSError:
        return False
    finally:
        if pidfd >= 0:
            os.close(pidfd)
    return True


def _linux_process_stat(pid: int) -> _LinuxProcessStat | None:
    """Return one process instance's lifecycle and ownership fields."""

    try:
        proc_path = Path(f"/proc/{pid}")
        proc_inode = proc_path.stat().st_ino
        stat = (proc_path / "stat").read_text(encoding="ascii")
        suffix = stat[stat.rfind(")") + 2 :].split()
        return _LinuxProcessStat(
            state=suffix[0],
            parent_pid=int(suffix[1]),
            process_group=int(suffix[2]),
            start_tick=int(suffix[19]),
            proc_inode=proc_inode,
        )
    except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
        return None


def _linux_process_identity(pid: int) -> tuple[int, int, int] | None:
    stat = _linux_process_stat(pid)
    if stat is None:
        return None
    return stat.process_group, stat.start_tick, stat.proc_inode


def _linux_process_group_pids(pgid: int) -> dict[int, tuple[int, int, int]]:
    pids: dict[int, tuple[int, int, int]] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            identity = _linux_process_identity(pid)
            if identity is not None and identity[0] == pgid:
                pids[pid] = identity
    return pids


def _linux_process_group_stats(pgid: int) -> dict[int, _LinuxProcessStat]:
    members: dict[int, _LinuxProcessStat] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            stat = _linux_process_stat(pid)
            if stat is not None and stat.process_group == pgid:
                members[pid] = stat
    return members


def _signal_owned_group_members(pgid: int, signal_number: int) -> None:
    for pid, expected_identity in _linux_process_group_pids(pgid).items():
        try:
            pidfd = _linux_pidfd_open(pid)
        except OSError as error:
            if error.errno == errno.ESRCH:
                continue
            raise
        try:
            # The census is only a candidate list. Revalidate group membership
            # after opening the exact process handle so exit/PID reuse between
            # those operations cannot redirect cleanup to unrelated work.
            identity = _linux_process_identity(pid)
            if identity != expected_identity:
                continue
            try:
                _linux_pidfd_send_signal(pidfd, signal_number)
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise
        finally:
            os.close(pidfd)


def _reap_anchor_children(
    server_pid: int, server_returncode: int | None
) -> tuple[int | None, bool]:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return server_returncode, False
        if pid == 0:
            return server_returncode, True
        if pid == server_pid and server_returncode is None:
            server_returncode = os.waitstatus_to_exitcode(status)


def _settle_anchor_tree(
    *,
    server_pid: int,
    server_returncode: int | None,
    term_timeout_s: float,
    kill_timeout_s: float,
    force: bool,
) -> tuple[int | None, bool]:
    pgid = os.getpgrp()
    if not force:
        _signal_owned_group_members(pgid, signal.SIGTERM)
        deadline = time.monotonic() + term_timeout_s
        while time.monotonic() < deadline:
            server_returncode, children_remain = _reap_anchor_children(
                server_pid,
                server_returncode,
            )
            if not children_remain and not _linux_process_group_pids(pgid):
                return server_returncode, True
            if _ANCHOR_FORCE_REQUESTED:
                break
            time.sleep(_POLL_INTERVAL_S)

    deadline = time.monotonic() + kill_timeout_s
    while True:
        _signal_owned_group_members(pgid, signal.SIGKILL)
        server_returncode, children_remain = _reap_anchor_children(
            server_pid,
            server_returncode,
        )
        group_members = _linux_process_group_pids(pgid)
        if not children_remain and not group_members:
            return server_returncode, True
        if time.monotonic() >= deadline:
            return server_returncode, False
        time.sleep(_POLL_INTERVAL_S)


def _settle_supervisor_tree(
    *,
    pgid: int,
    anchor_pid: int,
    term_timeout_s: float,
    kill_timeout_s: float,
) -> bool:
    """Freeze and settle one exact group while retaining its zombie leader."""

    freeze_deadline = time.monotonic() + term_timeout_s
    group_frozen = False
    while time.monotonic() < freeze_deadline:
        try:
            os.killpg(pgid, signal.SIGSTOP)
        except ProcessLookupError:
            return False
        members = _linux_process_group_stats(pgid)
        if anchor_pid not in members:
            return False
        if all(member.state in {"T", "t", "Z", "X"} for member in members.values()):
            group_frozen = True
            break
        time.sleep(_POLL_INTERVAL_S)
    if not group_frozen:
        return False

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return False

    deadline = time.monotonic() + kill_timeout_s
    while True:
        members = _linux_process_group_stats(pgid)
        if anchor_pid not in members:
            return False
        for pid in members:
            if pid == anchor_pid:
                continue
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
        members = _linux_process_group_stats(pgid)
        if set(members) == {anchor_pid} and members[anchor_pid].state in {"Z", "X"}:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL_S)


def _fixed_exit(code: int) -> NoReturn:
    os._exit(code)


def _exit_with_server_returncode(returncode: int) -> NoReturn:
    """Mirror the real server's normal or signal exit at the process boundary."""

    if returncode >= 0:
        _fixed_exit(returncode)
    signal_number = -returncode
    with suppress(OSError, ValueError):
        signal.signal(signal_number, signal.SIG_DFL)
    if hasattr(signal, "pthread_sigmask"):
        with suppress(OSError, ValueError):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal_number})
    os.kill(os.getpid(), signal_number)
    # A valid terminating signal should not return. Preserve a deterministic
    # failure if a platform unexpectedly declines to deliver it.
    _fixed_exit(128 + signal_number)


def _write_message(fd: int, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    os.write(fd, encoded + b"\n")


def _owner_is_gone(owner_fd: int, expected_parent_pid: int) -> bool:
    if os.getppid() != expected_parent_pid:
        return True
    if owner_fd not in _poll_readable_descriptors((owner_fd,), 0):
        return False
    try:
        return os.read(owner_fd, 1) == b""
    except OSError:
        return True


def _poll_readable_descriptors(fds: tuple[int, ...], timeout_s: float) -> set[int]:
    """Poll private descriptors without select's FD_SETSIZE ceiling."""

    poller = select.poll()
    events = select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL
    for fd in fds:
        poller.register(fd, events)
    timeout_ms = max(0, int(timeout_s * 1000 + 0.999))
    return {fd for fd, event in poller.poll(timeout_ms) if event & events}


def _child_exit_observed_without_reaping(pid: int) -> bool:
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        raise RuntimeError("waitid is unavailable")
    try:
        result = waitid(
            os.P_PID,
            pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:
        return True
    return result is not None


def _anchor(args: argparse.Namespace, command: list[str]) -> NoReturn:
    if os.getppid() != args.expected_parent_pid:
        _fixed_exit(_EXIT_OWNER_GONE)
    # Caught dispositions are reset by exec, so installing these before Popen
    # protects the anchor's spawn window without changing the server's defaults.
    signal.signal(signal.SIGTERM, _request_anchor_cleanup)
    signal.signal(signal.SIGUSR1, _request_forced_anchor_cleanup)
    server_env: dict[str, str] = {}
    try:
        _establish_linux_child_reaping_semantics()
        _set_linux_process_nondumpable()
        _set_linux_child_subreaper()
        _rendezvous = _validated_bound_rendezvous(
            args.rendezvous_fd,
            args.rendezvous_identity,
        )
        if _owner_is_gone(args.owner_fd, args.expected_parent_pid):
            _fixed_exit(_EXIT_OWNER_GONE)
        server_env = _read_server_environment(args.server_env_fd)
        anchor_pid = os.getpid()
        supervisor_pid = os.getppid()
        child = subprocess.Popen(
            command,
            stdin=0,
            stdout=1,
            stderr=2,
            close_fds=True,
            env=server_env,
            preexec_fn=partial(
                _install_linux_process_tree_filter,
                protected_process_pids=(anchor_pid, supervisor_pid),
                protected_process_groups=(
                    os.getpgrp(),
                    os.getpgid(supervisor_pid),
                ),
            ),
        )
        # The server owns its inherited stdio descriptors after Popen returns.
        # Keeping anchor-side copies would mask server-side stdin/stdout/stderr
        # closure from the client until the anchor itself begins cleanup.
        for fd in (0, 1, 2):
            with suppress(OSError):
                os.close(fd)
    except BaseException:
        _write_message(
            args.anchor_event_fd,
            {"nonce": args.nonce, "type": "start_failed"},
        )
        _fixed_exit(_EXIT_START_FAILED)
    finally:
        server_env.clear()

    trigger = ""
    server_returncode: int | None = None
    try:
        os.environ.clear()
        _write_message(
            args.anchor_event_fd,
            {
                "anchor_pid": os.getpid(),
                "nonce": args.nonce,
                "pgid": os.getpgrp(),
                "server_pid": child.pid,
                "type": "ready",
            },
        )
        while not trigger:
            if _owner_is_gone(args.owner_fd, args.expected_parent_pid):
                trigger = "owner_gone"
            elif _ANCHOR_CLEANUP_REQUESTED:
                trigger = "forced_close" if _ANCHOR_FORCE_REQUESTED else "graceful_close"
            else:
                server_returncode = child.poll()
                if server_returncode is not None:
                    trigger = "server_exit"
            if not trigger:
                time.sleep(_POLL_INTERVAL_S)
    except BaseException:
        # Once Popen succeeds, no publication or observation failure may release
        # this cleanup owner. In particular, supervisor loss can close the ready
        # pipe before this anchor enters its ordinary ownership loop.
        trigger = "post_dispatch_failure"
    cleanup_failure_reported = False
    force = trigger in {"forced_close", "server_exit"}
    while True:
        try:
            server_returncode, settled = _settle_anchor_tree(
                server_pid=child.pid,
                server_returncode=server_returncode,
                term_timeout_s=args.term_timeout_s,
                kill_timeout_s=args.kill_timeout_s,
                force=force,
            )
        except BaseException:
            settled = False
        if settled:
            break
        if not cleanup_failure_reported:
            with suppress(OSError):
                _write_message(
                    args.anchor_event_fd,
                    {
                        "nonce": args.nonce,
                        "returncode": server_returncode,
                        "type": "cleanup_failed",
                    },
                )
            cleanup_failure_reported = True
        # Retain exact ownership until this anchor proves quiescence or the
        # supervisor kills it and takes over the still-reserved process group.
        force = True
        time.sleep(_POLL_INTERVAL_S)

    with suppress(OSError):
        _write_message(
            args.anchor_event_fd,
            {
                "nonce": args.nonce,
                "returncode": server_returncode,
                "type": "server_exit",
            },
        )
    _fixed_exit(0)


def _decode_messages(buffer: bytearray) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    while True:
        newline = buffer.find(b"\n")
        if newline < 0:
            return messages
        raw = bytes(buffer[:newline])
        del buffer[: newline + 1]
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if type(value) is dict:
            messages.append(value)


def _nonce_matches(message: dict[str, object], nonce: str) -> bool:
    candidate = message.get("nonce")
    return type(candidate) is str and hmac.compare_digest(candidate, nonce)


def _validated_anchor_ready(
    message: dict[str, object],
    *,
    anchor_pid: int,
) -> dict[str, object] | None:
    reported_anchor = message.get("anchor_pid")
    reported_pgid = message.get("pgid")
    server_pid = message.get("server_pid")
    if (
        type(reported_anchor) is not int
        or reported_anchor != anchor_pid
        or type(reported_pgid) is not int
        or reported_pgid != anchor_pid
        or type(server_pid) is not int
        or server_pid <= 0
    ):
        return None
    anchor_stat = _linux_process_stat(anchor_pid)
    server_stat = _linux_process_stat(server_pid)
    if (
        anchor_stat is None
        or anchor_stat.process_group != anchor_pid
        or (
            server_stat is not None
            and (server_stat.parent_pid != anchor_pid or server_stat.process_group != anchor_pid)
        )
    ):
        return None
    return {
        "anchor_pid": anchor_pid,
        "nonce": message["nonce"],
        "pgid": anchor_pid,
        "server_pid": server_pid,
        "type": "ready",
    }


def _acquire_supervisor_rendezvous(
    args: argparse.Namespace,
    control: socket.socket,
) -> tuple[socket.socket | None, str]:
    """Wait for an older exact connector generation without dispatching a server."""

    try:
        address = _rendezvous_address(args.rendezvous_identity)
    except RuntimeError:
        return None, "start_failed"
    control_buffer = bytearray()
    control_fd = control.fileno()
    bound: socket.socket | None = None
    ready_reported = False
    launch_authorized = False
    while True:
        if bound is None:
            try:
                candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    candidate.bind(address)
                except OSError as error:
                    candidate.close()
                    if error.errno != errno.EADDRINUSE:
                        return None, "start_failed"
                else:
                    bound = candidate
            except OSError:
                return None, "start_failed"

        if _owner_is_gone(args.owner_fd, args.expected_parent_pid):
            if bound is not None:
                bound.close()
            return None, "owner_gone"
        readable = _poll_readable_descriptors(
            (args.owner_fd, control_fd),
            (
                0
                if bound is not None and not ready_reported and not control_buffer
                else _POLL_INTERVAL_S
            ),
        )
        trigger = ""
        if args.owner_fd in readable:
            try:
                if os.read(args.owner_fd, 1) == b"":
                    trigger = "owner_gone"
            except OSError:
                trigger = "owner_gone"
        if not trigger and control_fd in readable:
            try:
                chunk = control.recv(65536)
            except OSError:
                trigger = "owner_gone"
            else:
                if not chunk:
                    trigger = "owner_gone"
                else:
                    control_buffer.extend(chunk)
                    if len(control_buffer) > _MAX_CONTROL_BYTES:
                        trigger = "start_failed"
                    for message in _decode_messages(control_buffer):
                        if not _nonce_matches(message, args.nonce):
                            continue
                        message_type = message.get("type")
                        if message_type == "launch":
                            if bound is None or not ready_reported or launch_authorized:
                                trigger = "start_failed"
                                break
                            launch_authorized = True
                            continue
                        if message_type == "force":
                            trigger = "forced_close"
                            break
                        if message_type == "shutdown":
                            trigger = "graceful_close"
                            break
        if trigger:
            if bound is not None:
                bound.close()
            return None, trigger
        if bound is not None and not ready_reported and not control_buffer:
            try:
                control.sendall(
                    json.dumps(
                        {"nonce": args.nonce, "type": "rendezvous_ready"},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            except OSError:
                bound.close()
                return None, "owner_gone"
            ready_reported = True
        if bound is not None and launch_authorized and not control_buffer:
            return bound, ""


def _supervisor(args: argparse.Namespace, command: list[str]) -> NoReturn:
    if os.getppid() != args.expected_parent_pid:
        _fixed_exit(_EXIT_OWNER_GONE)
    try:
        _establish_linux_child_reaping_semantics()
        _set_linux_process_nondumpable()
        # If the anchor crashes, every surviving server descendant must be
        # adopted here so this process can become the replacement cleanup owner.
        _set_linux_child_subreaper()
    except BaseException:
        _fixed_exit(_EXIT_START_FAILED)
    control = socket.socket(fileno=args.control_fd)
    control.setblocking(False)
    rendezvous, rendezvous_trigger = _acquire_supervisor_rendezvous(args, control)
    if rendezvous_trigger:
        if rendezvous_trigger in {"forced_close", "graceful_close"}:
            with suppress(OSError):
                control.sendall(
                    json.dumps(
                        {
                            "nonce": args.nonce,
                            "reason": "startup_cancelled",
                            "type": "settled",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            _fixed_exit(0)
        if rendezvous_trigger == "start_failed":
            with suppress(OSError):
                control.sendall(
                    json.dumps(
                        {"nonce": args.nonce, "type": "start_failed"},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            _fixed_exit(_EXIT_START_FAILED)
        _fixed_exit(_EXIT_OWNER_GONE)
    if rendezvous is None:
        _fixed_exit(_EXIT_START_FAILED)
    anchor_read_fd, anchor_write_fd = os.pipe()
    helper = os.path.abspath(__file__)
    anchor_command = [
        sys.executable,
        "-I",
        "-S",
        helper,
        "--role",
        "anchor",
        "--nonce",
        args.nonce,
        "--expected-parent-pid",
        str(os.getpid()),
        "--owner-fd",
        str(args.owner_fd),
        "--anchor-event-fd",
        str(anchor_write_fd),
        "--server-env-fd",
        str(args.server_env_fd),
        "--rendezvous-fd",
        str(rendezvous.fileno()),
        "--rendezvous-identity",
        args.rendezvous_identity,
        "--term-timeout-s",
        str(args.term_timeout_s),
        "--kill-timeout-s",
        str(args.kill_timeout_s),
        "--",
        *command,
    ]
    try:
        anchor = subprocess.Popen(
            anchor_command,
            stdin=0,
            stdout=1,
            stderr=2,
            close_fds=True,
            env={},
            pass_fds=(
                args.owner_fd,
                args.server_env_fd,
                rendezvous.fileno(),
                anchor_write_fd,
            ),
            start_new_session=True,
        )
    except BaseException:
        with suppress(OSError):
            control.sendall(
                json.dumps(
                    {"nonce": args.nonce, "type": "start_failed"},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        _fixed_exit(_EXIT_START_FAILED)
    finally:
        os.close(anchor_write_fd)
        os.close(args.server_env_fd)

    os.environ.clear()

    for fd in (0, 1, 2):
        with suppress(OSError):
            os.close(fd)

    anchor_buffer = bytearray()
    control_buffer = bytearray()
    control_fd = control.fileno()
    ready: dict[str, object] | None = None
    server_returncode: int | None = None
    anchor_reported_settled = False
    trigger = ""
    while not trigger:
        if os.getppid() != args.expected_parent_pid:
            trigger = "owner_gone"
            break
        readable = _poll_readable_descriptors(
            (args.owner_fd, anchor_read_fd, control_fd),
            _POLL_INTERVAL_S,
        )
        if args.owner_fd in readable:
            try:
                if os.read(args.owner_fd, 1) == b"":
                    trigger = "owner_gone"
            except OSError:
                trigger = "owner_gone"
        if anchor_read_fd in readable:
            chunk = os.read(anchor_read_fd, 65536)
            if not chunk:
                trigger = trigger or "anchor_exit"
            else:
                anchor_buffer.extend(chunk)
                if len(anchor_buffer) > _MAX_CONTROL_BYTES:
                    trigger = "cleanup_failed"
                    break
                for message in _decode_messages(anchor_buffer):
                    if not _nonce_matches(message, args.nonce):
                        continue
                    if message.get("type") == "ready" and ready is None:
                        ready = _validated_anchor_ready(message, anchor_pid=anchor.pid)
                        if ready is None:
                            trigger = "cleanup_failed"
                            break
                        try:
                            control.sendall(
                                json.dumps(message, sort_keys=True, separators=(",", ":")).encode(
                                    "utf-8"
                                )
                                + b"\n"
                            )
                        except OSError:
                            trigger = "owner_gone"
                    elif message.get("type") == "start_failed":
                        with suppress(OSError):
                            control.sendall(
                                json.dumps(
                                    {"nonce": args.nonce, "type": "start_failed"},
                                    separators=(",", ":"),
                                ).encode("utf-8")
                                + b"\n"
                            )
                        trigger = "start_failed"
                    elif message.get("type") == "server_exit":
                        value = message.get("returncode")
                        server_returncode = value if type(value) is int else None
                        anchor_reported_settled = True
                        trigger = trigger or "server_exit"
                    elif message.get("type") == "cleanup_failed":
                        trigger = "cleanup_failed"
        if control_fd in readable:
            try:
                chunk = control.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                trigger = "owner_gone"
            else:
                control_buffer.extend(chunk)
                if len(control_buffer) > _MAX_CONTROL_BYTES:
                    trigger = "cleanup_failed"
                    break
                for message in _decode_messages(control_buffer):
                    if _nonce_matches(message, args.nonce) and message.get("type") == "shutdown":
                        trigger = "graceful_close"
                    elif _nonce_matches(message, args.nonce) and message.get("type") == "force":
                        trigger = "forced_close"
    settlement_reason = trigger
    # start_new_session makes the direct anchor the immutable group leader;
    # never derive cleanup authority from pipe-provided numeric identities.
    pgid = anchor.pid
    anchor_alive = not _child_exit_observed_without_reaping(anchor.pid)
    if trigger in {"forced_close", "server_exit"} and anchor_alive:
        with suppress(ProcessLookupError):
            os.kill(anchor.pid, signal.SIGUSR1)
    if anchor_alive:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            trigger = "cleanup_failed"

    deadline = time.monotonic() + args.term_timeout_s + args.kill_timeout_s + 0.5
    while not _child_exit_observed_without_reaping(anchor.pid) and time.monotonic() < deadline:
        readable = _poll_readable_descriptors(
            (anchor_read_fd, control_fd),
            min(_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())),
        )
        if anchor_read_fd in readable:
            chunk = os.read(anchor_read_fd, 65536)
            if not chunk:
                continue
            anchor_buffer.extend(chunk)
            if len(anchor_buffer) > _MAX_CONTROL_BYTES:
                trigger = "cleanup_failed"
                deadline = time.monotonic()
            for message in _decode_messages(anchor_buffer):
                if not _nonce_matches(message, args.nonce):
                    continue
                if message.get("type") == "server_exit":
                    value = message.get("returncode")
                    server_returncode = value if type(value) is int else None
                    anchor_reported_settled = True
                elif message.get("type") == "cleanup_failed":
                    trigger = "cleanup_failed"
        if control_fd in readable:
            try:
                chunk = control.recv(65536)
            except OSError:
                chunk = b""
            control_buffer.extend(chunk)
            if len(control_buffer) > _MAX_CONTROL_BYTES:
                trigger = "cleanup_failed"
                deadline = time.monotonic()
            for message in _decode_messages(control_buffer):
                if _nonce_matches(message, args.nonce) and message.get("type") == "force":
                    with suppress(ProcessLookupError):
                        os.kill(anchor.pid, signal.SIGUSR1)

    anchor_exited = _child_exit_observed_without_reaping(anchor.pid)
    if anchor_exited:
        # The anchor writes its settlement evidence before exiting. Drain the
        # now-closed pipe so a fast exit cannot race the select loop and hide a
        # valid receipt.
        while True:
            chunk = os.read(anchor_read_fd, 65536)
            if not chunk:
                break
            anchor_buffer.extend(chunk)
            if len(anchor_buffer) > _MAX_CONTROL_BYTES:
                trigger = "cleanup_failed"
                break
            for message in _decode_messages(anchor_buffer):
                if not _nonce_matches(message, args.nonce):
                    continue
                if message.get("type") == "server_exit":
                    value = message.get("returncode")
                    server_returncode = value if type(value) is int else None
                    anchor_reported_settled = True
                elif message.get("type") == "cleanup_failed":
                    trigger = "cleanup_failed"

    if not anchor_exited or not anchor_reported_settled:
        # The anchor is no longer positive cleanup evidence. Take over its
        # isolated group directly; as a subreaper, this supervisor owns every
        # surviving descendant after the anchor exits or is killed.
        try:
            fallback_settled = _settle_supervisor_tree(
                pgid=pgid,
                anchor_pid=anchor.pid,
                term_timeout_s=args.term_timeout_s,
                kill_timeout_s=args.kill_timeout_s,
            )
        except BaseException:
            fallback_settled = False
        if fallback_settled:
            anchor_reported_settled = True
            trigger = "fallback_settled"
            if settlement_reason in {"anchor_exit", "cleanup_failed", "start_failed"}:
                settlement_reason = "anchor_recovered"
        else:
            trigger = "cleanup_failed"

    if trigger != "cleanup_failed":
        # Reap the anchor only after every other member is gone. Until this
        # point its zombie identity reserves the numeric PID and PGID.
        with suppress(subprocess.TimeoutExpired):
            anchor.wait(timeout=0)

    if ready is None:
        _fixed_exit(_EXIT_START_FAILED)

    settled = {
        "nonce": args.nonce,
        "reason": settlement_reason if trigger != "cleanup_failed" else "cleanup_failed",
        "server_returncode": server_returncode,
        "type": "settled",
    }
    with suppress(OSError):
        control.sendall(
            json.dumps(settled, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
    if trigger == "cleanup_failed":
        # Publish the bounded-attempt failure, but retain exact cleanup
        # ownership. Exiting here would orphan work precisely when settlement
        # is uncertain. Each retry is independently bounded; a retained Cayu
        # close task observes the later authenticated success if the kernel
        # operation becomes quiescent.
        while True:
            time.sleep(_POLL_INTERVAL_S)
            try:
                recovered = _settle_supervisor_tree(
                    pgid=pgid,
                    anchor_pid=anchor.pid,
                    term_timeout_s=args.term_timeout_s,
                    kill_timeout_s=args.kill_timeout_s,
                )
            except BaseException:
                recovered = False
            if not recovered:
                continue
            with suppress(subprocess.TimeoutExpired):
                anchor.wait(timeout=0)
            with suppress(OSError):
                control.sendall(
                    json.dumps(
                        {
                            "nonce": args.nonce,
                            "reason": "anchor_recovered",
                            "server_returncode": server_returncode,
                            "type": "settled",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
            _fixed_exit(0)
    if settlement_reason == "owner_gone":
        _fixed_exit(_EXIT_OWNER_GONE)
    if server_returncode is not None and server_returncode != 0:
        _exit_with_server_returncode(server_returncode)
    _fixed_exit(0)


def main() -> NoReturn:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", choices=("supervisor", "anchor", "preflight"), required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-parent-pid", required=True, type=int)
    parser.add_argument("--owner-fd", required=True, type=int)
    parser.add_argument("--control-fd", type=int, default=-1)
    parser.add_argument("--anchor-event-fd", type=int, default=-1)
    parser.add_argument("--server-env-fd", type=int, default=-1)
    parser.add_argument("--rendezvous-fd", type=int, default=-1)
    parser.add_argument("--rendezvous-identity", default="")
    parser.add_argument("--term-timeout-s", required=True, type=float)
    parser.add_argument("--kill-timeout-s", type=float, default=2.0)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if args.role == "preflight":
        try:
            _verify_linux_containment_primitives()
        except BaseException:
            _fixed_exit(_EXIT_START_FAILED)
        _fixed_exit(0)
    if not command:
        _fixed_exit(_EXIT_START_FAILED)
    if args.role == "anchor":
        _anchor(args, command)
    _supervisor(args, command)


if __name__ == "__main__":
    main()

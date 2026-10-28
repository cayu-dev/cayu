"""Linux sandbox bootstrap, executed with Python -I before the vendor CLI.

Stdlib only. Landlock grants paths and TCP ports; seccomp keeps descendants in
one process group and closes socket and descriptor-transfer bypasses. No process
or time quotas are imposed here: the caller owns resource policy and deadlines.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import sys
from pathlib import Path


class _Ruleset(ctypes.Structure):
    _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _PortRule(ctypes.Structure):
    _fields_ = [("access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


class _Filter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(_Filter))]


# Native ABI only; alternate syscall ABIs are refused rather than normalized.
_SYSCALLS = {
    "x86_64": (
        0xC000003E,
        41,
        56,
        126,
        (101, 109, 112, 155, 165, 166, 272, 308, 310, 311, 321, 425, 426, 427, 438),
    ),
    "aarch64": (
        0xC00000B7,
        198,
        220,
        91,
        (117, 154, 157, 41, 40, 39, 97, 268, 270, 271, 280, 425, 426, 427, 438),
    ),
}
# Landlock ABI 6 does not mediate ownership, mode, timestamps or xattrs.
# Deny those changes globally, including through read-only descriptors.
_METADATA_WRITES = {
    "x86_64": (
        90,
        91,
        92,
        93,
        94,
        132,
        188,
        189,
        190,
        197,
        198,
        199,
        235,
        260,
        261,
        268,
        280,
        452,
        463,
        466,
        469,  # file_setattr (Linux 6.17): inode attributes by path
    ),
    "aarch64": (5, 6, 7, 14, 15, 16, 52, 53, 54, 55, 88, 452, 463, 466, 469),
}
_FS_READ = (1 << 0) | (1 << 2) | (1 << 3)  # execute, read file, read directory
_FS_FILE = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)


def _check(result: int, operation: str) -> int:
    if result < 0:
        raise OSError(ctypes.get_errno(), operation)
    return result


def confine(config: dict) -> None:
    """Install irreversible restrictions in this single-threaded child only."""
    if sys.platform != "linux" or platform.machine() not in _SYSCALLS:
        raise RuntimeError("linux_agent_launch requires Linux x86_64 or aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    abi = _check(libc.syscall(444, 0, 0, 1), "Landlock ABI query")
    if abi < 6:
        raise RuntimeError(
            "linux_agent_launch requires Landlock ABI 6 for signal and IPC isolation"
        )
    _check(libc.prctl(38, 1, 0, 0, 0), "no_new_privs")
    arch, socket_nr, clone_nr, capset_nr, denied = _SYSCALLS[platform.machine()]
    status = Path("/proc/self/status").read_text(encoding="ascii")
    capabilities = {
        name: int(value.strip(), 16)
        for line in status.splitlines()
        for name, separator, value in (line.partition(":"),)
        if separator and name in {"CapEff", "CapPrm", "CapInh", "CapAmb"}
    }
    if os.geteuid() == 0 and any(
        capabilities.get(name, -1) != 0 for name in ("CapEff", "CapPrm", "CapInh", "CapAmb")
    ):
        _check(libc.prctl(28, 0x0F, 0, 0, 0), "lock root privilege semantics")
    _check(libc.prctl(47, 4, 0, 0, 0), "clear ambient capabilities")
    header = (ctypes.c_uint32 * 2)(0x20080522, 0)
    words = (ctypes.c_uint32 * 6)()
    _check(libc.syscall(capset_nr, ctypes.byref(header), ctypes.byref(words)), "drop capabilities")

    # All filesystem rights through ABI 5, including truncate and device ioctl.
    fs = (1 << 16) - 1
    rules = _Ruleset(fs, 3, 3)  # deny TCP bind/connect; scope abstract sockets and signals
    fd = _check(
        libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0), "create Landlock ruleset"
    )
    try:
        for key, access in (("read_only", _FS_READ), ("writable", fs)):
            for name in config[key]:
                path = Path(name)
                parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    rights = access if path.is_dir() else access & _FS_FILE
                    rule = _PathRule(rights, parent)
                    _check(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), "add Landlock path")
                finally:
                    os.close(parent)
        for port in config["tcp_ports"]:
            rule = _PortRule(2, port)  # connect only; no listening TCP sockets
            _check(libc.syscall(445, fd, 2, ctypes.byref(rule), 0), "add Landlock TCP port")
        _check(libc.syscall(446, fd, 0), "restrict Landlock rights")
    finally:
        os.close(fd)

    allow, deny = 0x7FFF0000, 0x00050000 | errno.EPERM
    ins = [
        _Filter(0x20, 0, 0, 4),
        _Filter(0x15, 1, 0, arch),
        _Filter(0x06, 0, 0, 0x80000000),
        _Filter(0x20, 0, 0, 0),
    ]
    if arch == 0xC000003E:
        ins += [_Filter(0x45, 0, 1, 0x40000000), _Filter(0x06, 0, 0, deny)]
    for nr in (*denied, *_METADATA_WRITES[platform.machine()]):
        ins += [_Filter(0x15, 0, 1, nr), _Filter(0x06, 0, 0, deny)]
    # Filesystem ioctls can also mutate metadata through read-only descriptors.
    # Keep only terminal queries and nonblocking I/O controls for vendor stdio.
    ioctl_nr = 16 if arch == 0xC000003E else 29
    safe_ioctls = (0x5401, 0x5413, 0x541B, 0x5421, 0x5450, 0x5451)
    # TCGETS, TIOCGWINSZ, FIONREAD, FIONBIO, FIONCLEX, FIOCLEX.
    ins += [_Filter(0x15, 0, 2 * len(safe_ioctls) + 2, ioctl_nr), _Filter(0x20, 0, 0, 24)]
    for command in safe_ioctls:
        ins += [_Filter(0x15, 0, 1, command), _Filter(0x06, 0, 0, allow)]
    ins.append(_Filter(0x06, 0, 0, deny))
    # clone3 hides flags behind a pointer; ENOSYS lets libc use inspectable clone.
    ins += [_Filter(0x15, 0, 1, 435), _Filter(0x06, 0, 0, 0x00050000 | errno.ENOSYS)]
    # No new namespaces. Ordinary fork/threads continue to work.
    ins += [
        _Filter(0x15, 0, 4, clone_nr),
        _Filter(0x20, 0, 0, 16),
        _Filter(0x45, 0, 1, 0x7E020000),
        _Filter(0x06, 0, 0, deny),
        _Filter(0x06, 0, 0, allow),
    ]
    # Only INET stream sockets. Landlock mediates connect ports. UDP, raw,
    # packet, netlink and pathname UNIX sockets cannot bypass that policy.
    # Protocol 0 selects TCP for INET streams; all explicit protocols except
    # TCP are denied because Landlock does not mediate MPTCP or SCTP.
    ins += [
        _Filter(0x15, 0, 13, socket_nr),
        _Filter(0x20, 0, 0, 16),
        _Filter(0x15, 2, 0, 2),
        _Filter(0x15, 1, 0, 10),
        _Filter(0x06, 0, 0, deny),
        _Filter(0x20, 0, 0, 24),
        _Filter(0x54, 0, 0, 0xF),
        _Filter(0x15, 1, 0, 1),
        _Filter(0x06, 0, 0, deny),
        _Filter(0x20, 0, 0, 32),
        _Filter(0x15, 2, 0, 0),
        _Filter(0x15, 1, 0, 6),
        _Filter(0x06, 0, 0, deny),
        _Filter(0x06, 0, 0, allow),
    ]
    ins.append(_Filter(0x06, 0, 0, allow))
    array = (_Filter * len(ins))(*ins)
    program = _Program(len(ins), array)
    address = ctypes.cast(ctypes.pointer(program), ctypes.c_void_p).value
    _check(libc.prctl(22, 2, address, 0, 0), "install seccomp filter")


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    # SDK-specific command settings may be added by the trusted SDK. Host
    # environment and credentials are never inherited by the executed CLI.
    environment = config["environment"]
    for name in ("CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION"):
        if name in os.environ:
            environment[name] = os.environ[name]
    confine(config)
    if config.get("preflight"):
        return
    os.chdir(config["cwd"])
    os.execve(config["cli"], [config["cli"], *sys.argv[2:]], environment)


if __name__ == "__main__":
    main()

"""Owner tokens for allocations whose only records live in one process's memory.

When an application's session store does not outlive its process (the
in-memory store, or SQLite ``:memory:``), the allocation records of its
sessions end with that process. Nothing can resume those sessions or reconnect
to their sandboxes afterwards, so once the process is gone such a sandbox is
an orphan by construction. The runtime hands factories an owner token through
``EnvironmentFactoryRequest.process_local_owner``; a factory records it on
the resource (the Docker coding factory sets the
``io.cayu.process-local-owner`` container label), and a later process can use
:func:`process_local_owner_exited` to tell whether the owner is gone.

A token is ``v1:<boot>:<pid namespace>:<pid>:<start time>``. ``boot`` names
one boot of one kernel (Linux ``/proc/sys/kernel/random/boot_id``, macOS
``kern.bootsessionuuid``), the PID namespace is the inode of
``/proc/self/ns/pid`` on Linux (``host`` where there are none), and the start
time pins the exact process, so a reused PID never matches. A process that
cannot establish all of these gets no token, and its resources are never
reaped this way.

Allocations of durable stores never get a token: their sessions may still be
resumed or recovered, and ``CayuApp.recover_incomplete_sessions`` releases
them through their allocation records.
"""

from __future__ import annotations

import calendar
import os
import re
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

_OWNER_TOKEN = re.compile(
    r"v1:(?P<boot>[0-9a-f-]{8,64}):(?P<namespace>host|[0-9]{1,20}):"
    r"(?P<pid>[1-9][0-9]{0,9}):(?P<start>[0-9]{1,20})\Z"
)
_IDENTITY_TIMEOUT_S = 5.0


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=_IDENTITY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


@cache
def _boot_and_namespace() -> tuple[str, str] | None:
    """This kernel boot and PID namespace, or ``None`` when either is unknown."""

    if sys.platform.startswith("linux"):
        boot = _read_text("/proc/sys/kernel/random/boot_id")
        try:
            namespace = str(os.stat("/proc/self/ns/pid").st_ino)
        except OSError:
            return None
    elif sys.platform == "darwin":
        boot, namespace = _sysctl("kern.bootsessionuuid"), "host"
    else:
        return None
    if boot is None:
        return None
    boot = boot.lower()
    if re.fullmatch(r"[0-9a-f-]{8,64}", boot) is None:
        return None
    return boot, namespace


def _process_start(pid: int) -> str | None:
    """A value that differs between two processes that held ``pid`` in one boot."""

    if sys.platform.startswith("linux"):
        stat = _read_text(f"/proc/{pid}/stat")
        if stat is None or ")" not in stat:
            return None
        fields = stat.rsplit(")", 1)[1].split()
        # Field 22, start time in clock ticks since boot; fields[0] is field 3.
        start = fields[19] if len(fields) > 19 else None
        return start if start is not None and start.isdigit() else None
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=_IDENTITY_TIMEOUT_S,
                check=False,
                env={"LC_ALL": "C", "TZ": "UTC"},
            )
            if completed.returncode != 0:
                return None
            started = time.strptime(completed.stdout.strip(), "%a %b %d %H:%M:%S %Y")
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        # ps printed UTC; convert as UTC so every reader gets the same value.
        return str(calendar.timegm(started))
    return None


def _pid_runs(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    return True


def is_process_local_owner_token(value: object) -> bool:
    """Whether ``value`` has the shape of an owner token (not whether it is alive)."""

    return type(value) is str and _OWNER_TOKEN.fullmatch(value) is not None


def current_process_local_owner() -> str | None:
    """Return this process's owner token, or ``None`` when it cannot be proven."""

    identity = _boot_and_namespace()
    pid = os.getpid()
    start = _process_start(pid)
    if identity is None or start is None:
        return None
    boot, namespace = identity
    return f"v1:{boot}:{namespace}:{pid}:{start}"


def process_local_owner_exited(owner: str) -> bool:
    """Whether the process named by ``owner`` has certainly exited.

    True only for a well-formed token from this kernel boot and this PID
    namespace whose process is not this one and either no longer runs or was
    replaced by another process with the same PID (a different start time).
    Anything else returns False: another boot, host or PID namespace (where
    this process cannot see the owner's PIDs), a malformed token, a live owner,
    or an owner whose start time cannot be read. A caller that removes
    resources on True can therefore only err towards keeping one.
    """

    if type(owner) is not str:
        return False
    match = _OWNER_TOKEN.fullmatch(owner)
    identity = _boot_and_namespace()
    if match is None or identity is None or (match["boot"], match["namespace"]) != identity:
        return False
    pid = int(match["pid"])
    if pid == os.getpid():
        return False
    if not _pid_runs(pid):
        return True
    start = _process_start(pid)
    # A different start time means the PID was reused: the owner has exited.
    return start is not None and start != match["start"]


__all__ = [
    "current_process_local_owner",
    "is_process_local_owner_token",
    "process_local_owner_exited",
]

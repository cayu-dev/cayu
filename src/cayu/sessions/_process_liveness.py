"""Private, conservative same-host process-death evidence.

Never interpret a PID from another boot or PID namespace as a local process.
Evidence can only end ownership early when the process is provably gone; a live
process never extends an expired lease. Unavailable evidence changes nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    host_boot_id: str
    pid: int = Field(gt=0)
    start_id: str | None = None


@lru_cache(maxsize=1)
def _boot_id(pid: int) -> str | None:
    try:
        if sys.platform == "linux":
            identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        elif sys.platform == "darwin":
            identity = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1,
            ).strip()
        else:
            return None
        return sha256(identity.encode()).hexdigest() if identity else None
    except (OSError, subprocess.SubprocessError):
        return None


def _host_boot_id(pid: int) -> str | None:
    boot = _boot_id(pid)
    if boot is None:
        return None
    if sys.platform != "linux":
        return boot
    # Read namespaces each time: unshare/setns need not change the process PID.
    try:
        identity = (
            boot + ":" + os.readlink("/proc/self/ns/pid") + ":" + os.readlink("/proc/self/ns/user")
        )
    except OSError:
        return None
    return sha256(identity.encode()).hexdigest()


def _linux_process(pid: int) -> tuple[str, str]:
    # comm can contain spaces and parentheses. Fields after its closing ')' start
    # at field 3; starttime is field 22 and survives exec within the same process.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def _darwin_process_start(pid: int) -> str:
    return subprocess.check_output(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="],
        text=True,
        stderr=subprocess.DEVNULL,
        env={"LC_ALL": "C"},
        timeout=1,
    ).strip()


def current_process_identity() -> ProcessIdentity | None:
    pid = os.getpid()
    boot = _host_boot_id(pid)
    if boot is None:
        return None
    start_id = None
    if sys.platform == "linux":
        try:
            _, start_id = _linux_process(pid)
        except (OSError, ValueError, IndexError):
            return None
    elif sys.platform == "darwin":
        with suppress(OSError, subprocess.SubprocessError):
            start_id = _darwin_process_start(pid) or None
    return ProcessIdentity(host_boot_id=boot, pid=pid, start_id=start_id)


def process_liveness(identity: ProcessIdentity | None) -> Literal["alive", "dead", "unknown"]:
    if identity is None or identity.host_boot_id != _host_boot_id(os.getpid()):
        return "unknown"
    if sys.platform == "linux":
        try:
            state, start_id = _linux_process(identity.pid)
        except FileNotFoundError:
            # Restricted procfs visibility is not proof of process death.
            try:
                os.kill(identity.pid, 0)
            except ProcessLookupError:
                return "dead"
            except OSError:
                pass
            return "unknown"
        except (OSError, ValueError, IndexError):
            return "unknown"
        if state in {"Z", "X"}:
            return "dead"
        if identity.start_id is None:
            return "unknown"
        return "alive" if start_id == identity.start_id else "dead"
    try:
        os.kill(identity.pid, 0)
    except ProcessLookupError:
        return "dead"
    except OSError:
        return "unknown"
    if sys.platform == "darwin" and identity.start_id is not None:
        try:
            start_id = _darwin_process_start(identity.pid)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        if start_id:
            # A same-second PID reuse can conservatively delay recovery, but
            # cannot authorize takeover of a process that is still executing.
            return "alive" if start_id == identity.start_id else "dead"
    # Without a start identity, PID reuse cannot prove this owner alive.
    return "unknown"


def execution_owner_is_live(owner, now) -> bool:
    """Same-host process death ends ownership early; only the lease keeps it alive.

    A live process whose lease expired (for example, one cut off from the store)
    is not live: the lease contract lets a successor fence it, and fencing keeps
    the old process from publishing.
    """
    if owner.released:
        return False
    if process_liveness(owner.process_identity) == "dead":
        return False
    return owner.lease_expires_at > now

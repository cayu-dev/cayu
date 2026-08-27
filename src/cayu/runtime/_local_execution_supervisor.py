"""Stdlib-only Linux supervisor for one general local execution-attempt tree."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import json
import os
import select
import signal
import socket
import stat as stat_module
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

_POLL_SECONDS = 0.01
_MAX_CONTROL_BYTES = 65_536
_MAX_LAUNCH_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 65_536
_RENDEZVOUS_PREFIX = b"\0cayu-local-execution-v1-"
_TERMINATING_STATES = {"Z", "X"}
_STOPPED_STATES = {"T", "t", "Z", "X"}


class _LinuxProcessStat(Protocol):
    parent_pid: int
    process_group: int
    proc_inode: int
    start_tick: int
    state: str


class _KernelHelpers(Protocol):
    def _linux_process_stat(self, pid: int) -> _LinuxProcessStat | None: ...

    def _linux_pidfd_open(self, pid: int) -> int: ...

    def _linux_pidfd_send_signal(self, pidfd: int, signal_number: int) -> None: ...

    def _establish_linux_child_reaping_semantics(self) -> None: ...

    def _set_linux_process_nondumpable(self) -> None: ...

    def _set_linux_child_subreaper(self) -> None: ...


def _load_kernel_helpers() -> _KernelHelpers:
    helper_dir = Path(__file__).resolve().parents[1] / "mcp"
    sys.path.insert(0, str(helper_dir))
    try:
        kernel = importlib.import_module("_stdio_containment")
    finally:
        del sys.path[0]
    return cast("_KernelHelpers", kernel)


_kernel = _load_kernel_helpers()


def _exit(code: int) -> NoReturn:
    os._exit(code)


def _read_sealed_launch(fd: int) -> dict[str, Any]:
    import fcntl

    required_seals = 0x0001 | 0x0002 | 0x0004 | 0x0008
    if fcntl.fcntl(fd, 1034) & required_seals != required_seals:
        raise RuntimeError("local execution launch transfer was not sealed")
    size = os.fstat(fd).st_size
    if not 0 <= size <= _MAX_LAUNCH_BYTES:
        raise RuntimeError("local execution launch transfer exceeded its limit")
    chunks = bytearray()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while len(chunks) < size:
            chunk = os.read(fd, min(65_536, size - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) != size:
            raise RuntimeError("local execution launch transfer was incomplete")
        payload = json.loads(chunks.decode("utf-8"))
    finally:
        chunks.clear()
        os.close(fd)
    if type(payload) is not dict:
        raise RuntimeError("local execution launch transfer was malformed")
    if set(payload) != {"argv", "cwd", "env"}:
        raise RuntimeError("local execution launch transfer fields were invalid")
    argv = payload["argv"]
    cwd = payload["cwd"]
    environment = payload["env"]
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str or not item for item in argv)
        or (cwd is not None and type(cwd) is not str)
        or type(environment) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in environment.items())
    ):
        raise RuntimeError("local execution launch values were invalid")
    return {"argv": argv, "cwd": cwd, "env": environment}


def _rendezvous_address(identity: str) -> bytes:
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise RuntimeError("local execution rendezvous identity was invalid")
    address = (
        _RENDEZVOUS_PREFIX
        + b"u"
        + str(os.geteuid()).encode("ascii")
        + b"-"
        + identity.encode("ascii")
    )
    if len(address) > 107:
        raise RuntimeError("local execution rendezvous address exceeded its limit")
    return address


def _send(control: socket.socket, nonce: str, message_type: str, **payload: object) -> None:
    document = {"nonce": nonce, "type": message_type, **payload}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    control.sendall(encoded + b"\n")


def _process_identity(pid: int) -> dict[str, int] | None:
    stat = _kernel._linux_process_stat(pid)
    if stat is None:
        return None
    return {
        "pid": pid,
        "process_group": stat.process_group,
        "start_tick": stat.start_tick,
        "proc_inode": stat.proc_inode,
    }


def _all_process_stats() -> dict[int, _LinuxProcessStat]:
    stats: dict[int, _LinuxProcessStat] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            stat = _kernel._linux_process_stat(pid)
            if stat is not None:
                stats[pid] = stat
    return stats


def _descendants(supervisor_pid: int) -> dict[int, _LinuxProcessStat]:
    stats = _all_process_stats()
    owned = {supervisor_pid}
    changed = True
    while changed:
        changed = False
        for pid, stat in stats.items():
            if pid not in owned and stat.parent_pid in owned:
                owned.add(pid)
                changed = True
    owned.discard(supervisor_pid)
    return {pid: stats[pid] for pid in owned}


def _signal_exact(pid: int, stat: _LinuxProcessStat, signal_number: int) -> bool:
    try:
        pidfd = _kernel._linux_pidfd_open(pid)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        raise
    try:
        current = _kernel._linux_process_stat(pid)
        if current is None or (
            current.start_tick != stat.start_tick or current.proc_inode != stat.proc_inode
        ):
            return False
        try:
            _kernel._linux_pidfd_send_signal(pidfd, signal_number)
        except OSError as error:
            if error.errno == errno.ESRCH:
                return False
            raise
        return True
    finally:
        os.close(pidfd)


def _reap_children(root_pid: int, root_exit: int | None) -> tuple[int | None, bool]:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return root_exit, False
        if pid == 0:
            return root_exit, True
        if pid == root_pid and root_exit is None:
            root_exit = os.waitstatus_to_exitcode(status)


def _freeze_to_closure(
    supervisor_pid: int,
    deadline: float,
) -> tuple[dict[int, _LinuxProcessStat], bool]:
    previous: set[tuple[int, int, int]] | None = None
    while time.monotonic() < deadline:
        members = _descendants(supervisor_pid)
        identities = {(pid, stat.start_tick, stat.proc_inode) for pid, stat in members.items()}
        for pid, stat in members.items():
            if stat.state not in _STOPPED_STATES:
                _signal_exact(pid, stat, signal.SIGSTOP)
        refreshed = _descendants(supervisor_pid)
        refreshed_identities = {
            (pid, stat.start_tick, stat.proc_inode) for pid, stat in refreshed.items()
        }
        if (
            refreshed_identities == identities
            and identities == previous
            and all(stat.state in _STOPPED_STATES for stat in refreshed.values())
        ):
            return refreshed, True
        previous = refreshed_identities
        time.sleep(_POLL_SECONDS)
    return _descendants(supervisor_pid), False


def _settle_tree(
    *,
    supervisor_pid: int,
    root_pid: int,
    root_exit: int | None,
    term_grace: float,
    kill_grace: float,
    force: bool,
) -> tuple[int | None, bool, bool, bool, int]:
    observed: set[tuple[int, int, int]] = set()
    term_sent = False
    kill_sent = False
    if not force:
        members, frozen = _freeze_to_closure(
            supervisor_pid,
            time.monotonic() + term_grace,
        )
        for pid, stat in members.items():
            observed.add((pid, stat.start_tick, stat.proc_inode))
            _signal_exact(pid, stat, signal.SIGTERM)
            _signal_exact(pid, stat, signal.SIGCONT)
            term_sent = True
        if frozen:
            deadline = time.monotonic() + term_grace
            while time.monotonic() < deadline:
                root_exit, children_remain = _reap_children(root_pid, root_exit)
                members = _descendants(supervisor_pid)
                for pid, stat in members.items():
                    observed.add((pid, stat.start_tick, stat.proc_inode))
                if not children_remain and not members:
                    return root_exit, True, term_sent, kill_sent, len(observed)
                time.sleep(_POLL_SECONDS)

    deadline = time.monotonic() + kill_grace
    first_kill_pass = True
    while first_kill_pass or time.monotonic() < deadline:
        first_kill_pass = False
        members, _ = _freeze_to_closure(supervisor_pid, deadline)
        for pid, stat in members.items():
            observed.add((pid, stat.start_tick, stat.proc_inode))
            _signal_exact(pid, stat, signal.SIGKILL)
            kill_sent = True
        root_exit, children_remain = _reap_children(root_pid, root_exit)
        members = _descendants(supervisor_pid)
        if not children_remain and not members:
            return root_exit, True, term_sent, kill_sent, len(observed)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(_POLL_SECONDS, remaining))
    return root_exit, False, term_sent, kill_sent, len(observed)


def _retain_tree_cleanup_authority(
    *,
    args: argparse.Namespace,
    listener: socket.socket,
    state: dict[str, object],
    child: subprocess.Popen[bytes],
    root_exit: int | None,
    quiescent: bool,
    term_sent: bool,
    kill_sent: bool,
    descendants_observed: int,
) -> tuple[int | None, bool, bool, bool, int]:
    """Keep the authenticated subreaper alive until its exact tree is quiescent."""

    if quiescent:
        return root_exit, quiescent, term_sent, kill_sent, descendants_observed
    state["state"] = "terminal_not_quiescent"
    while not quiescent:
        with suppress(BaseException):
            _serve_rendezvous(listener, state)
        try:
            (
                root_exit,
                quiescent,
                attempted_term,
                attempted_kill,
                attempt_observed,
            ) = _settle_tree(
                supervisor_pid=os.getpid(),
                root_pid=child.pid,
                root_exit=root_exit,
                term_grace=args.term_grace_seconds,
                kill_grace=args.kill_grace_seconds,
                force=True,
            )
        except BaseException:
            time.sleep(_POLL_SECONDS)
            continue
        term_sent = term_sent or attempted_term
        kill_sent = kill_sent or attempted_kill
        descendants_observed = max(descendants_observed, attempt_observed)
        if not quiescent:
            time.sleep(_POLL_SECONDS)
    state["state"] = "quiescent"
    return root_exit, quiescent, term_sent, kill_sent, descendants_observed


def _path_contains_exact_receipt(path: Path, expected: bytes) -> bool:
    fd = -1
    try:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        observed = os.fstat(fd)
        if (
            not stat_module.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_size != len(expected)
        ):
            return False
        encoded = bytearray()
        while len(encoded) < len(expected):
            chunk = os.read(fd, len(expected) - len(encoded))
            if not chunk:
                break
            encoded.extend(chunk)
        return bytes(encoded) == expected
    except OSError:
        return False
    finally:
        if fd >= 0:
            os.close(fd)


def _atomic_receipt(path: Path, payload: dict[str, object]) -> None:
    staging = path.with_name(f"{path.name}.staging")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise RuntimeError("local execution receipt exceeded its limit")
    fd = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        observed = os.fstat(fd)
        if not stat_module.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid():
            raise RuntimeError("local execution receipt staging authority conflicted")
        os.fchmod(fd, 0o600)
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(staging, path)
    except OSError:
        # Recovery may have authenticated and promoted the deterministic stage
        # after this publisher closed its rendezvous listener.  An exact final
        # payload is positive acknowledgement of this publication; every other
        # rename failure remains authoritative.
        if not _path_contains_exact_receipt(path, encoded):
            raise
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_receipt(
    listener: socket.socket,
    path: Path,
    payload: dict[str, object],
) -> None:
    # Quiescence becomes durable only after the prior generation has released
    # its rendezvous identity.  A replacement can therefore never observe a
    # retry-admissible receipt while the old supervisor still owns that gate.
    with suppress(OSError):
        listener.close()
    _atomic_receipt(path, payload)


def _receipt(
    *,
    args: argparse.Namespace,
    root_identity: dict[str, int] | None,
    reason: str,
    exit_code: int | None,
    term_sent: bool,
    kill_sent: bool,
    descendants_observed: int,
    quiescence: str,
    effect_outcome: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt_id": args.attempt_id,
        "boot_id": args.boot_id,
        "descendants_observed": descendants_observed,
        "effect_outcome": effect_outcome,
        "exit_code": exit_code,
        "host_identity": args.host_identity,
        "kill_sent": kill_sent,
        "quiescence": quiescence,
        "request_sha256": args.request_sha256,
        "root": root_identity,
        "settled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "supervisor_nonce": args.nonce,
        "term_sent": term_sent,
        "terminal_reason": reason,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _owner_gone(owner_fd: int, expected_parent_pid: int) -> bool:
    if os.getppid() != expected_parent_pid:
        return True
    poller = select.poll()
    poller.register(owner_fd, select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL)
    if not poller.poll(0):
        return False
    try:
        return os.read(owner_fd, 1) == b""
    except OSError:
        return True


def _control_messages(control: socket.socket, buffer: bytearray) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    try:
        chunk = control.recv(65_536)
    except BlockingIOError:
        return messages
    if not chunk:
        return [{"type": "owner_gone"}]
    buffer.extend(chunk)
    if len(buffer) > _MAX_CONTROL_BYTES:
        raise RuntimeError("local execution control stream exceeded its limit")
    while b"\n" in buffer:
        line, _, remainder = buffer.partition(b"\n")
        buffer[:] = remainder
        value = json.loads(line.decode("utf-8"))
        if type(value) is dict:
            messages.append(value)
    return messages


def _serve_rendezvous(listener: socket.socket, state: dict[str, object]) -> None:
    while True:
        try:
            connection, _ = listener.accept()
        except BlockingIOError:
            return
        with connection:
            encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
            with suppress(OSError):
                connection.sendall(encoded + b"\n")


def _settle_after_internal_failure(
    *,
    args: argparse.Namespace,
    listener: socket.socket,
    child: subprocess.Popen[bytes],
    root_identity: dict[str, int] | None,
) -> NoReturn:
    """Fail closed after dispatch without abandoning the owned process tree."""

    root_exit: int | None = None
    quiescent = False
    term_sent = False
    kill_sent = False
    observed = 0
    root_exit, quiescent, term_sent, kill_sent, observed = _retain_tree_cleanup_authority(
        args=args,
        listener=listener,
        state={
            "attempt_id": args.attempt_id,
            "nonce": args.nonce,
            "state": "terminal_not_quiescent",
        },
        child=child,
        root_exit=root_exit,
        quiescent=quiescent,
        term_sent=term_sent,
        kill_sent=kill_sent,
        descendants_observed=observed,
    )
    receipt = _receipt(
        args=args,
        root_identity=root_identity,
        reason="supervisor_internal_failure",
        exit_code=root_exit,
        term_sent=term_sent,
        kill_sent=kill_sent,
        descendants_observed=observed,
        quiescence="quiescent",
        effect_outcome="outcome_unknown",
    )
    _publish_receipt(listener, Path(args.receipt_path), receipt)
    _exit(72)


def _supervise_started_attempt(
    *,
    args: argparse.Namespace,
    control: socket.socket,
    listener: socket.socket,
    state: dict[str, object],
    control_buffer: bytearray,
    child: subprocess.Popen[bytes],
    root_identity: dict[str, int],
    detached: bool,
) -> NoReturn:
    if detached:
        receipt = _receipt(
            args=args,
            root_identity=root_identity,
            reason="persistent_detached",
            exit_code=None,
            term_sent=False,
            kill_sent=False,
            descendants_observed=1,
            quiescence="persistent_detached",
            effect_outcome="outcome_unknown",
        )
        _publish_receipt(listener, Path(args.receipt_path), receipt)
        state.update({"receipt": receipt, "state": "settled"})
        with suppress(BaseException):
            _send(control, args.nonce, "settled", receipt=receipt)
        _exit(0)

    root_exit: int | None = None
    trigger = ""
    force = False
    deadline = None if args.deadline_seconds is None else time.monotonic() + args.deadline_seconds
    while not trigger:
        _serve_rendezvous(listener, state)
        root_exit, _children_remain = _reap_children(child.pid, root_exit)
        if root_exit is not None:
            trigger = "root_exit"
            break
        if deadline is not None and time.monotonic() >= deadline:
            trigger = "deadline"
            break
        if _owner_gone(args.owner_fd, args.expected_parent_pid):
            trigger = "owner_gone"
            break
        poller = select.poll()
        poller.register(control.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
        poller.register(listener.fileno(), select.POLLIN)
        for fd, _event in poller.poll(10):
            if fd == listener.fileno():
                _serve_rendezvous(listener, state)
            elif fd == control.fileno():
                for message in _control_messages(control, control_buffer):
                    message_type = message.get("type")
                    if message_type in {"shutdown", "owner_gone"}:
                        trigger = "cancelled" if message_type == "shutdown" else "owner_gone"
                    elif message_type == "kill":
                        trigger = "forced"
                        force = True
    root_exit, quiescent, term_sent, kill_sent, observed = _settle_tree(
        supervisor_pid=os.getpid(),
        root_pid=child.pid,
        root_exit=root_exit,
        term_grace=args.term_grace_seconds,
        kill_grace=args.kill_grace_seconds,
        force=force,
    )
    root_exit, quiescent, term_sent, kill_sent, observed = _retain_tree_cleanup_authority(
        args=args,
        listener=listener,
        state=state,
        child=child,
        root_exit=root_exit,
        quiescent=quiescent,
        term_sent=term_sent,
        kill_sent=kill_sent,
        descendants_observed=observed,
    )
    if not quiescent or args.effect_policy != "local_only":
        # Process termination is not a downstream effect receipt. Even a zero
        # exit status cannot prove whether an external mutation committed, and
        # a surviving process tree cannot prove even a local-only outcome.
        effect_outcome = "outcome_unknown"
    elif trigger == "root_exit":
        effect_outcome = "succeeded" if root_exit == 0 else "failed"
    else:
        effect_outcome = "failed"
    receipt = _receipt(
        args=args,
        root_identity=root_identity,
        reason=trigger,
        exit_code=root_exit,
        term_sent=term_sent,
        kill_sent=kill_sent,
        descendants_observed=observed,
        quiescence="quiescent",
        effect_outcome=effect_outcome,
    )
    _publish_receipt(listener, Path(args.receipt_path), receipt)
    state.update({"receipt": receipt, "state": "settled"})
    with suppress(BaseException):
        _send(control, args.nonce, "settled", receipt=receipt)
    _exit(0)


def _supervise(args: argparse.Namespace) -> NoReturn:
    if os.getppid() != args.expected_parent_pid:
        _exit(71)
    control = socket.socket(fileno=args.control_fd)
    control.setblocking(False)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.setblocking(False)
    try:
        listener.bind(_rendezvous_address(args.rendezvous_identity))
        listener.listen(8)
        _kernel._establish_linux_child_reaping_semantics()
        _kernel._set_linux_process_nondumpable()
        _kernel._set_linux_child_subreaper()
        launch = _read_sealed_launch(args.launch_fd)
        supervisor_identity = _process_identity(os.getpid())
        if supervisor_identity is None:
            raise RuntimeError("supervisor process identity was unavailable")
        _send(control, args.nonce, "ready", supervisor=supervisor_identity)
    except BaseException:
        with suppress(BaseException):
            _send(control, args.nonce, "start_failed")
        _exit(72)

    state: dict[str, object] = {
        "attempt_id": args.attempt_id,
        "nonce": args.nonce,
        "state": "ready",
    }
    control_buffer = bytearray()
    launch_authorized = False
    while not launch_authorized:
        if _owner_gone(args.owner_fd, args.expected_parent_pid):
            receipt = _receipt(
                args=args,
                root_identity=None,
                reason="owner_gone_before_dispatch",
                exit_code=None,
                term_sent=False,
                kill_sent=False,
                descendants_observed=0,
                quiescence="not_dispatched",
                effect_outcome="not_started",
            )
            _publish_receipt(listener, Path(args.receipt_path), receipt)
            _exit(0)
        poller = select.poll()
        poller.register(control.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
        poller.register(listener.fileno(), select.POLLIN)
        for fd, _event in poller.poll(10):
            if fd == listener.fileno():
                _serve_rendezvous(listener, state)
            elif fd == control.fileno():
                for message in _control_messages(control, control_buffer):
                    if message.get("nonce") == args.nonce and message.get("type") == "launch":
                        launch_authorized = True
                    elif message.get("type") in {"kill", "shutdown", "owner_gone"}:
                        receipt = _receipt(
                            args=args,
                            root_identity=None,
                            reason="cancelled_before_dispatch",
                            exit_code=None,
                            term_sent=False,
                            kill_sent=False,
                            descendants_observed=0,
                            quiescence="not_dispatched",
                            effect_outcome="not_started",
                        )
                        _publish_receipt(listener, Path(args.receipt_path), receipt)
                        _exit(0)

    detached = args.lifetime == "persistent_detached"
    devnull_fd: int | None = None
    child: subprocess.Popen[bytes] | None = None
    root_identity: dict[str, int] | None = None
    try:
        if detached:
            devnull_fd = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
        child = subprocess.Popen(
            launch["argv"],
            cwd=launch["cwd"],
            env=launch["env"],
            stdin=devnull_fd if detached else 0,
            stdout=devnull_fd if detached else 1,
            stderr=devnull_fd if detached else 2,
            close_fds=True,
            # Keep process-group-directed signals from the workload inside the
            # owned root tree. The supervisor remains its external subreaper
            # and cleanup authority even when the root calls kill(0, ...).
            start_new_session=True,
        )
        root_identity = _process_identity(child.pid)
        if root_identity is None:
            raise RuntimeError("root process identity was unavailable")
        _send(control, args.nonce, "started", root=root_identity)
        state.update({"root": root_identity, "state": "running"})
        for fd in (0, 1, 2):
            with suppress(OSError):
                os.close(fd)
        launch["env"].clear()
        launch.clear()
        if devnull_fd is not None:
            os.close(devnull_fd)
    except BaseException:
        if devnull_fd is not None:
            with suppress(OSError):
                os.close(devnull_fd)
        with suppress(BaseException):
            _send(control, args.nonce, "launch_failed")
        if child is not None:
            # Popen is the dispatch boundary.  Once it returns, neither a
            # missing /proc identity nor a lost owner acknowledgement can turn
            # the operation back into "not dispatched".  Settle the complete
            # descendant closure before publishing conservative evidence.
            (
                root_exit,
                quiescent,
                term_sent,
                kill_sent,
                observed,
            ) = _settle_tree(
                supervisor_pid=os.getpid(),
                root_pid=child.pid,
                root_exit=None,
                term_grace=args.term_grace_seconds,
                kill_grace=args.kill_grace_seconds,
                force=False,
            )
            (
                root_exit,
                quiescent,
                term_sent,
                kill_sent,
                observed,
            ) = _retain_tree_cleanup_authority(
                args=args,
                listener=listener,
                state=state,
                child=child,
                root_exit=root_exit,
                quiescent=quiescent,
                term_sent=term_sent,
                kill_sent=kill_sent,
                descendants_observed=observed,
            )
            receipt = _receipt(
                args=args,
                root_identity=root_identity,
                reason="launch_authority_failed",
                exit_code=root_exit,
                term_sent=term_sent,
                kill_sent=kill_sent,
                descendants_observed=observed,
                quiescence="quiescent",
                effect_outcome="outcome_unknown",
            )
            _publish_receipt(listener, Path(args.receipt_path), receipt)
            _exit(72)
        receipt = _receipt(
            args=args,
            root_identity=None,
            reason="launch_failed",
            exit_code=None,
            term_sent=False,
            kill_sent=False,
            descendants_observed=0,
            quiescence="not_dispatched",
            effect_outcome="not_started",
        )
        _publish_receipt(listener, Path(args.receipt_path), receipt)
        _exit(72)

    assert root_identity is not None
    try:
        _supervise_started_attempt(
            args=args,
            control=control,
            listener=listener,
            state=state,
            control_buffer=control_buffer,
            child=child,
            root_identity=root_identity,
            detached=detached,
        )
    except BaseException:
        _settle_after_internal_failure(
            args=args,
            listener=listener,
            child=child,
            root_identity=root_identity,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--owner-fd", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--launch-fd", type=int, required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--rendezvous-identity", required=True)
    parser.add_argument("--host-identity", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--effect-policy", required=True)
    parser.add_argument(
        "--lifetime",
        choices=("parent_death_containment", "graceful_cleanup", "persistent_detached"),
        required=True,
    )
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--term-grace-seconds", type=float, required=True)
    parser.add_argument("--kill-grace-seconds", type=float, required=True)
    return parser


if __name__ == "__main__":
    try:
        _supervise(_parser().parse_args())
    except BaseException:
        _exit(72)

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    staging = path.with_suffix(f"{path.suffix}.staging-{os.getpid()}")
    staging.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(staging, path)


def _rendezvous_rebind_denied() -> bool | None:
    identity = os.environ.get("CAYU_TEST_CONTAINMENT_RENDEZVOUS_IDENTITY")
    if identity is None:
        return None
    address = (
        b"\0cayu-mcp-containment-v1-u"
        + str(os.geteuid()).encode("ascii")
        + b"-"
        + identity.encode("ascii")
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as contender:
        try:
            contender.bind(address)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return True
            raise
    return False


def _grandchild(
    lock_path: Path,
    ready_path: Path,
    *,
    attempt_detach: bool,
    churn_descendants: bool,
) -> int:
    detachment_denied = False
    if attempt_detach:
        try:
            os.setsid()
        except PermissionError:
            detachment_denied = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock_file = lock_path.open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 23
    _write_json(
        ready_path,
        {
            "grandchild_detachment_denied": detachment_denied,
            "grandchild_pid": os.getpid(),
            "grandchild_pgid": os.getpgrp(),
            "grandchild_churns_descendants": churn_descendants,
        },
    )
    while True:
        if not churn_descendants:
            time.sleep(1)
            continue
        child_pid = os.fork()
        if child_pid == 0:
            time.sleep(0.02)
            os._exit(0)
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
        time.sleep(0.002)


def _server(
    lock_path: Path,
    state_path: Path,
    *,
    detach_grandchild: bool,
    churn_descendants: bool,
    eof_marker_path: Path | None,
    eof_release_path: Path | None,
    attack_parent: bool,
) -> int:
    ready_path = state_path.with_suffix(f"{state_path.suffix}.grandchild")
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            "grandchild",
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(ready_path),
            *(["--detach-grandchild"] if detach_grandchild else []),
            *(["--churn-descendants"] if churn_descendants else []),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    deadline = time.monotonic() + 5.0
    while not ready_path.exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ready_path.exists():
        return child.returncode if child.returncode is not None else 24
    grandchild = json.loads(ready_path.read_text(encoding="utf-8"))
    cleanup_owner_controls: dict[str, bool] = {}
    rendezvous_rebind_denied = _rendezvous_rebind_denied()
    if rendezvous_rebind_denied is not None:
        cleanup_owner_controls["server_rendezvous_rebind_denied"] = rendezvous_rebind_denied
    if attack_parent:
        process_status = Path("/proc/self/status").read_text(encoding="ascii")
        capability_values: dict[str, int] = {}
        for line in process_status.splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"CapEff", "CapPrm", "CapAmb"}:
                capability_values[name] = int(value.strip(), 16)
        cleanup_owner_controls["server_capabilities_dropped"] = all(
            capability_values.get(name) == 0 for name in ("CapEff", "CapPrm", "CapAmb")
        )
        anchor_pid = os.getppid()
        anchor_stat = Path(f"/proc/{anchor_pid}/stat").read_text(encoding="ascii")
        anchor_suffix = anchor_stat[anchor_stat.rfind(")") + 2 :].split()
        supervisor_pid = int(anchor_suffix[1])
        for name, pid in (("anchor", anchor_pid), ("supervisor", supervisor_pid)):
            try:
                with Path(f"/proc/{pid}/mem").open("rb", buffering=0):
                    pass
            except PermissionError:
                cleanup_owner_controls[f"{name}_memory_denied"] = True
            else:
                cleanup_owner_controls[f"{name}_memory_denied"] = False
        _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.prlimit(  # ty: ignore[unresolved-attribute]
                anchor_pid,
                resource.RLIMIT_NOFILE,
                (0, hard_limit),
            )
        except PermissionError:
            cleanup_owner_controls["anchor_prlimit_denied"] = True
        else:
            cleanup_owner_controls["anchor_prlimit_denied"] = False
        try:
            os.kill(supervisor_pid, signal.SIGKILL)
        except PermissionError:
            cleanup_owner_controls["supervisor_signal_denied"] = True
        else:
            cleanup_owner_controls["supervisor_signal_denied"] = False
        try:
            os.kill(anchor_pid, signal.SIGKILL)
        except PermissionError:
            cleanup_owner_controls["anchor_signal_denied"] = True
        else:
            cleanup_owner_controls["anchor_signal_denied"] = False
    _write_json(
        state_path,
        {
            **cleanup_owner_controls,
            "server_pid": os.getpid(),
            "server_pgid": os.getpgrp(),
            **grandchild,
        },
    )
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        request_id = message["id"]
        method = message.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "process-tree", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": []}
        else:
            result = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
        sys.stdout.flush()
    if eof_marker_path is not None:
        time.sleep(0.1)
        eof_marker_path.write_text("graceful", encoding="utf-8")
    while eof_release_path is not None and not eof_release_path.exists():
        time.sleep(0.01)
    return 0


def _launcher(
    lock_path: Path,
    state_path: Path,
    *,
    detach_grandchild: bool,
    churn_descendants: bool,
    eof_marker_path: Path | None,
    eof_release_path: Path | None,
    attack_parent: bool,
) -> int:
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            "server",
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
            *(["--detach-grandchild"] if detach_grandchild else []),
            *(["--churn-descendants"] if churn_descendants else []),
            *(["--attack-parent"] if attack_parent else []),
            *(["--eof-marker-path", str(eof_marker_path)] if eof_marker_path is not None else []),
            *(
                ["--eof-release-path", str(eof_release_path)]
                if eof_release_path is not None
                else []
            ),
        ],
        stdin=0,
        stdout=1,
        stderr=2,
        close_fds=True,
    )
    return process.wait()


def _probe(lock_path: Path) -> int:
    lock_file = lock_path.open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 25
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("grandchild", "server", "launcher", "probe"))
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--detach-grandchild", action="store_true")
    parser.add_argument("--churn-descendants", action="store_true")
    parser.add_argument("--attack-parent", action="store_true")
    parser.add_argument("--eof-marker-path", type=Path)
    parser.add_argument("--eof-release-path", type=Path)
    args = parser.parse_args()
    if args.role == "grandchild":
        assert args.state_path is not None
        return _grandchild(
            args.lock_path,
            args.state_path,
            attempt_detach=args.detach_grandchild,
            churn_descendants=args.churn_descendants,
        )
    if args.role == "probe":
        return _probe(args.lock_path)
    assert args.state_path is not None
    if args.role == "launcher":
        return _launcher(
            args.lock_path,
            args.state_path,
            detach_grandchild=args.detach_grandchild,
            churn_descendants=args.churn_descendants,
            eof_marker_path=args.eof_marker_path,
            eof_release_path=args.eof_release_path,
            attack_parent=args.attack_parent,
        )
    return _server(
        args.lock_path,
        args.state_path,
        detach_grandchild=args.detach_grandchild,
        churn_descendants=args.churn_descendants,
        eof_marker_path=args.eof_marker_path,
        eof_release_path=args.eof_release_path,
        attack_parent=args.attack_parent,
    )


if __name__ == "__main__":
    raise SystemExit(main())

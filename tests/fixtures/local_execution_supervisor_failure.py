"""Exercise the supervisor's post-dispatch internal-failure envelope."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cayu.runtime import _local_execution_supervisor as supervisor


def main() -> None:
    tree_fixture = Path(sys.argv[1])
    tree_state = Path(sys.argv[2])
    receipt_path = Path(sys.argv[3])
    supervisor._kernel._establish_linux_child_reaping_semantics()
    supervisor._kernel._set_linux_child_subreaper()
    child = subprocess.Popen(
        [sys.executable, str(tree_fixture), "child", str(tree_state)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    required = tuple(
        tree_state / f"{role}.json" for role in ("child", "grandchild", "background_server")
    )
    deadline = time.monotonic() + 5
    while not all(path.is_file() for path in required):
        if child.poll() is not None:
            raise RuntimeError("fixture tree exited before the internal failure boundary")
        if time.monotonic() >= deadline:
            raise TimeoutError("fixture tree did not become ready")
        time.sleep(0.01)
    root_identity = supervisor._process_identity(child.pid)
    if root_identity is None:
        raise RuntimeError("fixture root identity was unavailable")
    control, peer = socket.socketpair()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    args = SimpleNamespace(
        attempt_id="a" * 64,
        boot_id="fixture-boot",
        deadline_seconds=30.0,
        effect_policy="local_only",
        expected_parent_pid=0,
        host_identity="b" * 64,
        kill_grace_seconds=1.0,
        nonce="c" * 64,
        owner_fd=-1,
        receipt_path=str(receipt_path),
        request_sha256="d" * 64,
        term_grace_seconds=0.1,
    )

    def fail_after_dispatch(_listener: socket.socket, _state: dict[str, object]) -> None:
        raise RuntimeError("injected supervisor loop failure")

    supervisor._serve_rendezvous = fail_after_dispatch  # type: ignore[assignment]
    try:
        supervisor._supervise_started_attempt(
            args=args,
            control=control,
            listener=listener,
            state={"state": "running"},
            control_buffer=bytearray(),
            child=child,
            root_identity=root_identity,
            detached=False,
        )
    except BaseException:
        supervisor._settle_after_internal_failure(
            args=args,
            listener=listener,
            child=child,
            root_identity=root_identity,
        )
    finally:
        peer.close()
        control.close()


if __name__ == "__main__":
    main()

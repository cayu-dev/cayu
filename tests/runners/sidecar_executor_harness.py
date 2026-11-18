"""An in-process command executor for sidecar tests; it needs only the standard library."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_sidecar.executor import AgentNamespace, Executor, ExecutorServer
from examples.aws.lambda_microvm_sidecar.executor_protocol import ExecutorClient, listen, peer_uid

# Abstract sockets and SO_PEERCRED are Linux-only.
EXECUTOR_PEER_CREDENTIALS = sys.platform == "linux"


def agent_namespace_unavailable() -> str | None:
    """Why this host cannot create the executor's agent namespace, or None when it can."""
    if sys.platform != "linux":
        return "the agent namespace needs Linux"
    try:
        namespace = AgentNamespace(sys.executable)
    except (OSError, RuntimeError) as exc:
        return f"this host does not allow unprivileged user namespaces: {exc}"
    namespace.process.kill()
    namespace.process.wait()
    return None


class InProcessExecutor:
    """A command executor served from this process, as the agent container's is.

    On Linux it listens on an abstract socket and checks the caller's peer
    credentials, with this process's UID as the supervisor's. Elsewhere it
    listens on a socket in a private directory and skips the peer check.
    ``isolate`` runs commands in an agent namespace, which needs Linux with
    unprivileged user namespaces.
    """

    def __init__(self, root: Path, *, isolate: bool = False) -> None:
        self.executor = Executor(root=root, isolate=isolate)
        self._directory: str | None = None
        if EXECUTOR_PEER_CREDENTIALS:
            self.address = f"@cayu-test-executor-{uuid.uuid4().hex}"
            check = peer_uid
        else:
            # Unix socket paths are short; a test's tmp_path may not fit.
            self._directory = tempfile.mkdtemp(prefix="cayu-executor-", dir="/tmp")
            self.address = f"{self._directory}/executor.sock"
            uid = os.getuid()

            def check(_connection: Any) -> int:
                return uid

        self.server = ExecutorServer(
            self.executor, listen(self.address), supervisor_uid=os.getuid(), peer_uid=check
        )
        self.server.start()
        weakref.finalize(
            self, InProcessExecutor._close, self.server, self.executor, self._directory
        )

    def client(self, *, session: str | None = None, reconnect_s: float = 5.0) -> ExecutorClient:
        return ExecutorClient(
            self.address, session=session or str(uuid.uuid4()), reconnect_s=reconnect_s
        )

    def close(self) -> None:
        InProcessExecutor._close(self.server, self.executor, self._directory)

    @staticmethod
    def _close(server: ExecutorServer, executor: Executor, directory: str | None) -> None:
        server.close()
        executor.close()
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

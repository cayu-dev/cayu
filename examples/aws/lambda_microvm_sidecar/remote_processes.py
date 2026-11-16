"""Supervisor-side handles for commands the agent container's executor runs.

In Kubernetes mode the supervisor keeps every command record, owner fence,
output limit and credential, and asks the executor (``executor.py``) to do
each process step. ``RemoteProcess`` stands in for ``subprocess.Popen``: its
pipes move bytes by offset, so a retried request after a reset connection
neither loses nor repeats any, and the supervisor's own readers apply its
output limits and backpressure exactly as they do to a local pipe.
"""

from __future__ import annotations

import base64
import contextlib
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, cast

from .executor_protocol import (
    MAX_CHUNK_BYTES,
    MAX_WAIT_SECONDS,
    ExecutorClient,
    ExecutorError,
)
from .supervisor import (
    CommandExecutionBoundary,
    CommandRequestError,
    ExecutionProfile,
    ProcessHandle,
    _CommandRecord,
)


class RemoteProcesses:
    """Run commands in the agent container, as its own user, through its executor.

    The executor reports whether it can give interactive commands their own PID
    namespace; without one they fall back to process-group teardown, as the
    executor reports.
    """

    def __init__(self, client: ExecutorClient) -> None:
        self.client = client
        self._pid_namespace: bool | None = None

    def hello(self) -> dict[str, Any]:
        """The executor's boot ID and abilities, without waiting for it to come back."""
        response = self.client.call("hello", reconnect_s=0)
        if type(response.get("pid_namespace")) is not bool:
            raise ExecutorError("protocol", "the executor's hello is malformed")
        self._pid_namespace = response["pid_namespace"]
        return response

    def boot_id(self, supervisor_boot_id: str) -> str:
        """One boot ID that changes when either the supervisor or the executor restarts."""
        return str(uuid.uuid5(uuid.UUID(supervisor_boot_id), self.hello()["boot_id"]))

    def workspace_root(self, root: str | Path) -> Path:
        # The workspace is mounted only in the agent container, which checks it.
        if not os.path.isabs(root):
            raise ValueError("the workspace root must be an absolute path")
        return Path(os.path.normpath(root))

    def resolve_cwd(self, cwd: str, root: Path) -> str:
        del root
        try:
            resolved = self.client.call("resolve_cwd", cwd=cwd)["cwd"]
        except ExecutorError as exc:
            if exc.kind == "invalid_request":
                raise CommandRequestError(str(exc)) from exc
            raise
        if type(resolved) is not str:
            raise ExecutorError("protocol", "the executor returned no cwd")
        return resolved

    def process_tree(
        self,
        execution_boundary: CommandExecutionBoundary,
        *,
        execution_profile: ExecutionProfile,
        interactive: bool,
    ) -> str:
        del execution_boundary
        if not interactive or execution_profile != "agent":
            return "process_group"
        if self._pid_namespace is None:
            self.hello()
        return "pid_namespace" if self._pid_namespace else "process_group"

    def spawn(
        self, record: _CommandRecord, argv: list[str], *, cwd: str, env: dict[str, str]
    ) -> ProcessHandle:
        self.client.call(
            "spawn",
            command_id=record.command_id,
            argv=argv,
            env=env,
            cwd=cwd,
            process_tree=record.process_tree,
        )
        return RemoteProcess(self.client, record.command_id)

    def await_tree_init(self, record: _CommandRecord, process: ProcessHandle) -> None:
        """The executor finds the namespace init before it answers the spawn."""

    def stop_group(self, record: _CommandRecord, process: ProcessHandle) -> None:
        del process
        # Like a signal to a process that already exited, a lost executor stops nothing.
        with contextlib.suppress(ExecutorError):
            self.client.call("stop_group", command_id=record.command_id)

    def stop_tree(self, record: _CommandRecord, process: ProcessHandle) -> None:
        del process
        with contextlib.suppress(ExecutorError):
            self.client.call("stop_tree", command_id=record.command_id)

    def tree_stopped(self, record: _CommandRecord, *, timeout_s: float) -> bool:
        if record.process_tree != "pid_namespace":
            return False
        try:
            bounded = min(timeout_s, MAX_WAIT_SECONDS)
            response = self.client.call(
                "tree_stopped", command_id=record.command_id, timeout_s=bounded, wait_s=bounded
            )
        except ExecutorError:
            return False
        return response.get("stopped") is True

    def release(self, record: _CommandRecord) -> None:
        with contextlib.suppress(ExecutorError):
            self.client.call("release", command_id=record.command_id, reconnect_s=0)


class RemoteProcess:
    """A command the executor runs, with the ``Popen`` surface the supervisor uses."""

    def __init__(self, client: ExecutorClient, command_id: str) -> None:
        self._client = client
        self.command_id = command_id
        self.returncode: int | None = None
        self.stdin: Any = _RemoteStdin(client, command_id)
        self.stdout: Any = _RemoteOutput(client, command_id, "stdout")
        self.stderr: Any = _RemoteOutput(client, command_id, "stderr")

    def poll(self) -> int | None:
        if self.returncode is None:
            with contextlib.suppress(ExecutorError):
                self._wait_once(0)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.returncode is None:
            remaining = MAX_WAIT_SECONDS if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.command_id, cast("float", timeout))
            self._wait_once(min(remaining, MAX_WAIT_SECONDS))
        return self.returncode

    def _wait_once(self, wait_s: float) -> None:
        returncode = self._client.call("wait", command_id=self.command_id, wait_s=wait_s)[
            "returncode"
        ]
        if returncode is not None:
            if type(returncode) is not int:
                raise ExecutorError("protocol", "the executor returned a malformed returncode")
            self.returncode = returncode


class _RemoteOutput:
    """One output stream, read from the offset after every byte already read."""

    def __init__(self, client: ExecutorClient, command_id: str, stream: str) -> None:
        self._client = client
        self._command_id = command_id
        self._stream = stream
        self._offset = 0
        self._eof = False
        self.closed = False

    def read(self, size: int = MAX_CHUNK_BYTES) -> bytes:
        while not self._eof:
            if self.closed:
                raise ValueError("read of a closed executor stream")
            response = self._client.call(
                "read",
                command_id=self._command_id,
                stream=self._stream,
                offset=self._offset,
                max_bytes=max(1, min(size, MAX_CHUNK_BYTES)),
                wait_s=MAX_WAIT_SECONDS,
            )
            data = base64.b64decode(response["data_base64"], validate=True)
            self._offset += len(data)
            self._eof = response["eof"] is True
            if data:
                return data
        return b""

    read1 = read

    def close(self) -> None:
        self.closed = True


class _RemoteStdin:
    """Stdin written in chunks, each naming the offset it starts at.

    Each write waits until the command has taken the chunk, as a local pipe
    write does, so a slow reader holds the writer back instead of failing it.
    """

    def __init__(self, client: ExecutorClient, command_id: str) -> None:
        self._client = client
        self._command_id = command_id
        self._offset = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        for start in range(0, len(view), MAX_CHUNK_BYTES):
            chunk = bytes(view[start : start + MAX_CHUNK_BYTES])
            response = self._send(chunk, eof=False)
            if response["closed"] and response["offset"] < self._offset + len(chunk):
                raise BrokenPipeError("the command closed its stdin")
            self._offset = response["offset"]
        return len(data)

    def flush(self) -> None:
        """Every write is already delivered."""

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._send(b"", eof=True)

    def _send(self, chunk: bytes, *, eof: bool) -> dict[str, Any]:
        return self._client.call(
            "write",
            command_id=self._command_id,
            offset=self._offset,
            data_base64=base64.b64encode(chunk).decode("ascii"),
            eof=eof,
        )

"""Deterministic handlers used to verify Cayu's real isolated process boundary."""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.isolated_tools import ProcessIsolatedToolContext
from cayu.core.tools import ToolResult


class _DeterministicIsolatedToolHandler:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def run(
        self,
        context: ProcessIsolatedToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult | Any:
        mode = self._config.get("mode", "success")
        if mode == "success":
            return ToolResult(
                content=str(arguments.get("text", "ok")),
                structured={
                    "session_id": context.session_id,
                    "environment_marker": os.environ.get("CAYU_TEST_MARKER"),
                },
            )
        if mode == "counted_success":
            with Path(str(self._config["count_path"])).open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write("started\n")
            return ToolResult(content=str(arguments.get("text", "ok")))
        if mode == "secret_output":
            return ToolResult(content="ISOLATED_RESULT_SECRET_CANARY")
        if mode == "async_success":
            return self._async_success(arguments)
        if mode == "environment":
            return ToolResult(
                content="environment captured",
                structured={"environment": dict(sorted(os.environ.items()))},
            )
        if mode == "file_descriptors":
            opened: list[int] = []
            for descriptor in range(256):
                try:
                    os.fstat(descriptor)
                except OSError:
                    continue
                opened.append(descriptor)
            return ToolResult(
                content="file descriptors captured",
                structured={"file_descriptors": opened},
            )
        if mode == "conditional_gil_block":
            if arguments.get("text") == "block":
                self._publish_started()
                ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
                return ToolResult(content="native sleep completed")
            return ToolResult(content=str(arguments.get("text", "ok")))
        if mode == "exception":
            raise RuntimeError(str(self._config.get("message", "isolated failure")))
        if mode == "invalid_result":
            return {"content": "not a ToolResult"}
        if mode == "stdout":
            sys.stdout.write("x" * int(self._config.get("bytes", 1)))
            sys.stdout.flush()
            return ToolResult(content="stdout written")
        if mode == "stderr":
            sys.stderr.write("x" * int(self._config.get("bytes", 1)))
            sys.stderr.flush()
            return ToolResult(content="stderr written")
        if mode == "crash":
            os._exit(int(self._config.get("exit_code", 23)))
        if mode == "signal":
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL unexpectedly returned")
        if mode == "gil_block":
            seconds = int(self._config.get("seconds", 30))
            self._publish_started()
            ctypes.PyDLL(None).sleep(seconds)
            return ToolResult(content="native sleep completed")
        if mode == "counted_gil_block":
            with Path(str(self._config["count_path"])).open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write("started\n")
            self._publish_started()
            ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
            return ToolResult(content="native sleep completed")
        if mode == "ignore_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self._publish_started()
            ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
            return ToolResult(content="ignored termination")
        if mode == "grandchild":
            pid_path = Path(str(self._config["pid_path"]))
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ],
                close_fds=True,
            )
            pid_path.write_text(str(child.pid), encoding="utf-8")
            self._publish_started()
            ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
            return ToolResult(content="grandchild completed")
        if mode == "detached_descendant":
            pid_path = Path(str(self._config["pid_path"]))
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ],
                close_fds=True,
                start_new_session=True,
            )
            pid_path.write_text(str(child.pid), encoding="utf-8")
            if self._config.get("return_success") is True:
                return ToolResult(content="detached descendant started")
            self._publish_started()
            ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
            return ToolResult(content="detached descendant completed")
        if mode == "kill_supervisor":
            worker_pid_path = Path(str(self._config["worker_pid_path"]))
            descendant_pid_path = Path(str(self._config["pid_path"]))
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ],
                close_fds=True,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            worker_pid_path.write_text(str(os.getpid()), encoding="utf-8")
            descendant_pid_path.write_text(str(child.pid), encoding="utf-8")
            # This fixture tests missing supervisor settlement authority, not
            # diagnostic-pipe ownership.  Release the inherited pipe writers
            # so the parent can observe exact supervisor exit before its
            # bounded cleanup deadline.
            os.close(sys.stdout.fileno())
            os.close(sys.stderr.fileno())
            os.kill(os.getppid(), signal.SIGKILL)
            ctypes.PyDLL(None).sleep(int(self._config.get("seconds", 30)))
            raise AssertionError("supervisor SIGKILL unexpectedly settled the worker")
        if mode == "fork_then_success":
            child_pid = os.fork()
            if child_pid == 0:
                while True:
                    signal.pause()
            Path(str(self._config["pid_path"])).write_text(
                str(child_pid),
                encoding="utf-8",
            )
            return ToolResult(content="forked handler completed")
        raise ValueError("Unsupported deterministic isolated tool mode.")

    async def _async_success(self, arguments: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(0)
        return ToolResult(content=str(arguments.get("text", "async ok")))

    def _publish_started(self) -> None:
        started_path = self._config.get("started_path")
        if started_path is not None:
            Path(str(started_path)).write_text("started", encoding="utf-8")


def build_deterministic_isolated_tool(
    config: dict[str, Any],
) -> _DeterministicIsolatedToolHandler:
    """Return a deterministic handler reconstructed only inside the child."""

    return _DeterministicIsolatedToolHandler(config)


object.__setattr__(
    build_deterministic_isolated_tool,
    "execution_profile_identity",
    ExecutionProfileBehaviorIdentity(
        name="cayu:testing:deterministic-isolated-tool",
        behavior_version="1",
        implementation_version="1",
    ),
)


__all__ = ["build_deterministic_isolated_tool"]

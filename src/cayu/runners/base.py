from __future__ import annotations

import asyncio
import posixpath
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value, require_nonblank
from cayu.runners._cleanup import RunnerCleanupResult

DEFAULT_EXEC_OUTPUT_LIMIT_BYTES = 1024 * 1024


class RunnerUnavailableError(RuntimeError):
    """A runner cannot execute commands until it is reconnected or replaced."""

    def __init__(self, message: str, *, diagnostic: Mapping[str, Any]) -> None:
        copied = copy_json_value(dict(diagnostic), "diagnostic")
        self.diagnostic: dict[str, Any] = copied
        self.artifacts: list[dict[str, Any]] = [copy_json_value(self.diagnostic, "diagnostic")]
        super().__init__(require_nonblank(message, "message"))


class RunnerCancelledError(asyncio.CancelledError):
    """Cancelled runner execution with optional cleanup diagnostics.

    Retained for backward compatibility with third-party runners. Built-in
    runners no longer raise this subclass: they re-raise the original plain
    ``asyncio.CancelledError`` (preserving asyncio's cancellation bookkeeping)
    with diagnostics attached out-of-band via
    :func:`attach_cancellation_artifacts`. The runtime reads diagnostics from
    the exception's ``artifacts`` attribute either way.
    """

    def __init__(
        self,
        message: str = "Runner command was cancelled.",
        *,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifacts = copy_json_value([] if artifacts is None else artifacts, "artifacts")


def attach_cancellation_artifacts(
    exc: BaseException,
    artifacts: list[dict[str, Any]],
) -> None:
    """Attach runner cleanup diagnostics to a cancellation out-of-band.

    Substituting an exception subclass for the in-flight ``CancelledError``
    discards the exception instance asyncio saved for the awaiting task.
    Instead runners record diagnostics on the original exception's
    ``artifacts`` attribute and re-raise it unchanged; the runtime reads the
    attribute via ``getattr``.
    """

    copied = copy_json_value(artifacts, "artifacts")
    existing = getattr(exc, "artifacts", None)
    if isinstance(existing, list):
        existing.extend(copied)
        return
    exc.artifacts = copied  # type: ignore


def is_same_or_child(path: str, root: str) -> bool:
    """Return whether a normalized absolute POSIX path is ``root`` or inside it."""

    if root == "/":
        return posixpath.isabs(path)
    return path == root or path.startswith(f"{root.rstrip('/')}/")


class ExecCommand(BaseModel):
    """Command to execute.

    `argv` is the default safe process form. `shell` is reserved for explicit
    shell scripts where parsing, expansion, and quoting are intentional.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["process", "shell"] = "process"
    argv: list[str] | None = None
    shell: str | None = None

    @field_validator("argv")
    @classmethod
    def copy_argv(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(value)

    @classmethod
    def process(cls, *argv: str) -> ExecCommand:
        return cls(kind="process", argv=list(argv))

    @classmethod
    def bash(cls, script: str) -> ExecCommand:
        return cls(kind="shell", shell=script)

    @model_validator(mode="after")
    def validate_shape(self) -> ExecCommand:
        if self.kind == "process":
            if not self.argv:
                raise ValueError("Process commands require non-empty argv.")
            for item in self.argv:
                if type(item) is not str or not item.strip():
                    raise ValueError("Process argv entries must be non-empty strings.")
            if self.shell is not None:
                raise ValueError("Process commands cannot define shell script.")
        if self.kind == "shell":
            if self.shell is None:
                raise ValueError("Shell commands require a non-empty script.")
            if type(self.shell) is not str or not self.shell.strip():
                raise ValueError("Shell commands require a non-empty script.")
            if self.argv is not None:
                raise ValueError("Shell commands cannot define argv.")
        return self


class ExecResult(BaseModel):
    """Observed command outcome, including bounded capture and full byte totals."""

    model_config = ConfigDict(extra="forbid")

    stdout: str = ""
    stderr: str = ""
    exit_code: StrictInt = 0
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    stdout_truncated: StrictBool = False
    stderr_truncated: StrictBool = False
    stdout_bytes: StrictInt | None = Field(default=None, ge=0)
    stderr_bytes: StrictInt | None = Field(default=None, ge=0)
    artifacts: list[dict] = Field(default_factory=list)

    @field_validator("artifacts", mode="before")
    @classmethod
    def copy_artifacts(cls, value: list[dict]) -> list[dict]:
        return copy_json_value(value, "artifacts")


class Runner(ABC):
    """Executes commands/code in a workspace or sandbox.

    Shared lifecycle contract:

    - ``close()`` applies the adapter's configured lifecycle action once;
      further ``exec`` calls fail.
    - Interrupted commands (cancellation/timeout) run cleanup. When command
      cleanup cannot confirm the command stopped, the exec path latches shut
      (``_close_exec``) so an unknown still-running command cannot race new
      work.
    - ``reopen_exec()`` explicitly clears that latch after the caller verified
      out-of-band that no stale command is running.
    - ``close()`` is terminal for command execution, even for adapters whose
      configured close action intentionally leaves a remote sandbox alive.
    """

    isolation: str = "unknown"
    default_cwd: str = "/"

    _closed: bool = False
    _exec_closed: bool = False
    _exec_closed_reason: str | None = None

    @abstractmethod
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Execute a command and return stdout/stderr/exit metadata."""

    async def close(self) -> None:
        """Release the runner. The default implementation only marks it closed."""

        self._closed = True

    def reopen_exec(self) -> None:
        """Clear a latched exec-closed state on an otherwise-open runner.

        Cleanup after an interrupted command latches the exec path shut when it
        cannot confirm the command stopped (for example a flaky pid-file wait).
        After verifying out-of-band that no stale command is running, callers
        use this to resume executing instead of discarding the runner.
        """

        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed.")
        self._open_exec()

    def resolve_cwd(self, cwd: str | None = None) -> str:
        """Resolve a requested cwd to a canonical path inside the runner root.

        Relative requests are resolved against ``default_cwd``. An absolute
        input is accepted only when it is already contained by the runner root,
        making canonicalization idempotent for policy-authorized execution.
        """
        root = posixpath.normpath(self.default_cwd)
        if cwd is None:
            return root
        requested_cwd = require_nonblank(cwd, "cwd")
        if posixpath.isabs(requested_cwd):
            resolved = posixpath.normpath(requested_cwd)
            if not is_same_or_child(resolved, root):
                raise ValueError("Runner cwd is outside the runner root.")
            return resolved
        resolved = posixpath.normpath(posixpath.join(root, requested_cwd))
        if not is_same_or_child(resolved, root):
            raise ValueError("Runner cwd escapes the runner root.")
        return resolved

    def _ensure_exec_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed.")
        if self._exec_closed:
            reason = self._exec_closed_reason or "runner exec path is closed"
            raise RuntimeError(f"{type(self).__name__} is closed: {reason}")

    def _close_exec(self, reason: str) -> None:
        self._exec_closed = True
        self._exec_closed_reason = reason

    def _open_exec(self) -> None:
        self._exec_closed = False
        self._exec_closed_reason = None

    def _apply_cleanup_result(self, cleanup: RunnerCleanupResult) -> None:
        artifact = cleanup.artifact
        if cleanup.close_runner:
            self._close_exec("runner cleanup closed the exec path")
        if artifact.get("action") == "kill_sandbox" and artifact.get("status") == "completed":
            self._closed = True
            return
        if artifact.get("action") == "kill_command" and artifact.get("status") != "completed":
            self._close_exec(
                f"{self.isolation} command cleanup did not complete; command state is unknown"
            )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False

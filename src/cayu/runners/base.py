from __future__ import annotations

import asyncio
import posixpath
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast

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
from cayu.runners._diagnostics import (
    trusted_runner_error_type_name,
    trusted_runner_exception_type_name,
)

if TYPE_CHECKING:
    from cayu.environments.admission import ExecutionAdmissionCandidate
    from cayu.vaults import SecretRedactor

DEFAULT_EXEC_OUTPUT_LIMIT_BYTES = 1024 * 1024
RunnerSystemExecutionMode = Literal["shared", "separate"]


class RunnerWorkspaceCapability(ABC):
    """Narrow provider capability used by a first-party native workspace.

    Capability objects deliberately do not own runner lifecycle. A managed
    runner can therefore expose native filesystem behavior without exposing
    the raw provider runner or a second ``close()`` authority.
    """

    @property
    @abstractmethod
    def resource_key(self) -> tuple[object, ...]:
        """Stable identity of the sandbox that backs this capability."""


RunnerWorkspaceCapabilityT = TypeVar(
    "RunnerWorkspaceCapabilityT",
    bound=RunnerWorkspaceCapability,
)


class RunnerUnavailableError(RuntimeError):
    """A runner cannot execute commands until it is reconnected or replaced."""

    def __init__(self, message: str, *, diagnostic: Mapping[str, Any]) -> None:
        copied = copy_json_value(dict(diagnostic), "diagnostic")
        self.diagnostic: dict[str, Any] = copied
        self.artifacts: list[dict[str, Any]] = [copy_json_value(self.diagnostic, "diagnostic")]
        super().__init__(require_nonblank(message, "message"))


class RunnerExecutionError(RuntimeError):
    """A fixed-message command failure with typed, secret-safe evidence."""

    def __init__(self, *, diagnostic: dict[str, Any]) -> None:
        copied = _safe_runner_execution_diagnostic(diagnostic)
        self.diagnostic: dict[str, Any] = copied
        self.artifacts: list[dict[str, Any]] = [copy_json_value(copied, "diagnostic")]
        super().__init__("Runner command execution failed.")


def runner_execution_error(
    error: BaseException,
    *,
    adapter: str,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
) -> RunnerExecutionError:
    """Detach an opaque runner failure from its raw message and traceback."""

    if type(adapter) is not str or adapter not in {
        "docker",
        "e2b",
        "lambda-microvm",
        "local",
        "microsandbox",
    }:
        adapter = "unknown"
    error_type = trusted_runner_exception_type_name(error)
    source_diagnostic = _base_exception_namespace_value(error, "diagnostic")
    if type(source_diagnostic) is dict:
        source_diagnostic = cast("dict[str, Any]", source_diagnostic)
        source_type = trusted_runner_error_type_name(source_diagnostic.get("error_type"))
        if source_type is not None:
            error_type = source_type
        if stdout_bytes is None:
            candidate = source_diagnostic.get("stdout_bytes")
            if type(candidate) is int and candidate >= 0:
                stdout_bytes = candidate
        if stderr_bytes is None:
            candidate = source_diagnostic.get("stderr_bytes")
            if type(candidate) is int and candidate >= 0:
                stderr_bytes = candidate
    diagnostic: dict[str, Any] = {
        "type": "cayu.runner_execution_error.v1",
        "adapter": adapter,
        "status": "failed",
        "error_type": error_type,
        "timed_out": False,
        "cancelled": False,
    }
    if type(stdout_bytes) is int and stdout_bytes >= 0:
        diagnostic["stdout_bytes"] = stdout_bytes
    if type(stderr_bytes) is int and stderr_bytes >= 0:
        diagnostic["stderr_bytes"] = stderr_bytes
    return RunnerExecutionError(diagnostic=diagnostic)


def _safe_runner_execution_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    if type(diagnostic) is not dict:
        raise TypeError("Runner execution diagnostic must be a dict.")
    adapter = diagnostic.get("adapter")
    if type(adapter) is not str or adapter not in {
        "docker",
        "e2b",
        "lambda-microvm",
        "local",
        "microsandbox",
    }:
        adapter = "unknown"
    error_type = trusted_runner_error_type_name(diagnostic.get("error_type")) or "Exception"
    safe: dict[str, Any] = {
        "type": "cayu.runner_execution_error.v1",
        "adapter": adapter,
        "status": "failed",
        "error_type": error_type,
        "timed_out": diagnostic.get("timed_out") is True,
        "cancelled": diagnostic.get("cancelled") is True,
    }
    for field in ("stdout_bytes", "stderr_bytes"):
        value = diagnostic.get(field)
        if type(value) is int and value >= 0:
            safe[field] = value
    return safe


def _base_exception_namespace_value(error: BaseException, name: str) -> object:
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    except BaseException:
        return None
    return dict.get(namespace, name) if type(namespace) is dict else None


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

    - ``system_execution_mode`` declares whether ``exec_system()`` intentionally
      shares the ordinary command lane or selects a separate trusted lane.
      Runner wrappers must preserve both the declaration and dispatch.
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
    system_execution_mode: RunnerSystemExecutionMode = "shared"

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
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Execute a command and return stdout/stderr/exit metadata."""

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Execute with an invocation redactor at the closest supported boundary.

        Bundled runners override this to redact while capturing. The default
        preserves compatibility for custom runners but treats a channel that
        was already truncated as irrecoverably ambiguous and omits its text.
        """

        from cayu.runners._redacted_output import redact_completed_exec_result
        from cayu.vaults import SecretRedactor

        if not isinstance(redactor, SecretRedactor):
            raise TypeError("Runner.exec_redacted redactor must be a SecretRedactor.")
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "timeout_s": timeout_s,
            "stdin": stdin,
            "output_limit_bytes": output_limit_bytes,
        }
        if env_remove:
            kwargs["env_remove"] = env_remove
        result = await self.exec(command, **kwargs)
        return redact_completed_exec_result(
            result,
            redactor=redactor,
            output_limit_bytes=output_limit_bytes,
            omit_pretruncated=True,
        )

    async def exec_system(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Execute a control-plane lifecycle command on the declared system lane."""
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "timeout_s": timeout_s,
            "stdin": stdin,
            "output_limit_bytes": output_limit_bytes,
        }
        if env_remove:
            kwargs["env_remove"] = env_remove
        return await self.exec(command, **kwargs)

    async def close(self) -> None:
        """Release the runner. The default implementation only marks it closed."""

        self._closed = True

    async def await_pending_command_settlement(self) -> bool:
        """Wait for command cleanup deferred beyond an ``exec`` result.

        Return ``True`` only when every command-cleanup operation dispatched
        before this call has positively reached its terminal boundary. This hook
        is consulted only after a runner explicitly reports deferred cleanup.
        Such runners must override it; the default fails closed so an unknown
        extension cannot turn absence of evidence into quiescence.
        """

        return False

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        """Return explicit provider-neutral admission evidence, when implemented.

        The default deliberately makes no capability claim. Runtimes must fail
        closed when a workload requires evidence that its selected runner does
        not provide.
        """

        return None

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        """Stable identity of the runner-owned execution resource, when known."""

        return None

    @property
    def is_closed(self) -> bool:
        """Whether terminal runner finalization has completed."""

        return self._closed

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        """Return a narrow native-workspace capability, when supported.

        The returned object has no lifecycle methods. Callers must continue to
        finalize the owning runner or environment; they cannot close an
        unmanaged provider runner through this composition path.
        """

        if not isinstance(capability_type, type) or not issubclass(
            capability_type,
            RunnerWorkspaceCapability,
        ):
            raise TypeError(
                "Runner workspace capability type must derive from RunnerWorkspaceCapability."
            )
        return None

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

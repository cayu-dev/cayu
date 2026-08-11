from __future__ import annotations

import asyncio
import posixpath
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    LiteralString,
    NoReturn,
    ParamSpec,
    Self,
    TypeVar,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from cayu._validation import (
    DurableValueError,
    copy_json_value,
    extract_durable_value_error,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
    require_nonblank,
)
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

_COMMAND_VALIDATION_TITLES = frozenset({"CommandRequest", "ExecCommand"})
_COMMAND_VALIDATION_LOCATIONS = frozenset(
    {
        "argv",
        "canonical_cwd",
        "command",
        "cwd",
        "env",
        "kind",
        "shell",
        "stdin",
        "timeout_s",
    }
)
_SAFE_COMMAND_SHAPE_MESSAGES = frozenset(
    {
        "Process commands require non-empty argv.",
        "Process argv entries must be non-empty strings.",
        "Process commands cannot define shell script.",
        "Shell commands require a non-empty script.",
        "Shell commands cannot define argv.",
    }
)


def _safe_command_validation_failure(
    exc: ValidationError,
    *,
    title: str,
) -> ValidationError:
    """Return a fresh command error that cannot retain rejected input."""

    details = [_safe_command_error_detail(error) for error in exc.errors(include_input=False)]
    if not details:
        details = [_generic_command_error_detail(())]
    return ValidationError.from_exception_data(
        title if title in _COMMAND_VALIDATION_TITLES else "Runner command",
        details,
        hide_input=True,
    )


def _safe_command_error_detail(error: Mapping[str, Any]) -> InitErrorDetails:
    location = _safe_command_error_location(error.get("loc"))
    error_type = error.get("type")
    context = error.get("ctx")
    if error.get("msg") == "Runner command is invalid.":
        return _generic_command_error_detail(location, error_type=error_type)
    if error_type == "value_error" and type(context) is dict:
        safe_failure = _safe_command_value_failure(context.get("error"))
        return InitErrorDetails(
            type="value_error",
            loc=location,
            input=None,
            ctx={"error": safe_failure},
        )
    if type(error_type) is str and context is None:
        return InitErrorDetails(
            type=error_type,
            loc=location,
            input=None,
        )
    return _generic_command_error_detail(location, error_type=error_type)


def _safe_command_value_failure(value: object) -> ValueError:
    durable_failure = (
        extract_durable_value_error(value) if isinstance(value, BaseException) else None
    )
    if durable_failure is not None:
        return DurableValueError(
            durable_failure.code,
            durable_failure.field_name,
            path=durable_failure.path,
        )
    if type(value) is ValueError and len(value.args) == 1:
        message = value.args[0]
        if type(message) is str and message in _SAFE_COMMAND_SHAPE_MESSAGES:
            return ValueError(message)
    return ValueError("Runner command is invalid.")


def _safe_command_error_location(value: object) -> tuple[str | int, ...]:
    if type(value) not in {list, tuple}:
        return ()
    location: list[str | int] = []
    for item in cast("list[object] | tuple[object, ...]", value):
        if type(item) is int and item >= 0:
            location.append(item)
            continue
        if type(item) is str and item in _COMMAND_VALIDATION_LOCATIONS:
            location.append(item)
            continue
        location.append("invalid_input")
        break
    return tuple(location)


def _generic_command_error_detail(
    location: tuple[str | int, ...],
    *,
    error_type: object = "value_error",
) -> InitErrorDetails:
    raw_code = (
        error_type
        if type(error_type) is str
        and 0 < len(error_type) <= 128
        and all(
            character.isascii() and (character.isalnum() or character == "_")
            for character in error_type
        )
        else "value_error"
    )
    code = cast("LiteralString", raw_code)
    return InitErrorDetails(
        type=PydanticCustomError(code, "Runner command is invalid."),
        loc=location,
        input=None,
    )


def _raise_clean_command_validation_failure(error: ValidationError) -> NoReturn:
    """Raise a sanitized validation failure without retaining an old traceback."""

    error.__traceback__ = None
    raise error from None


def _clear_preflight_traceback_frames(error: BaseException) -> None:
    """Drop inactive validation-frame locals without changing the failure."""

    traceback.clear_frames(error.__traceback__)


_PreflightP = ParamSpec("_PreflightP")
_PreflightResultT = TypeVar("_PreflightResultT")


def _clean_runner_preflight(
    operation: Callable[_PreflightP, _PreflightResultT],
) -> Callable[_PreflightP, _PreflightResultT]:
    """Publish preflight failures without retaining rejected request locals."""

    @wraps(operation)
    def clean_preflight(
        *args: _PreflightP.args,
        **kwargs: _PreflightP.kwargs,
    ) -> _PreflightResultT:
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            published_error = error
            del args, kwargs, error
            published_error.__traceback__ = None
            raise published_error from None

    return clean_preflight


_CommandValidationResultT = TypeVar("_CommandValidationResultT")
_COMMAND_VALIDATION_MISSING = object()


def _capture_command_validation(
    operation: Callable[[], _CommandValidationResultT],
    *,
    title: str,
) -> tuple[_CommandValidationResultT | object, ValidationError | None]:
    """Run one Pydantic entrance and detach any rejected input from its error."""

    try:
        return operation(), None
    except ValidationError as exc:
        return (
            _COMMAND_VALIDATION_MISSING,
            _safe_command_validation_failure(exc, title=title),
        )


class _CommandValidationModel(BaseModel):
    """Pydantic command model that never exposes rejected input."""

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Any = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        validate = [super().model_validate]
        result, validation_failure = _capture_command_validation(
            lambda: validate[0](
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
            title=cls.__name__,
        )
        obj = None
        context = None
        validate.clear()
        if validation_failure is not None:
            _raise_clean_command_validation_failure(validation_failure)
        return cast("Self", result)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        validate = [super().model_validate_json]
        result, validation_failure = _capture_command_validation(
            lambda: validate[0](
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
            title=cls.__name__,
        )
        json_data = ""
        context = None
        validate.clear()
        if validation_failure is not None:
            _raise_clean_command_validation_failure(validation_failure)
        return cast("Self", result)

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        validate = [super().model_validate_strings]
        result, validation_failure = _capture_command_validation(
            lambda: validate[0](
                obj,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
            title=cls.__name__,
        )
        obj = None
        context = None
        validate.clear()
        if validation_failure is not None:
            _raise_clean_command_validation_failure(validation_failure)
        return cast("Self", result)

    def __init__(self, **data: Any) -> None:
        initialize = [super().__init__]
        _, validation_failure = _capture_command_validation(
            lambda: initialize[0](**data),
            title=type(self).__name__,
        )
        data.clear()
        initialize.clear()
        if validation_failure is not None:
            _raise_clean_command_validation_failure(validation_failure)


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


class ExecCommand(_CommandValidationModel):
    """Command to execute.

    `argv` is the default safe process form. `shell` is reserved for explicit
    shell scripts where parsing, expansion, and quoting are intentional.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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
        validation_failure: ValidationError | None = None
        try:
            return cls(kind="process", argv=list(argv))
        except ValidationError as exc:
            validation_failure = exc
        argv = ()
        _raise_clean_command_validation_failure(validation_failure)

    @classmethod
    def bash(cls, script: str) -> ExecCommand:
        validation_failure: ValidationError | None = None
        try:
            return cls(kind="shell", shell=script)
        except ValidationError as exc:
            validation_failure = exc
        script = ""
        _raise_clean_command_validation_failure(validation_failure)

    @model_validator(mode="after")
    def validate_shape(self) -> ExecCommand:
        if self.kind == "process":
            if not self.argv:
                raise ValueError("Process commands require non-empty argv.")
            for item in self.argv:
                if type(item) is not str or not item.strip():
                    raise ValueError("Process argv entries must be non-empty strings.")
                require_durable_text(item, "Process argv entry")
            if self.shell is not None:
                raise ValueError("Process commands cannot define shell script.")
        if self.kind == "shell":
            if self.shell is None:
                raise ValueError("Shell commands require a non-empty script.")
            if type(self.shell) is not str or not self.shell.strip():
                raise ValueError("Shell commands require a non-empty script.")
            require_durable_text(self.shell, "Shell command")
            if self.argv is not None:
                raise ValueError("Shell commands cannot define argv.")
        return self


def copy_exec_command(command: ExecCommand) -> ExecCommand:
    """Return a detached, revalidated exact ``ExecCommand`` snapshot."""

    if type(command) is not ExecCommand:
        raise TypeError("Runner command must be an ExecCommand.")
    validation_failure: ValidationError | None = None
    try:
        return ExecCommand.model_validate(command.model_dump(mode="python", warnings=False))
    except ValidationError as exc:
        validation_failure = exc
    del command
    _raise_clean_command_validation_failure(validation_failure)


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

    @_clean_runner_preflight
    def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate one complete request without lookup, dispatch, or mutation.

        Wrappers call this seam before consulting invocation-scoped secret state
        or admitting managed workspace work. Backends with stricter transport
        rules override it and must remain side-effect free.
        """

        from cayu.runners._subprocess import (
            copy_runner_env,
            validate_output_limit,
            validate_runner_env_remove,
            validate_stdin,
            validate_timeout,
        )

        self._ensure_exec_open()
        owned_command = copy_exec_command(command)
        owned_cwd = self.resolve_cwd(cwd)
        owned_env = copy_runner_env(env, inherit_env=False)
        owned_env_remove = validate_runner_env_remove(env_remove)
        owned_timeout = validate_timeout(timeout_s)
        owned_stdin = validate_stdin(stdin)
        owned_output_limit = validate_output_limit(output_limit_bytes)
        del (
            owned_command,
            owned_cwd,
            owned_env,
            owned_env_remove,
            owned_timeout,
            owned_stdin,
            owned_output_limit,
        )

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
        if type(env_remove) is not tuple or env_remove:
            kwargs["env_remove"] = env_remove
        operation = self.exec(command, **kwargs)
        del command, cwd, env, env_remove, timeout_s, stdin, kwargs
        try:
            result = await operation
        except BaseException:
            redactor = SecretRedactor()
            output_limit_bytes = None
            raise
        finally:
            del operation
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
        if type(env_remove) is not tuple or env_remove:
            kwargs["env_remove"] = env_remove
        operation = self.exec(command, **kwargs)
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes, kwargs
        try:
            return await operation
        finally:
            del operation

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
        root_value = require_durable_clean_nonblank(self.default_cwd, "default_cwd")
        if not posixpath.isabs(root_value):
            raise ValueError("Runner default_cwd must be an absolute path.")
        root = posixpath.normpath(root_value)
        if cwd is None:
            return root
        requested_cwd = require_durable_nonblank(cwd, "cwd")
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

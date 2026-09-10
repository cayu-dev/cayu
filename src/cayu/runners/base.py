from __future__ import annotations

import asyncio
import posixpath
import secrets
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
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

from cayu._exception_groups import iter_exception_tree
from cayu._exception_state import exception_state
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    DurableValueError,
    copy_json_value,
    extract_durable_value_error,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
    require_nonblank,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runners._cleanup import (
    RunnerCleanupPolicy,
    RunnerCleanupProgress,
    RunnerCleanupResult,
    RunnerFailureProgress,
    attach_runner_cancellation_failure,
    runner_cancellation_failure,
)
from cayu.runners._diagnostics import (
    runner_failure_fields,
    safe_runner_failure_fields,
    trusted_runner_error_type_name,
    trusted_runner_exception_type_name,
)

if TYPE_CHECKING:
    from cayu.environments.admission import (
        ExecutionAdmissionCandidate,
        ExecutionEnvironmentAuthority,
        ExecutionRequirements,
    )
    from cayu.vaults import SecretRedactor

DEFAULT_EXEC_OUTPUT_LIMIT_BYTES = 1024 * 1024
RunnerSystemExecutionMode = Literal["shared", "separate"]
RunnerLifecycleState = Literal["reusable", "fenced", "poisoned", "closing", "closed"]
# Keep cancellation-opaque cleanup alive after its bounded observer returns.
# A late result never grants reuse to an instance already poisoned by timeout.
_PENDING_RUNNER_CLEANUPS: set[asyncio.Task[Any]] = set()


def _contains_runner_fatal_signal(error: BaseException) -> bool:
    return any(
        not isinstance(leaf, (Exception, asyncio.CancelledError, BaseExceptionGroup))
        for leaf in iter_exception_tree(error)
    )


@dataclass(frozen=True, slots=True)
class RunnerWorkloadAuthority:
    """Runner-owned identity for one provisioned image workload.

    Higher layers may compare this value with an exact workload they support,
    but runners never need to import those higher-layer tools to declare what
    is installed in their selected image.
    """

    name: str
    image: str
    command: tuple[str, ...]
    protocol_version: str
    worker_version: str
    component_versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        fields = {
            "name": self.name,
            "image": self.image,
            "protocol_version": self.protocol_version,
            "worker_version": self.worker_version,
        }
        for field_name, value in fields.items():
            owned = require_durable_clean_nonblank(value, field_name)
            if len(owned.encode("utf-8")) > 512:
                raise ValueError(f"{field_name} must not exceed 512 bytes.")
            object.__setattr__(self, field_name, owned)
        if type(self.command) is not tuple or not self.command or len(self.command) > 32:
            raise ValueError("command must contain between 1 and 32 entries.")
        command = tuple(
            require_durable_clean_nonblank(entry, f"command[{index}]")
            for index, entry in enumerate(self.command)
        )
        if any(len(entry.encode("utf-8")) > 1024 for entry in command):
            raise ValueError("command entries must not exceed 1024 bytes.")
        if type(self.component_versions) is not tuple or len(self.component_versions) > 16:
            raise ValueError("component_versions must be a tuple with at most 16 entries.")
        components: list[tuple[str, str]] = []
        for index, entry in enumerate(self.component_versions):
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(f"component_versions[{index}] must be a name/version pair.")
            component_name = require_durable_clean_nonblank(
                entry[0], f"component_versions[{index}].name"
            )
            component_version = require_durable_clean_nonblank(
                entry[1], f"component_versions[{index}].version"
            )
            if any(
                len(value.encode("utf-8")) > 128 for value in (component_name, component_version)
            ):
                raise ValueError("component names and versions must not exceed 128 bytes.")
            components.append((component_name, component_version))
        if len({name for name, _ in components}) != len(components):
            raise ValueError("component_versions must not contain duplicate names.")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "component_versions", tuple(components))


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


class RunnerBinaryStreamCapability(ABC):
    """Nominal runner capability for bounded binary standard-I/O transfers.

    Binary streams are control-plane transport inputs and outputs. Implementations
    consume from and write to the caller-owned objects without closing them and
    must settle every delegated read, write, and child process before returning,
    including after cancellation or timeout. Paths are deliberately not accepted:
    remote adapters receive bytes through their transport, never host filenames.
    """

    @abstractmethod
    async def exec_stream(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stdout_limit_bytes: int | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Execute with optional binary stdin/stdout streams.

        ``stdout_limit_bytes`` is a hard capture limit for the binary channel.
        Bytes beyond it are drained but not written and set ``stdout_truncated``.
        Text stderr remains redacted and bounded by ``output_limit_bytes``.
        """


class RemoteWorkspaceBranchCapability(RunnerWorkspaceCapability):
    """Explicit provider proof for the durable guest branch protocol.

    Merely being able to execute Python or reconnect to a runner is not this
    capability. An exact runner adapter opts in only after verifying that its
    allocation filesystem retains branch state across Cayu process loss and
    supports the guest-side cooperative publication guard.
    """

    protocol_version = "cayu-runner-workspace-branch-v1"

    @property
    @abstractmethod
    def allocation_fingerprint(self) -> str:
        """Privacy-safe stable identity for the retained remote allocation."""


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
    failure_fields = runner_failure_fields(error)
    source_diagnostic = _base_exception_namespace_value(error, "diagnostic")
    if type(source_diagnostic) is dict:
        source_diagnostic = {
            key: value for key, value in dict.items(source_diagnostic) if type(key) is str
        }
        if type(error) is RunnerExecutionError:
            failure_fields = safe_runner_failure_fields(
                source_diagnostic.get("errno"), source_diagnostic.get("execution_phase")
            )
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
        **failure_fields,
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
    diagnostic = {key: value for key, value in dict.items(diagnostic) if type(key) is str}
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
        **safe_runner_failure_fields(diagnostic.get("errno"), diagnostic.get("execution_phase")),
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
    if type(namespace) is dict:
        for key, value in dict.items(namespace):
            if type(key) is str and key == name:
                return value
    return None


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


RunnerWorkspaceMutationSettlement = Literal[
    "complete",
    "runner_quiescent",
    "deferred",
    "uncertain",
]


def runner_workspace_mutation_settlement(
    *,
    result: ExecResult | None,
    error: BaseException | None,
) -> RunnerWorkspaceMutationSettlement:
    """Classify whether a returned command can still mutate its workspace."""

    completed_result = False
    raw_artifacts: object
    if result is not None:
        if type(result) is not ExecResult:
            return "uncertain"
        if type(result.timed_out) is not bool or type(result.cancelled) is not bool:
            return "uncertain"
        completed_result = not result.timed_out and not result.cancelled
        raw_artifacts = result.artifacts
    elif error is not None:
        raw_artifacts = exception_state(error, "artifacts")
    else:
        return "uncertain"
    if type(raw_artifacts) is not list:
        return "uncertain"
    cleanup_results: list[tuple[str, str]] = []
    for artifact in list(raw_artifacts):
        if type(artifact) is not dict:
            continue
        artifact_type: object | None = None
        action: object | None = None
        status: object | None = None
        artifact_type_present = False
        for key, value in dict.items(artifact):
            if type(key) is not str:
                continue
            if key == "type":
                artifact_type_present = True
                artifact_type = value
            elif key == "action":
                action = value
            elif key == "status":
                status = value
        if not artifact_type_present:
            continue
        if type(artifact_type) is not str:
            return "uncertain"
        if artifact_type != "cayu.runner_cleanup.v1":
            continue
        if type(action) is str and action == "close_transports":
            if type(status) is not str or status not in {
                "completed",
                "deferred",
                "failed",
                "timeout",
                "unsupported",
                "skipped",
            }:
                return "uncertain"
            # Transport closure is neither proof nor disproof of guest quiescence.
            continue
        if (
            type(action) is not str
            or action not in {"kill_command", "kill_sandbox"}
            or type(status) is not str
        ):
            return "uncertain"
        cleanup_results.append((action, status))
    if not cleanup_results:
        return "complete" if completed_result else "uncertain"
    if cleanup_results and all(status == "completed" for _, status in cleanup_results):
        if any(action == "kill_sandbox" for action, _ in cleanup_results):
            return "runner_quiescent"
        return "complete"
    if cleanup_results and all(
        status in {"completed", "deferred"} for _, status in cleanup_results
    ):
        return "deferred"
    return "uncertain"


@dataclass(frozen=True, slots=True, repr=False)
class RunnerExecutionAdmissionObserver:
    """One exact runner/requirement observation lifetime, never admission authority.

    Extensions may retain bounded evidence in a request-local state object.
    Every dispatched observation must settle or transfer authenticated cleanup
    ownership before returning or raising, including cancellation and timeout.
    """

    runner: Runner
    requirements: ExecutionRequirements

    def __post_init__(self) -> None:
        from cayu.environments.admission import ExecutionRequirements

        if not isinstance(self.runner, Runner):
            raise TypeError("Admission observer requires a Runner.")
        owned = ExecutionRequirements.model_validate(
            self.requirements.model_dump(mode="python", warnings=False)
        )
        object.__setattr__(self, "requirements", owned)

    def snapshot(self) -> ExecutionAdmissionCandidate | None:
        """Read exact evidence without dispatching an external operation."""
        return self.runner.execution_admission_candidate_for(self.requirements)

    async def collect(self) -> ExecutionAdmissionCandidate | None:
        """Observe after binding/setup has selected the exact final runner."""
        return await self.runner.collect_execution_admission_candidate_for(self.requirements)

    async def refresh(self) -> None:
        """Renew evidence without replacing this observer's runner or requirements."""
        await self.runner.refresh_execution_admission()

    def __repr__(self) -> str:
        return "RunnerExecutionAdmissionObserver(<request-scoped>)"

    def __reduce_ex__(self, protocol):
        raise TypeError("Live admission observers cannot be serialized.")


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
    - Inconclusive remote command cleanup permanently poisons execution on
      that runner instance. ``reopen_exec()`` clears only an intentional fence,
      not uncertainty from failed cleanup.
    - ``close()`` is terminal for command execution, even for adapters whose
      configured close action intentionally leaves a remote sandbox alive.
    """

    pending_command_settlement_cancellation_safe: ClassVar[bool] = False
    """Whether the deferred-settlement waiter can be cancelled as an observer.

    Cancellation of this waiter never proves that the underlying command has
    stopped. The flag only permits Cayu to run the waiter in a caller-owned
    event loop and cancel that observation during loop shutdown while keeping
    the environment mutation fence closed for a later fresh probe.
    """

    isolation: str = "unknown"
    default_cwd: str = "/"
    system_execution_mode: RunnerSystemExecutionMode = "shared"
    _environment_authority_lock: ClassVar[threading.Lock] = threading.Lock()

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        """Return runner-owned authority for an exact provisioned workload."""

        del name
        return None

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
        """Return the exact environment boundary that owns this runner."""

        from cayu.environments.admission import ExecutionEnvironmentAuthority

        with Runner._environment_authority_lock:
            authority = vars(self).get("_cayu_execution_environment_authority")
            if type(authority) is not ExecutionEnvironmentAuthority:
                authority = ExecutionEnvironmentAuthority(
                    identity=f"runner_{secrets.token_hex(24)}"
                )
                vars(self)["_cayu_execution_environment_authority"] = authority
        return authority

    def output_secret_values_present(self) -> bool | None:
        """Report whether command output can contain runner-owned secret values.

        ``None`` is fail-closed unknown authority. Wrappers must merge their own
        secret registry with the wrapped runner's declaration.
        """

        return None

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return a stable application declaration, or ``None`` when non-portable."""

        return None

    _closed: bool = False
    _exec_closed: bool = False
    _exec_closed_reason: str | None = None
    _exec_poisoned: bool = False
    _command_cleanups_pending: int = 0
    _closing: bool = False
    _terminal_lifecycle_task: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None
    _terminal_failure_progress: RunnerFailureProgress | None = None
    _terminal_lifecycle_action: str | None = None

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

    async def refresh_execution_admission(self) -> None:
        """Re-probe live admission evidence when supported by this runner.

        The default makes no new claim. A supporting implementation may renew
        only the exact runner represented by its current admission candidate;
        it must not replace the environment, immutable image, or toolchain
        identity. It must also settle every dispatched probe, or transfer an
        authenticated settlement owner, before returning or raising. Callers
        must re-read and validate the complete candidate after this hook;
        requesting renewal never authorizes execution by itself.
        """

        return None

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        """Return explicit provider-neutral admission evidence, when implemented.

        The default deliberately makes no capability claim. Runtimes must fail
        closed when a workload requires evidence that its selected runner does
        not provide.
        """

        return None

    async def collect_execution_admission_candidate(
        self,
    ) -> ExecutionAdmissionCandidate | None:
        """Collect final evidence after lifecycle-owned mutating setup completes.

        The default adapts existing side-effect-free snapshot implementations.
        Runners whose evidence depends on live external state override this hook
        and complete or positively transfer settlement ownership for every
        dispatched probe before returning or raising for any reason.
        """

        return self.execution_admission_candidate()

    def execution_admission_candidate_for(
        self,
        requirements: ExecutionRequirements,
    ) -> ExecutionAdmissionCandidate | None:
        """Read evidence for an explicit, caller-owned requirement set.

        This snapshot must not dispatch work or mutate a shared probe plan.
        The default reads the runner's existing candidate and makes no new
        claim from the supplied requirements alone.
        """
        return self.execution_admission_candidate()

    def execution_admission_observer(
        self,
        requirements: ExecutionRequirements,
    ) -> RunnerExecutionAdmissionObserver:
        """Create a request-scoped evidence owner for this exact runner.

        Creation is side-effect-free. The lifecycle owns collection, renewal,
        and any transferred settlement tasks; the observer cannot admit work.
        """
        return RunnerExecutionAdmissionObserver(self, requirements)

    async def collect_execution_admission_candidate_for(
        self,
        requirements: ExecutionRequirements,
    ) -> ExecutionAdmissionCandidate | None:
        """Collect final request-bound evidence under lifecycle ownership.

        Requirements are input, never proof. Implementations must isolate
        concurrent requirement sets and settle or positively transfer every
        dispatched probe under the same contract as the ordinary collector.
        """
        return await self.collect_execution_admission_candidate()

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        """Stable identity of the runner-owned execution resource, when known."""

        return None

    @property
    def is_closed(self) -> bool:
        """Whether terminal runner finalization has completed."""

        return self._closed

    @property
    def lifecycle_state(self) -> RunnerLifecycleState:
        """Observe execution eligibility without exposing provider internals."""

        if self._closed:
            return "closed"
        if self._exec_poisoned:
            return "poisoned"
        if self._closing or self._command_cleanups_pending:
            return "closing"
        return "fenced" if self._exec_closed else "reusable"

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
        """Clear an intentional execution fence on an otherwise-open runner.

        This is not recovery authority for inconclusive command cleanup.
        Poisoned instances cannot be reopened, including after a late cleanup
        acknowledgement; a new independently verified allocation owner is needed.
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
        if self._closing or self._command_cleanups_pending:
            raise RuntimeError(f"{type(self).__name__} is closed for cleanup.")
        if self._exec_closed:
            reason = self._exec_closed_reason or "runner exec path is closed"
            raise RuntimeError(f"{type(self).__name__} is closed: {reason}")

    def _close_exec(self, reason: str) -> None:
        self._exec_closed = True
        self._exec_closed_reason = reason

    def _open_exec(self) -> None:
        if self._exec_poisoned:
            raise RuntimeError(f"{type(self).__name__} is permanently poisoned.")
        if self._closing or self._command_cleanups_pending:
            raise RuntimeError(f"{type(self).__name__} is closed for cleanup.")
        self._exec_closed = False
        self._exec_closed_reason = None

    def _poison_exec(self, reason: str | None = None) -> None:
        self._exec_poisoned = True
        self._close_exec(
            reason or f"{self.isolation} command cleanup did not complete; command state is unknown"
        )

    async def _cleanup_failed_execution(
        self, failure: RunnerExecutionError, cleanup: Callable[[], Awaitable[RunnerCleanupResult]]
    ) -> RunnerCleanupResult:
        """Keep the sanitized execution failure authoritative through owned cleanup."""
        try:
            result = await cleanup()
        except asyncio.CancelledError as cancellation:
            cleanup_failure = runner_cancellation_failure(cancellation)
            attach_runner_cancellation_failure(
                cancellation,
                failure
                if cleanup_failure is None
                else BaseExceptionGroup(
                    "Runner execution and cleanup failed.", [failure, cleanup_failure]
                ),
            )
            raise
        except BaseException as cleanup_failure:
            raise BaseExceptionGroup(
                "Runner execution and cleanup failed.", [failure, cleanup_failure]
            ) from None
        failure.artifacts.extend(result.artifacts)
        if result.failure is not None:
            raise failure from result.failure
        return result

    async def _settle_command_cleanup(
        self,
        operation: Callable[[], Awaitable[RunnerCleanupResult]],
        *,
        adapter: str,
        timeout_s: float,
        policy: RunnerCleanupPolicy = "command",
        cancellation: asyncio.CancelledError | None = None,
        progress: RunnerCleanupProgress | None = None,
    ) -> RunnerCleanupResult:
        """Fence command reuse until owned cleanup has a positive outcome.

        The timeout bounds observation, not external mutation. A timed-out child
        remains retained and the runner stays poisoned even if it settles later.
        """

        self._command_cleanups_pending += 1
        try:
            task = asyncio.create_task(capture_awaitable_outcome(operation))
            _PENDING_RUNNER_CLEANUPS.add(task)
            task.add_done_callback(_PENDING_RUNNER_CLEANUPS.discard)
            outcome = await await_shielded_task_outcome(
                task, timeout_s=timeout_s, cancellation=cancellation
            )
            captured = outcome.result
            failure = outcome.error if captured is None else captured.error
            result = None if captured is None else captured.result
            if result is not None and result.failure is not None:
                failure = result.failure
                self._poison_exec()
            if outcome.timed_out or result is None:
                self._poison_exec()
                pending = None if progress is None else progress.pending
                result = RunnerCleanupResult(
                    artifact={
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": adapter,
                        "action": {
                            "command": "kill_command",
                            "sandbox": "kill_sandbox",
                            "none": "none",
                        }[policy],
                        "status": "timeout" if outcome.timed_out else "failed",
                        "timeout_s": timeout_s,
                    },
                    close_runner=True,
                )
                if pending is not None:
                    result = RunnerCleanupResult(
                        artifact={**pending.artifact, "status": result.artifact["status"]},
                        close_runner=True,
                        preceding_artifacts=pending.preceding_artifacts,
                        failure=pending.failure,
                    )
                    if failure is None:
                        failure = pending.failure
            self._apply_cleanup_result(result)
        except BaseException:
            self._poison_exec()
            raise
        finally:
            self._command_cleanups_pending -= 1
            self._command_cleanup_settled()
        if failure is not None and _contains_runner_fatal_signal(failure):
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            raise failure
        if outcome.cancellation is not None:
            attach_cancellation_artifacts(outcome.cancellation, result.artifacts)
            if failure is not None:
                attach_runner_cancellation_failure(outcome.cancellation, failure)
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            raise outcome.cancellation
        return result

    def _command_cleanup_settled(self) -> None:
        """Let an adapter reconcile its owned deferred fence after settlement."""

        return None

    async def _settle_terminal_lifecycle(
        self,
        operation: Callable[[], Awaitable[None]],
        *,
        action: str,
        timeout_s: float | None,
        progress: RunnerFailureProgress | None = None,
    ) -> None:
        """Single-flight terminal cleanup; interruption never abandons ownership."""

        if self._closed:
            return
        task = self._terminal_lifecycle_task
        if task is not None and self._terminal_lifecycle_action != action:
            raise RuntimeError("A different runner finalization is already in progress.")
        if task is None:
            self._closing = True
            self._close_exec("runner finalization has started")
            try:
                task = asyncio.create_task(capture_awaitable_outcome(operation))
            except BaseException:
                self._closing = False
                self._poison_exec()
                raise
            self._terminal_lifecycle_task = task
            self._terminal_failure_progress = progress
            self._terminal_lifecycle_action = action
            _PENDING_RUNNER_CLEANUPS.add(task)
            task.add_done_callback(self._record_terminal_lifecycle)
        owned_progress = getattr(self, "_terminal_failure_progress", None)
        outcome = await await_shielded_task_outcome(task, timeout_s=timeout_s)
        captured = outcome.result
        failure = outcome.error if captured is None else captured.error
        if outcome.timed_out:
            self._poison_exec()
            failure = (owned_progress or RunnerFailureProgress()).with_timeout(
                "Runner finalization has not settled within its deadline."
            )
        elif captured is not None:
            self._record_terminal_lifecycle(task)
        cancellation = outcome.cancellation
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=cancellation
        )
        if failure is not None and _contains_runner_fatal_signal(failure):
            raise failure
        if cancellation is not None:
            if failure is not None:
                attach_runner_cancellation_failure(cancellation, failure)
                raise cancellation from failure
            raise cancellation
        if failure is not None:
            if isinstance(failure, asyncio.CancelledError):
                raise RuntimeError("Runner finalization was cancelled by its dependency.") from None
            raise failure

    def _record_terminal_lifecycle(
        self, task: asyncio.Task[CapturedAwaitableOutcome[None]]
    ) -> None:
        _PENDING_RUNNER_CLEANUPS.discard(task)
        if task is not self._terminal_lifecycle_task:
            return
        self._closing = False
        try:
            failure = task.result().error
        except BaseException as error:
            failure = error
        if failure is None:
            self._closed = True
        else:
            self._poison_exec()
        # Failed, definitely settled cleanup can be retried; an in-flight
        # attempt stays attached, so retry cannot dispatch overlapping effects.
        self._terminal_lifecycle_task = None
        self._terminal_failure_progress = None
        self._terminal_lifecycle_action = None

    def _apply_cleanup_result(self, cleanup: RunnerCleanupResult) -> None:
        artifact = cleanup.artifact
        if cleanup.close_runner:
            self._close_exec("runner cleanup closed the exec path")
        if artifact.get("action") == "kill_sandbox" and artifact.get("status") == "completed":
            self._closed = True
            return
        if artifact.get("action") == "kill_command" and artifact.get("status") not in {
            "completed",
            "deferred",
        }:
            self._poison_exec()
        if artifact.get("action") == "kill_sandbox" and artifact.get("status") != "completed":
            self._poison_exec()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False


def runner_pending_command_settlement_cancellation_safe(runner: Runner) -> bool:
    """Read one exact class-level settlement observer safety declaration."""

    try:
        runner_type = type(runner)
        namespace = type.__getattribute__(runner_type, "__dict__")
        return namespace.get("pending_command_settlement_cancellation_safe") is True
    except BaseException:
        return False

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import StrEnum
from inspect import isawaitable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import (
    _COMMAND_POLICY_DENIAL_SOURCE,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _bound_policy_denial_result,
)
from cayu.runners import ExecCommand, ExecResult, RunnerUnavailableError
from cayu.runners._cleanup import runner_cancellation_failure
from cayu.runners.base import (
    _clear_preflight_traceback_frames,
    _CommandValidationModel,
    runner_execution_error,
)
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.tools._operation_boundary import (
    await_invocation_cancellation_checkpoint,
    await_invocation_operation,
)
from cayu.tools._runner import InvocationRunnerHandle

DEFAULT_OUTPUT_LIMIT_BYTES = 50_000
MAX_OUTPUT_LIMIT_BYTES = 200_000
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600


class CommandPolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    # Refuses the command inline from ExecCommandTool's own seam (an error ToolResult). This does NOT
    # create a durable approval checkpoint — for real pause/resume HITL use
    # `ToolPolicyDecision.REQUIRE_APPROVAL`. Named/valued distinctly so neither surface collides.
    REQUIRE_COMMAND_APPROVAL = "require_command_approval"


class CommandRequest(_CommandValidationModel):
    """Exec request evaluated by a :class:`CommandPolicy`.

    ``cwd`` preserves the original requested value for audit and backward
    compatibility. ``canonical_cwd`` is the active runner's resolved working
    directory and is the value execution will receive. The canonical field is
    optional only so policy metadata serialized before it was introduced can
    still be loaded; :class:`ExecCommandTool` always supplies it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    command: ExecCommand
    cwd: str | None = None
    canonical_cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_s: int
    stdin: str | None = None

    @field_validator("command")
    @classmethod
    def copy_command(cls, value: ExecCommand) -> ExecCommand:
        if type(value) is not ExecCommand:
            raise TypeError("CommandRequest.command must be an ExecCommand.")
        return ExecCommand.model_validate(value.model_dump(mode="python"))


class CommandPolicyResult(BaseModel):
    """Authorization decision for one exec_command invocation."""

    model_config = ConfigDict(extra="forbid")

    decision: CommandPolicyDecision
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class CommandPolicy(ABC):
    """Authorizes exec_command requests before they reach the runner.

    With a policy, ``ExecCommandTool`` resolves the requested ``cwd`` through
    the active runner and runs the selected backend's side-effect-free preflight
    before evaluation, then supplies both the requested and canonical forms. An
    allowed command executes with that same canonical value. Without a policy
    the model-controlled command, ``cwd``, and ``env`` pass straight to the
    runner. Hosts that need an allow/deny/approval seam attach a policy via
    ``ExecCommandTool(policy=...)``.
    """

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return a stable application declaration, or ``None`` when non-portable."""

        return None

    @abstractmethod
    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        """Return whether this command may execute."""


class ExecCommandTool(Tool):
    spec = ToolSpec(
        name="exec_command",
        # Runs commands with side effects; never overlaps other tools in a round.
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        workspace_mutation=True,
        description="Execute a process or explicit shell command with the active runner.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["process", "shell"],
                },
                "argv": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"\S",
                    },
                    "minItems": 1,
                },
                "shell": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r"\S",
                },
                "cwd": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "timeout_s": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS,
                },
                "stdin": {"type": "string"},
                "max_output_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_OUTPUT_LIMIT_BYTES,
                    "default": DEFAULT_OUTPUT_LIMIT_BYTES,
                },
            },
            "description": (
                "Use kind='process' with argv, or kind='shell' with shell. "
                "Do not provide both argv and shell. When kind is omitted it "
                "is inferred from whichever of argv or shell is provided."
            ),
        },
    )

    def _execution_profile_material(self) -> dict[str, object]:
        """Return Cayu-owned behavior inputs; command policy is profiled separately."""

        return {"default_timeout_seconds": self._default_timeout_seconds}

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        policy: CommandPolicy | None = None,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(default_timeout_seconds, bool)
            or not isinstance(default_timeout_seconds, int)
            or not 1 <= default_timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"default_timeout_seconds must be an integer between 1 and {MAX_TIMEOUT_SECONDS}"
            )
        base_spec = type(self).spec if spec is None else spec
        schema = base_spec.input_schema
        schema["properties"]["timeout_s"]["default"] = default_timeout_seconds
        super().__init__(base_spec.model_copy(update={"input_schema": schema}))
        if policy is not None and not isinstance(policy, CommandPolicy):
            raise TypeError("ExecCommandTool policy must implement CommandPolicy.")
        self._policy = policy
        self._default_timeout_seconds = default_timeout_seconds

    @property
    def command_policy(self) -> CommandPolicy | None:
        """Return the enforcing command policy, if one is configured."""

        return self._policy

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        runner = _require_runner(ctx)
        if runner is None:
            return ToolResult(
                content="No runner configured for this tool call.",
                is_error=True,
            )
        with tool_argument_validation():
            max_output_bytes = _optional_limited_int(
                args,
                "max_output_bytes",
                default=DEFAULT_OUTPUT_LIMIT_BYTES,
                maximum=MAX_OUTPUT_LIMIT_BYTES,
            )
            timeout_s = _optional_limited_int(
                args,
                "timeout_s",
                default=self._default_timeout_seconds,
                maximum=MAX_TIMEOUT_SECONDS,
            )
            command = _command_from_args(args)
            cwd = _optional_string(args, "cwd")
            if cwd is not None:
                _validate_command_text(cwd, "cwd")
            env = _optional_env(args)
            stdin = _optional_string(args, "stdin", allow_blank=True)
            if stdin is not None:
                require_unicode_scalar_text(stdin, "stdin")
            canonical_cwd = runner.resolve_cwd(cwd)
        if self._policy is not None:
            preflight_runner = _require_runner_preflight(runner)
            preflight_failure: Exception | None = None
            try:
                await _run_command_preflight(
                    preflight_runner,
                    command,
                    cwd=canonical_cwd,
                    env=env,
                    timeout_s=timeout_s,
                    stdin=stdin,
                    output_limit_bytes=max_output_bytes,
                )
            except Exception as exc:
                preflight_failure = exc
            if preflight_failure is not None:
                caller_cancellation = await await_invocation_cancellation_checkpoint()
                if caller_cancellation is not None:
                    raise caller_cancellation from preflight_failure
                if isinstance(preflight_failure, RunnerUnavailableError):
                    return _runner_unavailable_result(preflight_failure)
                raise preflight_failure from None
            verdict = await self._policy.evaluate(
                ctx,
                CommandRequest(
                    command=command,
                    cwd=cwd,
                    canonical_cwd=canonical_cwd,
                    env=env,
                    timeout_s=timeout_s,
                    stdin=stdin,
                ),
            )
            if type(verdict) is not CommandPolicyResult:
                raise TypeError("Command policy must return a CommandPolicyResult.")
            if verdict.decision is not CommandPolicyDecision.ALLOW:
                return _policy_refusal_result(verdict, ctx=ctx, source=self)
            cwd = canonical_cwd
        try:
            result = await runner.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=max_output_bytes,
            )
        except RunnerUnavailableError as exc:
            return _runner_unavailable_result(exc)
        result = _require_exec_result(result)
        content = _command_content(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            timeout_s=timeout_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )
        return ToolResult(
            content=content,
            structured={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
                "artifacts": copy_json_value(result.artifacts, "artifacts"),
            },
            artifacts=result.artifacts,
            is_error=result.timed_out or result.cancelled,
        )


def _policy_refusal_result(
    verdict: CommandPolicyResult,
    *,
    ctx: ToolContext,
    source: object,
) -> ToolResult:
    if verdict.decision is CommandPolicyDecision.DENY:
        content = "Command denied by policy."
        error = "command_denied"
    else:
        content = "Command requires approval before it can run."
        error = "command_approval_required"
    reason = verdict.reason or content
    if verdict.reason is not None:
        content = f"{content} {verdict.reason}"
    raw_result = ToolResult(
        content=content,
        structured={
            "error": error,
            "decision": str(verdict.decision),
            "reason": verdict.reason,
        },
        is_error=True,
    )
    ctx._record_policy_denial(
        source=source,
        denied_by=_COMMAND_POLICY_DENIAL_SOURCE,
        decision=verdict.decision.value,
        reason=reason,
        result=raw_result,
    )
    return _bound_policy_denial_result(raw_result)


def _command_from_args(args: dict) -> ExecCommand:
    argv = args.get("argv")
    shell = args.get("shell")
    if argv is not None and shell is not None:
        raise ValueError("Tool arguments `argv` and `shell` cannot both be provided.")
    kind = args.get("kind")
    if kind is None:
        if argv is not None:
            kind = "process"
        elif shell is not None:
            kind = "shell"
        else:
            raise ValueError("Tool arguments must include `argv` or `shell`.")
    if kind == "process":
        if shell is not None:
            raise ValueError("Tool argument `shell` cannot be provided when kind is `process`.")
        if argv is None:
            raise ValueError("Tool argument `argv` is required when kind is `process`.")
        if type(argv) is not list:
            raise ValueError("Tool argument `argv` must be a list.")
        for item in argv:
            if type(item) is not str:
                raise ValueError("Tool argument `argv` entries must be strings.")
            _validate_command_text(item, "argv entry")
        return ExecCommand.process(*argv)
    if kind == "shell":
        if argv is not None:
            raise ValueError("Tool argument `argv` cannot be provided when kind is `shell`.")
        if shell is None:
            raise ValueError("Tool argument `shell` is required when kind is `shell`.")
        if type(shell) is not str:
            raise ValueError("Tool argument `shell` must be a string.")
        _validate_command_text(shell, "shell")
        return ExecCommand.bash(shell)
    raise ValueError("Tool argument `kind` must be `process` or `shell`.")


def _optional_string(
    args: dict,
    key: str,
    *,
    allow_blank: bool = False,
) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    if allow_blank:
        return value
    return require_nonblank(value, key)


def _optional_limited_int(
    args: dict,
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value <= 0:
        raise ValueError(f"Tool argument `{key}` must be greater than zero.")
    if value > maximum:
        raise ValueError(f"Tool argument `{key}` must be at most {maximum}.")
    return value


def _optional_env(args: dict) -> dict[str, str] | None:
    value = args.get("env")
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Tool argument `env` must be an object.")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise ValueError("Tool argument `env` keys must be strings.")
        key = require_clean_nonblank(key, "env key")
        _validate_command_text(key, "env key")
        if "=" in key or "\n" in key or "\r" in key:
            raise ValueError(
                "Tool argument `env key` must not contain '=', carriage returns, or newlines."
            )
        if type(item) is not str:
            raise ValueError("Tool argument `env` values must be strings.")
        _validate_command_text(item, "env value")
        if "\n" in item or "\r" in item:
            raise ValueError(
                "Tool argument `env value` must not contain carriage returns or newlines."
            )
        copied[key] = item
    return copied


def _validate_command_text(value: str, field_name: str) -> None:
    require_unicode_scalar_text(value, field_name)
    if "\0" in value:
        raise ValueError(f"Tool argument `{field_name}` must not contain NUL characters.")


@runtime_checkable
class _CommandRunnerHandle(Protocol):
    async def exec(self, command: ExecCommand, **kwargs) -> object: ...

    def resolve_cwd(self, cwd: str | None = None) -> str: ...


@runtime_checkable
class _CommandPreflightHandle(Protocol):
    def preflight_exec(self, command: ExecCommand, **kwargs) -> object: ...


def _require_runner(ctx: ToolContext) -> _CommandRunnerHandle | None:
    if ctx.runner is None:
        return None
    if not isinstance(ctx.runner, _CommandRunnerHandle):
        raise TypeError("Tool context runner must support command execution and cwd resolution.")
    return ctx.runner


def _require_runner_preflight(runner: _CommandRunnerHandle) -> _CommandPreflightHandle:
    if not isinstance(runner, _CommandPreflightHandle):
        raise TypeError("Policy-enabled command runners must support command preflight.")
    return runner


async def _run_command_preflight(
    runner: _CommandPreflightHandle,
    command: ExecCommand,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout_s: int | None,
    stdin: str | None,
    output_limit_bytes: int | None,
) -> None:
    """Run one policy preflight without surrendering caller cancellation."""

    if type(runner) is InvocationRunnerHandle:
        await _run_invocation_command_preflight(
            runner,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        return

    async def operation(
        preflight_runner: _CommandPreflightHandle = runner,
        owned_command: ExecCommand = command,
        owned_cwd: str | None = cwd,
        owned_env: dict[str, str] | None = env,
        owned_timeout_s: int | None = timeout_s,
        owned_stdin: str | None = stdin,
        owned_output_limit_bytes: int | None = output_limit_bytes,
    ) -> object:
        result = preflight_runner.preflight_exec(
            owned_command,
            cwd=owned_cwd,
            env=owned_env,
            timeout_s=owned_timeout_s,
            stdin=owned_stdin,
            output_limit_bytes=owned_output_limit_bytes,
        )
        if isawaitable(result):
            result = await result
        if result is not None:
            raise TypeError("Command runner preflight must return None.")
        return None

    pending = await_invocation_operation(operation)
    del operation
    outcome = await pending
    del pending
    failure = outcome.error
    caller_cancellation = outcome.cancellation
    del outcome
    if failure is not None:
        _clear_preflight_traceback_frames(failure)
        failure.__traceback__ = None
    del runner, command, cwd, env, timeout_s, stdin, output_limit_bytes
    if caller_cancellation is not None:
        secondary_failure = (
            runner_cancellation_failure(failure)
            if isinstance(failure, asyncio.CancelledError)
            else failure
        )
        del failure
        raise caller_cancellation from secondary_failure
    if isinstance(failure, asyncio.CancelledError):
        safe_failure = runner_execution_error(failure, adapter="unknown")
        del failure
        raise safe_failure from None
    if failure is not None:
        raise failure from None


async def _run_invocation_command_preflight(
    runner: InvocationRunnerHandle,
    command: ExecCommand,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout_s: int | None,
    stdin: str | None,
    output_limit_bytes: int | None,
) -> None:
    """Preserve the invocation handle's redaction-aware cancellation boundary."""

    invocation_failure: BaseException | None = None
    result: object | None = None
    try:
        result = runner.preflight_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        if isawaitable(result):
            result = await result
        if result is not None:
            raise TypeError("Command runner preflight must return None.")
    except BaseException as error:
        invocation_failure = error
    if invocation_failure is None:
        return
    _clear_preflight_traceback_frames(invocation_failure)
    invocation_failure.__traceback__ = None
    published_failure = invocation_failure
    del runner, command, cwd, env, timeout_s, stdin, output_limit_bytes
    del invocation_failure, result
    raise published_failure


def _runner_unavailable_result(exc: RunnerUnavailableError) -> ToolResult:
    return ToolResult(
        content=str(exc),
        structured={
            "error": "runner_unavailable",
            "diagnostic": copy_json_value(exc.diagnostic, "diagnostic"),
        },
        artifacts=exc.artifacts,
        is_error=True,
    )


def _require_exec_result(result: object) -> ExecResult:
    if type(result) is not ExecResult:
        raise TypeError("Runner returned invalid result type.")
    return result


def _command_content(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    timed_out: bool,
    cancelled: bool,
    timeout_s: int,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> str:
    output = stdout.strip()
    error = stderr.strip()
    suffix = ""
    if stdout_truncated or stderr_truncated:
        suffix = "\n\n[output truncated]"
    if timed_out:
        reason = f"Command timed out after {timeout_s} seconds."
        if output and error:
            return f"{reason}\n\nstdout:\n{output}\n\nstderr:\n{error}{suffix}"
        if output:
            return f"{reason}\n\nstdout:\n{output}{suffix}"
        if error:
            return f"{reason}\n\nstderr:\n{error}{suffix}"
        return f"{reason}{suffix}"
    if cancelled:
        reason = "Command was cancelled."
        if output and error:
            return f"{reason}\n\nstdout:\n{output}\n\nstderr:\n{error}{suffix}"
        if output:
            return f"{reason}\n\nstdout:\n{output}{suffix}"
        if error:
            return f"{reason}\n\nstderr:\n{error}{suffix}"
        return f"{reason}{suffix}"
    if exit_code != 0:
        reason = f"Command exited with code {exit_code}."
        if output and error:
            return f"{reason}\n\nstdout:\n{output}\n\nstderr:\n{error}{suffix}"
        if output:
            return f"{reason}\n\nstdout:\n{output}{suffix}"
        if error:
            return f"{reason}\n\nstderr:\n{error}{suffix}"
        return f"{reason}{suffix}"
    if output and error:
        return f"stdout:\n{output}\n\nstderr:\n{error}{suffix}"
    if output:
        return f"{output}{suffix}"
    if error:
        return f"{error}{suffix}"
    return f"Command exited with code {exit_code}.{suffix}"

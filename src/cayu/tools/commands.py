from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.tools._errors import structured_invalid_arguments

DEFAULT_OUTPUT_LIMIT_BYTES = 50_000
MAX_OUTPUT_LIMIT_BYTES = 200_000
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600


class CommandPolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class CommandRequest(BaseModel):
    """Resolved exec request evaluated by a :class:`CommandPolicy`.

    Carries everything the tool would hand to the runner (command shape plus
    ``cwd``/``env``/``stdin``/timeout) so policies can vet the full blast
    radius, not just the command line.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: ExecCommand
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_s: int
    stdin: str | None = None


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

    Without a policy the model-controlled command, ``cwd``, and ``env`` pass
    straight to the runner. Hosts that need an allow/deny/approval seam attach
    a policy via ``ExecCommandTool(policy=...)``.
    """

    @abstractmethod
    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        """Return whether this command may execute."""


class ExecCommandTool(Tool):
    spec = ToolSpec(
        name="exec_command",
        # Runs commands with side effects; never overlaps other tools in a round.
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
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

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        policy: CommandPolicy | None = None,
    ) -> None:
        super().__init__(spec)
        if policy is not None and not isinstance(policy, CommandPolicy):
            raise TypeError("ExecCommandTool policy must implement CommandPolicy.")
        self._policy = policy

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        runner = _require_runner(ctx)
        if runner is None:
            return ToolResult(
                content="No runner configured for this tool call.",
                is_error=True,
            )
        max_output_bytes = _optional_limited_int(
            args,
            "max_output_bytes",
            default=DEFAULT_OUTPUT_LIMIT_BYTES,
            maximum=MAX_OUTPUT_LIMIT_BYTES,
        )
        timeout_s = _optional_limited_int(
            args,
            "timeout_s",
            default=DEFAULT_TIMEOUT_SECONDS,
            maximum=MAX_TIMEOUT_SECONDS,
        )
        command = _command_from_args(args)
        cwd = _optional_string(args, "cwd")
        env = _optional_env(args)
        stdin = _optional_string(args, "stdin", allow_blank=True)
        if self._policy is not None:
            verdict = await self._policy.evaluate(
                ctx,
                CommandRequest(
                    command=command,
                    cwd=cwd,
                    env=env,
                    timeout_s=timeout_s,
                    stdin=stdin,
                ),
            )
            if type(verdict) is not CommandPolicyResult:
                raise TypeError("Command policy must return a CommandPolicyResult.")
            if verdict.decision is not CommandPolicyDecision.ALLOW:
                return _policy_refusal_result(verdict)
        result = await runner.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=max_output_bytes,
        )
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
            is_error=result.timed_out or result.cancelled,
        )


def _policy_refusal_result(verdict: CommandPolicyResult) -> ToolResult:
    if verdict.decision is CommandPolicyDecision.DENY:
        content = "Command denied by policy."
        error = "command_denied"
    else:
        content = "Command requires approval before it can run."
        error = "command_approval_required"
    if verdict.reason is not None:
        content = f"{content} {verdict.reason}"
    return ToolResult(
        content=content,
        structured={
            "error": error,
            "decision": str(verdict.decision),
            "reason": verdict.reason,
        },
        is_error=True,
    )


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
        return ExecCommand.process(*argv)
    if kind == "shell":
        if argv is not None:
            raise ValueError("Tool argument `argv` cannot be provided when kind is `shell`.")
        if shell is None:
            raise ValueError("Tool argument `shell` is required when kind is `shell`.")
        if type(shell) is not str:
            raise ValueError("Tool argument `shell` must be a string.")
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
        if type(item) is not str:
            raise ValueError("Tool argument `env` values must be strings.")
        copied[key] = item
    return copied


def _require_runner(ctx: ToolContext) -> Runner | None:
    if ctx.runner is None:
        return None
    if not isinstance(ctx.runner, Runner):
        raise TypeError("Tool context runner must implement Runner.")
    return ctx.runner


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
        return reason
    if cancelled:
        reason = "Command was cancelled."
        if output and error:
            return f"{reason}\n\nstdout:\n{output}\n\nstderr:\n{error}{suffix}"
        if output:
            return f"{reason}\n\nstdout:\n{output}{suffix}"
        if error:
            return f"{reason}\n\nstderr:\n{error}{suffix}"
        return reason
    if exit_code != 0:
        reason = f"Command exited with code {exit_code}."
        if output and error:
            return f"{reason}\n\nstdout:\n{output}\n\nstderr:\n{error}{suffix}"
        if output:
            return f"{reason}\n\nstdout:\n{output}{suffix}"
        if error:
            return f"{reason}\n\nstderr:\n{error}{suffix}"
        return reason
    if output and error:
        return f"stdout:\n{output}\n\nstderr:\n{error}{suffix}"
    if output:
        return f"{output}{suffix}"
    if error:
        return f"{error}{suffix}"
    return f"Command exited with code {exit_code}."

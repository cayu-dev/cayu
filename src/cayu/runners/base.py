from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value

DEFAULT_EXEC_OUTPUT_LIMIT_BYTES = 1024 * 1024


class RunnerCancelledError(asyncio.CancelledError):
    """Cancelled runner execution with optional cleanup diagnostics."""

    def __init__(
        self,
        message: str = "Runner command was cancelled.",
        *,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifacts = copy_json_value([] if artifacts is None else artifacts, "artifacts")


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
    model_config = ConfigDict(extra="forbid")

    stdout: str = ""
    stderr: str = ""
    exit_code: StrictInt = 0
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    stdout_truncated: StrictBool = False
    stderr_truncated: StrictBool = False
    artifacts: list[dict] = Field(default_factory=list)

    @field_validator("artifacts", mode="before")
    @classmethod
    def copy_artifacts(cls, value: list[dict]) -> list[dict]:
        return copy_json_value(value, "artifacts")


class Runner(ABC):
    """Executes commands/code in a workspace or sandbox."""

    isolation: str = "unknown"

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

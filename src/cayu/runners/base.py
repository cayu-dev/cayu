from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecCommand(BaseModel):
    """Command to execute.

    `argv` is the default safe process form. `shell` is reserved for explicit
    shell scripts where parsing, expansion, and quoting are intentional.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["process", "shell"] = "process"
    argv: list[str] | None = None
    shell: str | None = None

    @classmethod
    def process(cls, *argv: str) -> "ExecCommand":
        return cls(kind="process", argv=list(argv))

    @classmethod
    def bash(cls, script: str) -> "ExecCommand":
        return cls(kind="shell", shell=script)

    @model_validator(mode="after")
    def validate_shape(self) -> "ExecCommand":
        if self.kind == "process":
            if not self.argv:
                raise ValueError("Process commands require non-empty argv.")
            if any(not item.strip() for item in self.argv):
                raise ValueError("Process argv entries must be non-empty strings.")
            if self.shell is not None:
                raise ValueError("Process commands cannot define shell script.")
        if self.kind == "shell":
            if not self.shell or not self.shell.strip():
                raise ValueError("Shell commands require a non-empty script.")
            if self.argv is not None:
                raise ValueError("Shell commands cannot define argv.")
        return self


class ExecResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    cancelled: bool = False
    artifacts: list[dict] = Field(default_factory=list)


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
    ) -> ExecResult:
        """Execute a command and return stdout/stderr/exit metadata."""

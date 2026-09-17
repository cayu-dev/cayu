"""Public, value-free process restrictions and explicitly published capabilities."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProcessCommandDenialCode(StrEnum):
    EXECUTABLE = "executable"
    COMMAND_KIND = "command_kind"
    SHELL = "shell"
    WORKING_DIRECTORY = "working_directory"
    ENVIRONMENT_NAME = "environment_name"
    ENVIRONMENT_VALUE_SIZE = "environment_value_size"
    ENVIRONMENT_VALUE = "environment_value"
    STDIN = "stdin"
    STDIN_SIZE = "stdin_size"
    TIMEOUT = "timeout"
    COMMAND_APPROVAL = "command_approval"


PROCESS_CORRECTION_HINTS = {
    ProcessCommandDenialCode.EXECUTABLE: "Use an exact published allowed executable; if none is published, ask the operator for an admitted executable.",
    ProcessCommandDenialCode.COMMAND_KIND: "Use a process argv request with an admitted executable.",
    ProcessCommandDenialCode.SHELL: "Use process argv instead of shell text, or ask the operator for shell authority.",
    ProcessCommandDenialCode.WORKING_DIRECTORY: "Choose a cwd within the operator-declared process policy roots; runner containment alone is insufficient. Ask the operator for an admitted directory if unknown.",
    ProcessCommandDenialCode.ENVIRONMENT_NAME: "Remove the rejected environment override or use a published allowed name.",
    ProcessCommandDenialCode.ENVIRONMENT_VALUE_SIZE: "Remove the override or reduce its UTF-8 value to max_env_value_bytes.",
    ProcessCommandDenialCode.ENVIRONMENT_VALUE: "Remove the override or ask the operator for the required value; configured values are never published.",
    ProcessCommandDenialCode.STDIN: "Omit stdin; this policy does not admit input.",
    ProcessCommandDenialCode.STDIN_SIZE: "Omit stdin or reduce its UTF-8 size to max_stdin_bytes.",
    ProcessCommandDenialCode.TIMEOUT: "Set timeout_s explicitly to a positive integer at or below max_timeout_s.",
    ProcessCommandDenialCode.COMMAND_APPROVAL: "Choose an allowed executable or ask the operator for authorization; this refusal does not create an approval checkpoint.",
}


class ProcessCommandCapabilities(BaseModel):
    """Safe discovery snapshot. Names require explicit host publication consent.

    A profile ID is an application-owned public label, not a fingerprint or an
    execution grant. None means the application has not declared a public ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_type: Literal["process_command_policy"] = "process_command_policy"
    profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    allowed_executables: tuple[str, ...] = Field(default=(), max_length=64)
    approval_required_executables: tuple[str, ...] = Field(default=(), max_length=64)
    executable_names_withheld: bool
    allowed_env_names: tuple[str, ...] = Field(default=(), max_length=64)
    environment_names_withheld: bool
    cwd_values_state: Literal["withheld_private_paths"] = "withheld_private_paths"
    max_env_value_bytes: int = Field(ge=0)
    allow_stdin: bool
    max_stdin_bytes: int = Field(ge=0)
    max_timeout_s: int = Field(ge=1, le=600)
    shell_decision: Literal["allow", "deny", "require_command_approval"]


class ProcessCommandDiagnostic(BaseModel):
    """Typed correction for one refusal; never contains argv or private values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ProcessCommandDenialCode
    rejected_name: str | None = Field(default=None, max_length=128)
    value_state: Literal[
        "published_by_policy",
        "withheld_not_declared_public",
        "withheld_private_value",
        "not_applicable",
    ]
    capabilities: ProcessCommandCapabilities

    @property
    def correction_hint(self) -> str:
        return PROCESS_CORRECTION_HINTS[self.code]

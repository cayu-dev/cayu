"""Immutable admitted toolchain profiles for trusted-repository Docker coding."""

from __future__ import annotations

import os
import re
import stat
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionRequirements,
    evaluate_execution_admission,
)
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions
from cayu.workspaces.base import WorkspaceReadResult

DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA = "cayu.docker_coding_toolchain_profile.v1"
DOCKER_CODING_COMMAND_AUTHORITY_SCHEMA = "cayu.docker_coding_command_authority.v1"
MAX_DOCKER_CODING_PROFILE_COMMANDS = 128
MAX_DOCKER_CODING_PROFILE_DEPENDENCY_INPUTS = 64
MAX_DOCKER_CODING_PROFILE_PROBES = 32
MAX_DOCKER_CODING_DEPENDENCY_INPUT_BYTES = 16 * 1024 * 1024
MAX_STRUCTURED_COMMAND_ARGS = 64
MAX_STRUCTURED_COMMAND_ARG_BYTES = 4 * 1024
MAX_STRUCTURED_COMMAND_TOTAL_ARG_BYTES = 32 * 1024

_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_SELECTOR = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,127}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHELL_CONTROL_CHARACTERS = frozenset("|&;<>()`$\n\r\0")
_MODEL_CONTROL_CHARACTERS = _SHELL_CONTROL_CHARACTERS | frozenset("*?{}~'\"")
_PROTECTED_PATH_PARTS = frozenset({".cayu", ".git", ".runtime"})
_REQUIRED_PROHIBITED_CAPABILITIES = frozenset(
    {
        "docker_socket",
        "host_credentials",
        "host_git",
        "host_mounts",
        "network",
        "runtime_package_installation",
    }
)


class DockerCodingToolchainError(RuntimeError):
    """A typed safe failure at the toolchain profile boundary."""

    def __init__(self, code: str, message: str, *, paths: tuple[str, ...] = ()) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        self.paths = tuple(paths)
        super().__init__(require_durable_clean_nonblank(message, "message"))

    @property
    def path_count(self) -> int:
        """Return the number of affected private source paths."""

        return len(self.paths)

    @property
    def paths_fingerprint(self) -> str:
        """Return a stable identity without publishing repository paths."""

        return _fingerprint(
            {"paths": list(sorted(self.paths))},
            "toolchain_dependency_paths",
        )


class DockerCodingDependencyInput(BaseModel):
    """One source input whose exact bytes determine toolchain freshness."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str = Field(max_length=4096)
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    max_bytes: StrictInt = Field(
        default=MAX_DOCKER_CODING_DEPENDENCY_INPUT_BYTES,
        ge=1,
        le=MAX_DOCKER_CODING_DEPENDENCY_INPUT_BYTES,
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _workspace_relative_path(value, field_name="path", allow_root=False)


class DockerCodingFixedEnvironmentVariable(BaseModel):
    """One non-secret implementation-owned environment value."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(max_length=128)
    value: str = Field(max_length=4096)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = require_durable_clean_nonblank(value, "name")
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError("Environment names must use uppercase portable identifiers.")
        return name

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if type(value) is not str or any(character in value for character in "\0\r\n"):
            raise ValueError("Environment values must be single-line Unicode strings.")
        value.encode("utf-8")
        return value


class DockerCodingAdmissionProbe(BaseModel):
    """One bounded implementation-owned probe against the exact final container."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    probe_id: str = Field(max_length=128)
    argv: tuple[str, ...] = Field(min_length=1, max_length=32)
    expected_exit_codes: tuple[StrictInt, ...] = (0,)
    stdout_sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    timeout_seconds: StrictInt = Field(default=10, ge=1, le=60)
    max_output_bytes: StrictInt = Field(default=16 * 1024, ge=1, le=64 * 1024)

    @field_validator("probe_id")
    @classmethod
    def validate_probe_id(cls, value: str) -> str:
        probe_id = require_durable_clean_nonblank(value, "probe_id")
        if _SELECTOR.fullmatch(probe_id) is None:
            raise ValueError("probe_id must be a portable lowercase identifier.")
        return probe_id

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_owned_argv(value, field_name="argv")

    @field_validator("expected_exit_codes")
    @classmethod
    def validate_expected_exit_codes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) > 16:
            raise ValueError("expected_exit_codes must contain between 1 and 16 entries.")
        if any(type(item) is not int or item < 0 or item > 255 for item in value):
            raise ValueError("expected_exit_codes entries must be integers from 0 through 255.")
        if tuple(sorted(set(value))) != value:
            raise ValueError("expected_exit_codes must be unique and sorted.")
        return value


class DockerCodingCommandAuthority(BaseModel):
    """Closed application-owned authority for one named check or structured command."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.docker_coding_command_authority.v1"] = Field(
        default=DOCKER_CODING_COMMAND_AUTHORITY_SCHEMA,
        alias="schema",
    )
    selector: str = Field(max_length=128)
    revision: str = Field(max_length=128)
    description: str = Field(max_length=4096)
    exposure: Literal["named_check", "structured_command"]
    executable: str = Field(max_length=4096)
    fixed_arguments: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    suffix_arguments: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    allowed_flags: tuple[str, ...] = Field(default_factory=tuple, max_length=64)
    required_flags: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    flags_with_values: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    path_value_flags: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    allowed_literals: tuple[str, ...] = Field(default_factory=tuple, max_length=64)
    allow_positional_arguments: StrictBool = False
    positional_arguments_are_paths: StrictBool = False
    positional_path_prefixes: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    positional_path_suffixes: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    allow_pytest_node_ids: StrictBool = False
    min_arguments: StrictInt = Field(default=0, ge=0, le=MAX_STRUCTURED_COMMAND_ARGS)
    max_arguments: StrictInt = Field(default=16, ge=0, le=MAX_STRUCTURED_COMMAND_ARGS)
    max_argument_bytes: StrictInt = Field(
        default=1024,
        ge=1,
        le=MAX_STRUCTURED_COMMAND_ARG_BYTES,
    )
    max_total_argument_bytes: StrictInt = Field(
        default=8 * 1024,
        ge=1,
        le=MAX_STRUCTURED_COMMAND_TOTAL_ARG_BYTES,
    )
    default_working_directory: str = "."
    allowed_working_directories: tuple[str, ...] = (".",)
    fixed_environment: tuple[DockerCodingFixedEnvironmentVariable, ...] = ()
    timeout_seconds: StrictInt = Field(default=60, ge=1, le=600)
    max_output_bytes: StrictInt = Field(default=50_000, ge=1, le=200_000)
    max_model_output_bytes: StrictInt = Field(default=16_000, ge=256, le=50_000)
    allowed_exit_codes: tuple[StrictInt, ...] = (0,)
    effect: Literal["read_only", "workspace_mutating"] = "read_only"
    parallel_safe: StrictBool = False
    idempotent: StrictBool = False
    mutation_path_prefixes: tuple[str, ...] = ()
    dependency_sensitive: StrictBool = True
    approval: Literal["ordinary", "required"] = "ordinary"
    approval_expires_in_seconds: StrictInt | None = Field(
        default=None,
        ge=1,
        le=86_400,
    )

    @field_validator("selector")
    @classmethod
    def validate_selector(cls, value: str) -> str:
        selector = require_durable_clean_nonblank(value, "selector")
        if _SELECTOR.fullmatch(selector) is None:
            raise ValueError("selector must be a portable lowercase identifier.")
        return selector

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        revision = require_durable_clean_nonblank(value, "revision")
        if _REVISION.fullmatch(revision) is None:
            raise ValueError("revision must be a portable version identifier.")
        return revision

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "description")

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        executable = require_durable_clean_nonblank(value, "executable")
        if not executable.startswith("/") or any(
            character.isspace() or character in _SHELL_CONTROL_CHARACTERS
            for character in executable
        ):
            raise ValueError("Command-authority executables must be exact absolute guest paths.")
        parsed = PurePosixPath(executable)
        if ".." in parsed.parts or str(parsed) != executable:
            raise ValueError("Command-authority executable paths must be normalized.")
        return executable

    @field_validator("fixed_arguments", "suffix_arguments")
    @classmethod
    def validate_owned_arguments(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _validated_owned_argv(value, field_name=info.field_name, allow_empty=True)

    @field_validator(
        "allowed_flags",
        "required_flags",
        "flags_with_values",
        "path_value_flags",
    )
    @classmethod
    def validate_flags(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        for flag in value:
            if (
                type(flag) is not str
                or not flag.startswith("-")
                or flag in {"-", "--"}
                or "=" in flag
                or any(character.isspace() for character in flag)
                or any(character in flag for character in _SHELL_CONTROL_CHARACTERS)
            ):
                raise ValueError(f"{info.field_name} entries must be exact portable flags.")
        return value

    @field_validator("allowed_literals")
    @classmethod
    def validate_literals(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("allowed_literals must be unique and sorted.")
        return _validated_owned_argv(value, field_name="allowed_literals", allow_empty=True)

    @field_validator("positional_path_prefixes", "mutation_path_prefixes")
    @classmethod
    def validate_path_prefixes(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        owned = tuple(
            _workspace_relative_path(item, field_name=info.field_name, allow_root=False)
            for item in value
        )
        if tuple(sorted(set(owned))) != owned:
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        return owned

    @field_validator("positional_path_suffixes")
    @classmethod
    def validate_path_suffixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("positional_path_suffixes must be unique and sorted.")
        if any(
            type(item) is not str
            or not item.startswith(".")
            or "/" in item
            or "\\" in item
            or any(character in item for character in _SHELL_CONTROL_CHARACTERS)
            for item in value
        ):
            raise ValueError("positional_path_suffixes entries must be portable suffixes.")
        return value

    @field_validator("default_working_directory")
    @classmethod
    def validate_default_working_directory(cls, value: str) -> str:
        return _workspace_relative_path(
            value,
            field_name="default_working_directory",
            allow_root=True,
        )

    @field_validator("allowed_working_directories")
    @classmethod
    def validate_allowed_working_directories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("allowed_working_directories must not be empty.")
        owned = tuple(
            _workspace_relative_path(
                item,
                field_name="allowed_working_directories",
                allow_root=True,
            )
            for item in value
        )
        if tuple(sorted(set(owned))) != owned:
            raise ValueError("allowed_working_directories must be unique and sorted.")
        return owned

    @field_validator("fixed_environment")
    @classmethod
    def validate_fixed_environment(
        cls,
        value: tuple[DockerCodingFixedEnvironmentVariable, ...],
    ) -> tuple[DockerCodingFixedEnvironmentVariable, ...]:
        names = [item.name for item in value]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("fixed_environment must be unique and sorted by name.")
        return value

    @field_validator("allowed_exit_codes")
    @classmethod
    def validate_allowed_exit_codes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) > 16:
            raise ValueError("allowed_exit_codes must contain between 1 and 16 entries.")
        if any(type(item) is not int or item < 0 or item > 255 for item in value):
            raise ValueError("allowed_exit_codes entries must be integers from 0 through 255.")
        if tuple(sorted(set(value))) != value:
            raise ValueError("allowed_exit_codes must be unique and sorted.")
        return value

    @model_validator(mode="after")
    def validate_relationships(self) -> DockerCodingCommandAuthority:
        if not set(self.flags_with_values).issubset(self.allowed_flags):
            raise ValueError("flags_with_values must be a subset of allowed_flags.")
        if not set(self.required_flags).issubset(self.allowed_flags):
            raise ValueError("required_flags must be a subset of allowed_flags.")
        if not set(self.path_value_flags).issubset(self.flags_with_values):
            raise ValueError("path_value_flags must be a subset of flags_with_values.")
        if self.exposure == "named_check" and (
            self.allowed_flags
            or self.allowed_literals
            or self.allow_positional_arguments
            or self.max_arguments != 0
        ):
            raise ValueError("Named-check authorities cannot accept model arguments.")
        if self.positional_arguments_are_paths and not self.allow_positional_arguments:
            raise ValueError("positional_arguments_are_paths requires allow_positional_arguments.")
        if self.positional_path_prefixes and not self.positional_arguments_are_paths:
            raise ValueError("positional_path_prefixes require positional path arguments.")
        if self.positional_path_suffixes and not self.positional_arguments_are_paths:
            raise ValueError("positional_path_suffixes require positional path arguments.")
        if self.allow_pytest_node_ids and not self.positional_arguments_are_paths:
            raise ValueError("allow_pytest_node_ids requires positional path arguments.")
        if self.default_working_directory not in self.allowed_working_directories:
            raise ValueError(
                "default_working_directory must be one of allowed_working_directories."
            )
        if self.min_arguments > self.max_arguments:
            raise ValueError("min_arguments cannot exceed max_arguments.")
        if self.effect == "read_only" and self.mutation_path_prefixes:
            raise ValueError("Read-only command authorities cannot declare mutation paths.")
        if self.effect == "workspace_mutating" and not self.mutation_path_prefixes:
            raise ValueError("Mutating command authorities require bounded mutation paths.")
        if self.effect == "workspace_mutating" and self.parallel_safe:
            raise ValueError("Workspace-mutating command authorities cannot be parallel-safe.")
        if self.approval == "ordinary" and self.approval_expires_in_seconds is not None:
            raise ValueError("Only approval-required authorities can declare approval expiry.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json", by_alias=True), "command_authority")

    @property
    def environment(self) -> dict[str, str] | None:
        if not self.fixed_environment:
            return None
        return {item.name: item.value for item in self.fixed_environment}

    def command_argv(self, arguments: tuple[str, ...] = ()) -> tuple[str, ...]:
        validated = self.validate_model_arguments(arguments)
        return (self.executable, *self.fixed_arguments, *validated, *self.suffix_arguments)

    def validate_model_arguments(self, arguments: tuple[str, ...]) -> tuple[str, ...]:
        if type(arguments) is not tuple or any(type(item) is not str for item in arguments):
            raise ValueError("Structured command arguments must be a tuple of strings.")
        if len(arguments) > self.max_arguments:
            raise ValueError(f"Structured command accepts at most {self.max_arguments} arguments.")
        if len(arguments) < self.min_arguments:
            raise ValueError(
                f"Structured command requires at least {self.min_arguments} arguments."
            )
        total = 0
        expect_value_for: str | None = None
        seen_flags: set[str] = set()
        for argument in arguments:
            try:
                encoded = argument.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(
                    "Structured command arguments must contain Unicode scalars."
                ) from None
            if not argument or len(encoded) > self.max_argument_bytes:
                raise ValueError("Structured command argument is empty or exceeds its byte limit.")
            total += len(encoded)
            if total > self.max_total_argument_bytes:
                raise ValueError("Structured command arguments exceed their aggregate byte limit.")
            if any(character in argument for character in _MODEL_CONTROL_CHARACTERS):
                raise ValueError(
                    "Structured command arguments cannot contain shell or expansion syntax."
                )
            if argument.startswith("@"):
                raise ValueError("Structured command response-file indirection is not allowed.")
            if expect_value_for is not None:
                if argument.startswith("-"):
                    raise ValueError(
                        f"Structured command flag {expect_value_for!r} requires a value."
                    )
                if expect_value_for in self.path_value_flags:
                    _validate_model_path_argument(argument, authority=self)
                expect_value_for = None
                continue
            if argument.startswith("-"):
                flag, separator, attached = argument.partition("=")
                if flag not in self.allowed_flags:
                    raise ValueError("Structured command argument contains an undeclared flag.")
                if flag in seen_flags:
                    raise ValueError("Structured command flag cannot be repeated.")
                seen_flags.add(flag)
                if flag in self.flags_with_values:
                    if separator:
                        if not attached:
                            raise ValueError("Structured command flag requires a nonempty value.")
                        if flag in self.path_value_flags:
                            _validate_model_path_argument(attached, authority=self)
                    else:
                        expect_value_for = flag
                elif separator:
                    raise ValueError("Structured command flag does not accept an attached value.")
                continue
            if argument in self.allowed_literals:
                continue
            if self.allowed_literals:
                raise ValueError("Structured command argument contains an undeclared literal.")
            if not self.allow_positional_arguments:
                raise ValueError(
                    "Structured command selector does not accept positional arguments."
                )
            if self.positional_arguments_are_paths:
                _validate_model_path_argument(argument, authority=self)
        if expect_value_for is not None:
            raise ValueError(f"Structured command flag {expect_value_for!r} requires a value.")
        missing_flags = set(self.required_flags) - seen_flags
        if missing_flags:
            raise ValueError("Structured command request omits a required flag.")
        return tuple(arguments)

    def validate_working_directory(self, value: str | None) -> str:
        candidate = self.default_working_directory if value is None else value
        candidate = _workspace_relative_path(
            candidate,
            field_name="working_directory",
            allow_root=True,
        )
        if candidate not in self.allowed_working_directories:
            raise ValueError("Structured command working directory is not admitted.")
        return candidate


class DockerCodingToolchainProfile(BaseModel):
    """Versioned reconstructable authority for one exact Docker coding environment."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.docker_coding_toolchain_profile.v1"] = Field(
        default=DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA,
        alias="schema",
    )
    profile_id: str = Field(max_length=128)
    revision: str = Field(max_length=128)
    image_identity: DockerImageIdentity
    platform_os: Literal["linux"] = "linux"
    platform_architecture: Literal["amd64", "arm64"]
    runtime_user: str = Field(default="1000:1000", max_length=64)
    working_directory: str = "/workspace"
    workspace_path: str = "/workspace"
    read_only_support_paths: tuple[str, ...] = ()
    restrictions: DockerWorkloadRestrictions = Field(default_factory=DockerWorkloadRestrictions)
    command_authorities: tuple[DockerCodingCommandAuthority, ...] = Field(
        default_factory=tuple,
        max_length=MAX_DOCKER_CODING_PROFILE_COMMANDS,
    )
    dependency_inputs: tuple[DockerCodingDependencyInput, ...] = Field(
        default_factory=tuple,
        max_length=MAX_DOCKER_CODING_PROFILE_DEPENDENCY_INPUTS,
    )
    trusted_build_context_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    application_dependency_identity: str | None = Field(default=None, max_length=256)
    admission_probes: tuple[DockerCodingAdmissionProbe, ...] = Field(
        default_factory=tuple,
        max_length=MAX_DOCKER_CODING_PROFILE_PROBES,
    )
    prohibited_capabilities: tuple[str, ...] = tuple(sorted(_REQUIRED_PROHIBITED_CAPABILITIES))
    result_publication_max_bytes: StrictInt = Field(default=50_000, ge=256, le=200_000)
    redaction_behavior_version: str = "1"
    adoption_behavior_version: str = "1"

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        profile_id = require_durable_clean_nonblank(value, "profile_id")
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("profile_id must be a portable identifier.")
        return profile_id

    @field_validator("revision", "redaction_behavior_version", "adoption_behavior_version")
    @classmethod
    def validate_version(cls, value: str, info) -> str:
        version = require_durable_clean_nonblank(value, info.field_name)
        if _REVISION.fullmatch(version) is None:
            raise ValueError(f"{info.field_name} must be a portable version identifier.")
        return version

    @field_validator("application_dependency_identity")
    @classmethod
    def validate_application_dependency_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "application_dependency_identity")

    @field_validator("runtime_user")
    @classmethod
    def validate_runtime_user(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "runtime_user")
        parts = value.split(":")
        if len(parts) != 2 or any(not part.isascii() or not part.isdigit() for part in parts):
            raise ValueError("runtime_user must be a numeric uid:gid pair.")
        if any(int(part) <= 0 or int(part) > 2**31 - 1 for part in parts):
            raise ValueError("runtime_user uid and gid must be positive 32-bit integers.")
        return value

    @field_validator("working_directory", "workspace_path")
    @classmethod
    def validate_absolute_workspace_path(cls, value: str, info) -> str:
        path = require_durable_clean_nonblank(value, info.field_name)
        parsed = PurePosixPath(path)
        if not parsed.is_absolute() or str(parsed) != path or ".." in parsed.parts:
            raise ValueError(f"{info.field_name} must be a normalized absolute guest path.")
        return path

    @field_validator("read_only_support_paths")
    @classmethod
    def validate_read_only_support_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        owned: list[str] = []
        for item in value:
            path = require_durable_clean_nonblank(item, "read_only_support_paths item")
            parsed = PurePosixPath(path)
            if not parsed.is_absolute() or str(parsed) != path or ".." in parsed.parts:
                raise ValueError(
                    "read_only_support_paths entries must be normalized absolute paths."
                )
            owned.append(path)
        if tuple(sorted(set(owned))) != tuple(owned):
            raise ValueError("read_only_support_paths must be unique and sorted.")
        return tuple(owned)

    @field_validator("command_authorities")
    @classmethod
    def validate_command_authorities(
        cls,
        value: tuple[DockerCodingCommandAuthority, ...],
    ) -> tuple[DockerCodingCommandAuthority, ...]:
        selectors = [item.selector for item in value]
        if selectors != sorted(selectors) or len(selectors) != len(set(selectors)):
            raise ValueError("command_authorities must be unique and sorted by selector.")
        return value

    @field_validator("dependency_inputs")
    @classmethod
    def validate_dependency_inputs(
        cls,
        value: tuple[DockerCodingDependencyInput, ...],
    ) -> tuple[DockerCodingDependencyInput, ...]:
        paths = [item.path for item in value]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("dependency_inputs must be unique and sorted by path.")
        return value

    @field_validator("admission_probes")
    @classmethod
    def validate_admission_probes(
        cls,
        value: tuple[DockerCodingAdmissionProbe, ...],
    ) -> tuple[DockerCodingAdmissionProbe, ...]:
        names = [item.probe_id for item in value]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("admission_probes must be unique and sorted by probe_id.")
        return value

    @field_validator("prohibited_capabilities")
    @classmethod
    def validate_prohibited_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        owned = tuple(
            require_durable_clean_nonblank(item, "prohibited_capability") for item in value
        )
        if tuple(sorted(set(owned))) != owned:
            raise ValueError("prohibited_capabilities must be unique and sorted.")
        missing = sorted(_REQUIRED_PROHIBITED_CAPABILITIES - set(owned))
        if missing:
            raise ValueError(
                "Docker coding profiles cannot omit prohibited capabilities: " + ", ".join(missing)
            )
        return owned

    @model_validator(mode="after")
    def validate_profile_relationships(self) -> DockerCodingToolchainProfile:
        if self.runtime_user != self.restrictions.user:
            raise ValueError("runtime_user must match the Docker restriction uid and gid.")
        if self.working_directory != self.workspace_path:
            raise ValueError("The first profile version requires workdir to equal workspace_path.")
        if any(
            path == self.workspace_path or path.startswith(self.workspace_path + "/")
            for path in self.read_only_support_paths
        ):
            raise ValueError("Read-only support paths must stay outside the writable workspace.")
        executable_paths = {authority.executable for authority in self.command_authorities}
        if any(
            authority.max_model_output_bytes * 2 > self.result_publication_max_bytes
            for authority in self.command_authorities
        ):
            raise ValueError(
                "Command model-output bounds must fit the profile publication ceiling."
            )
        for probe in self.admission_probes:
            if probe.argv[0] not in executable_paths:
                raise ValueError("Admission probes must use a declared executable authority.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json", by_alias=True), "toolchain_profile")

    @property
    def dependency_identity(self) -> str:
        return _fingerprint(
            {
                "dependency_inputs": [
                    item.model_dump(mode="json") for item in self.dependency_inputs
                ],
                "trusted_build_context_sha256": self.trusted_build_context_sha256,
                "application_dependency_identity": self.application_dependency_identity,
            },
            "toolchain_dependencies",
        )

    @property
    def required_executables(self) -> tuple[str, ...]:
        return tuple(sorted({item.executable for item in self.command_authorities}))

    @property
    def named_check_authorities(self) -> tuple[DockerCodingCommandAuthority, ...]:
        return tuple(item for item in self.command_authorities if item.exposure == "named_check")

    @property
    def structured_command_authorities(self) -> tuple[DockerCodingCommandAuthority, ...]:
        return tuple(
            item for item in self.command_authorities if item.exposure == "structured_command"
        )

    def command_authority(
        self,
        selector: str,
        *,
        exposure: Literal["named_check", "structured_command"] | None = None,
    ) -> DockerCodingCommandAuthority | None:
        return next(
            (
                item
                for item in self.command_authorities
                if item.selector == selector and (exposure is None or item.exposure == exposure)
            ),
            None,
        )

    def evidence(self) -> dict[str, object]:
        """Return the bounded non-secret identity included in runtime receipts."""

        restrictions = self.restrictions.model_dump(mode="json")
        return {
            "toolchain_profile_id": self.profile_id,
            "toolchain_profile_revision": self.revision,
            "toolchain_profile_fingerprint": self.fingerprint,
            "toolchain_image_fingerprint": self.image_identity.fingerprint,
            "toolchain_image_content_digest": self.image_identity.content_digest,
            "toolchain_platform": f"{self.platform_os}/{self.platform_architecture}",
            "toolchain_runtime_user": self.runtime_user,
            "toolchain_working_directory": self.working_directory,
            "toolchain_runtime_restrictions_identity": _fingerprint(
                restrictions,
                "toolchain_runtime_restrictions",
            ),
            "toolchain_resource_limits": {
                "pids_limit": self.restrictions.pids_limit,
                "memory_bytes": self.restrictions.memory_bytes,
                "memory_swap_bytes": self.restrictions.memory_swap_bytes,
                "cpu_period_us": self.restrictions.cpu_period_us,
                "cpu_quota_us": self.restrictions.cpu_quota_us,
                "shm_size_bytes": self.restrictions.shm_size_bytes,
            },
            "toolchain_prohibited_capabilities": list(self.prohibited_capabilities),
            "toolchain_dependency_identity": self.dependency_identity,
            "toolchain_result_publication_max_bytes": self.result_publication_max_bytes,
            "toolchain_redaction_behavior_version": self.redaction_behavior_version,
            "toolchain_adoption_behavior_version": self.adoption_behavior_version,
        }


@runtime_checkable
class _DependencyWorkspace(Protocol):
    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult: ...


@runtime_checkable
class _AdmittedRunner(Protocol):
    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None: ...


def docker_coding_toolchain_runner_admission_failure(
    runner: object,
    *,
    profile: DockerCodingToolchainProfile,
) -> str | None:
    """Return a stable refusal code unless the active runner matches the profile."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    if not isinstance(runner, _AdmittedRunner):
        return "docker_admission_unavailable"
    try:
        candidate = runner.execution_admission_candidate()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return "docker_admission_unavailable"
    if candidate is None:
        return "docker_admission_unavailable"
    if type(candidate) is not ExecutionAdmissionCandidate or candidate.candidate != "docker":
        return "docker_admission_mismatch"
    evidence = candidate.evidence
    if evidence.image_fingerprint != profile.image_identity.fingerprint:
        return "toolchain_image_drift"
    if evidence.toolchain_profile_fingerprint != profile.fingerprint:
        return "toolchain_profile_drift"
    network = evidence.claim_for("deny_by_default_network")
    if network is None or network.state != "live_verified" or network.observation != "denied":
        return "docker_network_admission_missing"
    requirements = evidence.tool_requirements
    if requirements is None:
        return "toolchain_executable_admission_missing"
    for executable in profile.required_executables:
        observed = requirements.executable_for(executable)
        if observed is None or observed.state != "live_verified":
            return "toolchain_executable_admission_missing"
    admission = evaluate_execution_admission(
        candidate=candidate.candidate,
        requirements=ExecutionRequirements.trusted(
            network_access="deny_by_default",
            minimum_evidence="live_verified",
            required_executables=profile.required_executables,
        ),
        evidence=evidence,
        stage="pre_exposure",
        now=datetime.now(UTC),
    )
    if admission.status != "admitted":
        if any(
            refusal.code in {"future_evidence", "overlong_evidence", "stale_evidence"}
            for refusal in admission.refusals
        ):
            return "docker_admission_stale"
        return "docker_admission_invalid"
    return None


async def verify_docker_coding_toolchain_dependencies(
    profile: DockerCodingToolchainProfile,
    workspace: _DependencyWorkspace,
) -> None:
    """Fail before dispatch when the active workspace no longer matches the profile."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    if not isinstance(workspace, _DependencyWorkspace):
        raise TypeError("workspace must support bounded reads for dependency admission.")
    changed: list[str] = []
    unavailable: list[str] = []
    for dependency in profile.dependency_inputs:
        try:
            result = await workspace.read_bytes(
                dependency.path,
                offset=0,
                max_bytes=dependency.max_bytes + 1,
            )
            if type(result) is not WorkspaceReadResult:
                raise TypeError("Workspace dependency read returned an invalid result.")
            content = result.content
            truncated = result.truncated
            total_bytes = result.total_bytes
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            unavailable.append(dependency.path)
            continue
        if (
            type(content) is not bytes
            or type(truncated) is not bool
            or type(total_bytes) is not int
            or truncated
            or total_bytes > dependency.max_bytes
        ):
            unavailable.append(dependency.path)
            continue
        observed = "sha256:" + sha256(content).hexdigest()
        if observed != dependency.content_sha256:
            changed.append(dependency.path)
    if unavailable:
        raise DockerCodingToolchainError(
            "dependency_inputs_unavailable",
            "Toolchain dependency inputs could not be verified.",
            paths=tuple(unavailable),
        )
    if changed:
        raise DockerCodingToolchainError(
            "dependency_inputs_changed",
            "Toolchain dependency inputs changed; a new admitted profile or build is required.",
            paths=tuple(changed),
        )


def verify_local_docker_coding_toolchain_dependencies(
    profile: DockerCodingToolchainProfile,
    root: Path,
) -> None:
    """Verify source dependency inputs without allocating Docker resources."""

    if type(profile) is not DockerCodingToolchainProfile:
        raise TypeError("profile must be an exact DockerCodingToolchainProfile.")
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path.")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise DockerCodingToolchainError(
            "source_workspace_unavailable",
            "Toolchain source workspace is unavailable.",
        ) from None
    changed: list[str] = []
    unavailable: list[str] = []
    try:
        root_fd = os.open(
            resolved_root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        raise DockerCodingToolchainError(
            "source_workspace_unavailable",
            "Toolchain source workspace is unavailable.",
        ) from None
    try:
        for dependency in profile.dependency_inputs:
            try:
                content = _read_local_dependency_input(root_fd, dependency)
            except OSError:
                unavailable.append(dependency.path)
                continue
            if "sha256:" + sha256(content).hexdigest() != dependency.content_sha256:
                changed.append(dependency.path)
    finally:
        os.close(root_fd)
    if unavailable:
        raise DockerCodingToolchainError(
            "dependency_inputs_unavailable",
            "Toolchain dependency inputs could not be verified.",
            paths=tuple(unavailable),
        )
    if changed:
        raise DockerCodingToolchainError(
            "dependency_inputs_changed",
            "Toolchain dependency inputs changed; a new admitted profile or build is required.",
            paths=tuple(changed),
        )


def _read_local_dependency_input(
    root_fd: int,
    dependency: DockerCodingDependencyInput,
) -> bytes:
    """Read one regular input without following a symlink below the source root."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | os.O_DIRECTORY
    active_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        parts = PurePosixPath(dependency.path).parts
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=active_fd)
            os.close(active_fd)
            active_fd = next_fd
        file_fd = os.open(parts[-1], flags, dir_fd=active_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > dependency.max_bytes:
            raise OSError("Dependency input is not a bounded regular file.")
        chunks: list[bytes] = []
        remaining = dependency.max_bytes + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > dependency.max_bytes:
            raise OSError("Dependency input exceeds its declared byte limit.")
        final_metadata = os.fstat(file_fd)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_size != metadata.st_size
            or final_metadata.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise OSError("Dependency input changed while it was observed.")
        return content
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(active_fd)


def _validated_owned_argv(
    value: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        raise ValueError(f"{field_name} must be a nonempty tuple of strings.")
    total = 0
    for item in value:
        if type(item) is not str or not item or "\0" in item:
            raise ValueError(f"{field_name} entries must be nonempty strings without NULs.")
        try:
            encoded = item.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"{field_name} entries must contain Unicode scalars.") from None
        if len(encoded) > MAX_STRUCTURED_COMMAND_ARG_BYTES:
            raise ValueError(f"{field_name} entries exceed their byte limit.")
        total += len(encoded)
    if total > MAX_STRUCTURED_COMMAND_TOTAL_ARG_BYTES:
        raise ValueError(f"{field_name} exceeds its aggregate byte limit.")
    return value


def _workspace_relative_path(value: str, *, field_name: str, allow_root: bool) -> str:
    path = require_durable_clean_nonblank(value, field_name)
    if "\\" in path or path.startswith("/"):
        raise ValueError(f"{field_name} must be a POSIX workspace-relative path.")
    parsed = PurePosixPath(path)
    if any(part in {"", ".."} or part in _PROTECTED_PATH_PARTS for part in parsed.parts):
        raise ValueError(f"{field_name} must stay outside protected workspace paths.")
    normalized = str(parsed)
    if normalized == ".":
        if not allow_root:
            raise ValueError(f"{field_name} cannot name the workspace root.")
        return normalized
    if normalized != path or any(part == "." for part in parsed.parts):
        raise ValueError(f"{field_name} must be normalized.")
    return normalized


def _validate_model_path_argument(
    argument: str,
    *,
    authority: DockerCodingCommandAuthority,
) -> None:
    path_text, separator, node_id = argument.partition("::")
    if any(character in path_text for character in _MODEL_CONTROL_CHARACTERS):
        raise ValueError("Structured command path contains expansion syntax.")
    if separator:
        if not authority.allow_pytest_node_ids or not node_id:
            raise ValueError("Structured command path does not allow a node selector.")
        if any(
            not part or re.fullmatch(r"[A-Za-z0-9_.\[\]-]+", part) is None
            for part in node_id.split("::")
        ):
            raise ValueError("Structured command node selector is invalid.")
    normalized = _workspace_relative_path(
        path_text,
        field_name="structured command path",
        allow_root=False,
    )
    parsed = PurePosixPath(normalized)
    if authority.positional_path_prefixes and not any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in authority.positional_path_prefixes
    ):
        raise ValueError("Structured command path is outside its admitted scope.")
    if authority.positional_path_suffixes and parsed.suffix not in set(
        authority.positional_path_suffixes
    ):
        raise ValueError("Structured command path has an unsupported suffix.")


def _fingerprint(value: object, field_name: str) -> str:
    return "sha256:" + sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


__all__ = [
    "DOCKER_CODING_COMMAND_AUTHORITY_SCHEMA",
    "DOCKER_CODING_TOOLCHAIN_PROFILE_SCHEMA",
    "DockerCodingAdmissionProbe",
    "DockerCodingCommandAuthority",
    "DockerCodingDependencyInput",
    "DockerCodingFixedEnvironmentVariable",
    "DockerCodingToolchainError",
    "DockerCodingToolchainProfile",
    "docker_coding_toolchain_runner_admission_failure",
    "verify_docker_coding_toolchain_dependencies",
    "verify_local_docker_coding_toolchain_dependencies",
]

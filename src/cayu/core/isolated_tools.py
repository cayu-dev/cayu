"""Public contracts for hard-bounded host-side tool execution."""

from __future__ import annotations

import re
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Final, Literal, cast, final

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    copy_bounded_durable_json_value,
    copy_durable_json_object,
    freeze_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec

ISOLATED_TOOL_PROTOCOL_NAME: Final = "cayu.isolated-tool"
ISOLATED_TOOL_PROTOCOL_VERSION: Final = 1

DEFAULT_ISOLATED_TOOL_MAX_REQUEST_BYTES: Final = 1 << 20
DEFAULT_ISOLATED_TOOL_MAX_RESPONSE_BYTES: Final = 1 << 20
DEFAULT_ISOLATED_TOOL_MAX_STDOUT_BYTES: Final = 64 << 10
DEFAULT_ISOLATED_TOOL_MAX_STDERR_BYTES: Final = 64 << 10
MAX_ISOLATED_TOOL_MESSAGE_BYTES: Final = 16 << 20
MAX_ISOLATED_TOOL_DIAGNOSTIC_STREAM_BYTES: Final = 1 << 20
MAX_ISOLATED_TOOL_JSON_NODES: Final = 100_000
MAX_ISOLATED_TOOL_DEADLINE_SECONDS: Final = 24 * 60 * 60
MAX_ISOLATED_TOOL_CLEANUP_GRACE_SECONDS: Final = 30
MAX_ISOLATED_TOOL_CONTEXT_FIELDS: Final = 16
MAX_ISOLATED_TOOL_METADATA_KEYS: Final = 64
MAX_ISOLATED_TOOL_ENVIRONMENT_ITEMS: Final = 128

_IMPORT_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z", flags=re.ASCII)
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z", flags=re.ASCII)
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "LC_ALL",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONWARNINGS",
        "__CF_USER_TEXT_ENCODING",
    }
)
_FORBIDDEN_ENVIRONMENT_PREFIXES = ("DYLD_", "LD_", "PYTHON")


class ToolTimeoutStrength(StrEnum):
    """Truthful strength of one configured tool deadline."""

    NONE = "none"
    COOPERATIVE_IN_PROCESS = "cooperative_in_process"
    HARD_PROCESS_DEADLINE = "hard_process_deadline"


class ToolExecutionBoundary(StrEnum):
    """Runtime boundary that owns one tool implementation."""

    IN_PROCESS = "in_process"
    POSIX_PROCESS = "posix_process"


class ProcessIsolatedToolFactoryRef(BaseModel):
    """Stable import reference for a child-only isolated-tool factory."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    module: str = Field(max_length=512)
    qualname: str = Field(max_length=512)
    identity: ExecutionProfileBehaviorIdentity

    @field_validator("module", "qualname")
    @classmethod
    def validate_import_reference(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        parts = value.split(".")
        if any(not _IMPORT_SEGMENT.fullmatch(part) for part in parts):
            raise ValueError(f"{info.field_name} must be a dotted Python identifier.")
        if info.field_name == "module" and value == "__main__":
            raise ValueError("module cannot be __main__.")
        if info.field_name == "qualname" and "<locals>" in value:
            raise ValueError("qualname cannot refer to a process-local closure.")
        return value

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: object) -> object:
        if isinstance(value, ExecutionProfileBehaviorIdentity):
            copied = copy_execution_profile_behavior_identity(value)
            if copied is None:  # pragma: no cover - narrowed by isinstance
                raise AssertionError("Factory identity copy unexpectedly returned None.")
            return copied.model_dump(mode="python")
        return value


class ProcessIsolatedToolLimits(BaseModel):
    """Hard deadline and bounded transport limits for one isolated tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    deadline_seconds: float
    term_grace_seconds: float = 1.0
    kill_grace_seconds: float = 1.0
    max_request_bytes: StrictInt = Field(
        default=DEFAULT_ISOLATED_TOOL_MAX_REQUEST_BYTES,
        ge=1024,
        le=MAX_ISOLATED_TOOL_MESSAGE_BYTES,
    )
    max_response_bytes: StrictInt = Field(
        default=DEFAULT_ISOLATED_TOOL_MAX_RESPONSE_BYTES,
        ge=1024,
        le=MAX_ISOLATED_TOOL_MESSAGE_BYTES,
    )
    max_stdout_bytes: StrictInt = Field(
        default=DEFAULT_ISOLATED_TOOL_MAX_STDOUT_BYTES,
        ge=0,
        le=MAX_ISOLATED_TOOL_DIAGNOSTIC_STREAM_BYTES,
    )
    max_stderr_bytes: StrictInt = Field(
        default=DEFAULT_ISOLATED_TOOL_MAX_STDERR_BYTES,
        ge=0,
        le=MAX_ISOLATED_TOOL_DIAGNOSTIC_STREAM_BYTES,
    )

    @field_validator("deadline_seconds", mode="before")
    @classmethod
    def validate_deadline(cls, value: object) -> float:
        return _bounded_seconds(
            value,
            "deadline_seconds",
            maximum=MAX_ISOLATED_TOOL_DEADLINE_SECONDS,
            allow_zero=False,
        )

    @field_validator("term_grace_seconds", mode="before")
    @classmethod
    def validate_term_grace(cls, value: object) -> float:
        return _bounded_seconds(
            value,
            "term_grace_seconds",
            maximum=MAX_ISOLATED_TOOL_CLEANUP_GRACE_SECONDS,
            allow_zero=True,
        )

    @field_validator("kill_grace_seconds", mode="before")
    @classmethod
    def validate_kill_grace(cls, value: object) -> float:
        return _bounded_seconds(
            value,
            "kill_grace_seconds",
            maximum=MAX_ISOLATED_TOOL_CLEANUP_GRACE_SECONDS,
            allow_zero=False,
        )


IsolatedToolContextField = Literal[
    "session_id",
    "agent_name",
    "environment_name",
    "causal_budget_id",
    "workspace_id",
    "artifact_store_id",
    "idempotency_key",
]


class ProcessIsolatedToolContextProjection(BaseModel):
    """Explicit JSON-only projection of runtime context exposed to the child."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    fields: tuple[IsolatedToolContextField, ...] = ()
    metadata_keys: tuple[str, ...] = ()

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value: object) -> tuple[object, ...]:
        return _bounded_unique_text_items(
            value,
            "fields",
            maximum=MAX_ISOLATED_TOOL_CONTEXT_FIELDS,
        )

    @field_validator("metadata_keys", mode="before")
    @classmethod
    def normalize_metadata_keys(cls, value: object) -> tuple[str, ...]:
        items = _bounded_unique_text_items(
            value,
            "metadata_keys",
            maximum=MAX_ISOLATED_TOOL_METADATA_KEYS,
        )
        return tuple(require_durable_clean_nonblank(item, "metadata_keys item") for item in items)


class ProcessIsolatedToolContext(BaseModel):
    """Portable context received by an isolated tool handler."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    causal_budget_id: str | None = None
    workspace_id: str | None = None
    artifact_store_id: str | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "session_id",
        "agent_name",
        "environment_name",
        "causal_budget_id",
        "workspace_id",
        "artifact_store_id",
        "idempotency_key",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "isolated_tool_context.metadata")


@final
class ProcessIsolatedTool(Tool):
    """Explicit host-side tool executed by Cayu in a disposable POSIX process."""

    _factory: ProcessIsolatedToolFactoryRef
    _limits: ProcessIsolatedToolLimits
    _context_projection: ProcessIsolatedToolContextProjection
    _factory_config: Any
    _environment: MappingProxyType[str, str]

    def __init__(
        self,
        spec: ToolSpec,
        *,
        factory: ProcessIsolatedToolFactoryRef,
        limits: ProcessIsolatedToolLimits,
        factory_config: dict[str, Any] | None = None,
        context_projection: ProcessIsolatedToolContextProjection | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        super().__init__(spec)
        if self.spec.workspace_mutation:
            raise ValueError(
                "Process-isolated tools cannot request Cayu workspace mutation authority."
            )
        if type(factory) is not ProcessIsolatedToolFactoryRef:
            raise TypeError("factory must be a ProcessIsolatedToolFactoryRef.")
        if type(limits) is not ProcessIsolatedToolLimits:
            raise TypeError("limits must be ProcessIsolatedToolLimits.")
        if context_projection is None:
            context_projection = ProcessIsolatedToolContextProjection()
        if type(context_projection) is not ProcessIsolatedToolContextProjection:
            raise TypeError(
                "context_projection must be a ProcessIsolatedToolContextProjection or None."
            )
        object.__setattr__(
            self,
            "_factory",
            ProcessIsolatedToolFactoryRef.model_validate(factory.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "_limits",
            ProcessIsolatedToolLimits.model_validate(limits.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "_context_projection",
            ProcessIsolatedToolContextProjection.model_validate(
                context_projection.model_dump(mode="python")
            ),
        )
        if factory_config is not None and type(factory_config) is not dict:
            raise TypeError("factory_config must be a JSON object or None.")
        copied_factory_config = copy_bounded_durable_json_value(
            {} if factory_config is None else factory_config,
            "factory_config",
            max_bytes=limits.max_request_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
        if type(copied_factory_config) is not dict:  # pragma: no cover - checked above
            raise AssertionError("Factory configuration copy did not produce an object.")
        object.__setattr__(self, "_factory_config", freeze_json_value(copied_factory_config))
        if environment is not None and type(environment) is not dict:
            raise TypeError("environment must be a string map or None.")
        object.__setattr__(
            self,
            "_environment",
            MappingProxyType(
                _copy_isolated_environment(
                    {} if environment is None else environment,
                    max_bytes=limits.max_request_bytes,
                )
            ),
        )
        copy_bounded_durable_json_value(
            {
                "factory": self._factory.model_dump(mode="json"),
                "limits": self._limits.model_dump(mode="json"),
                "factory_config": self._factory_config,
                "context_projection": self._context_projection.model_dump(mode="json"),
                "environment": dict(self._environment),
            },
            "isolated_tool_static_configuration",
            max_bytes=limits.max_request_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )

    @property
    def factory(self) -> ProcessIsolatedToolFactoryRef:
        return self._factory

    @property
    def limits(self) -> ProcessIsolatedToolLimits:
        return self._limits

    @property
    def context_projection(self) -> ProcessIsolatedToolContextProjection:
        return self._context_projection

    def factory_config_copy(self) -> dict[str, Any]:
        return copy_durable_json_object(self._factory_config, "factory_config")

    def environment_copy(self) -> dict[str, str]:
        return dict(self._environment)

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        raise RuntimeError(
            "ProcessIsolatedTool requires Cayu's runtime-owned isolated execution boundary."
        )


def _bounded_seconds(
    value: object,
    field_name: str,
    *,
    maximum: float,
    allow_zero: bool,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(cast("int | float", value))
    minimum_ok = normalized >= 0 if allow_zero else normalized > 0
    if not isfinite(normalized) or not minimum_ok or normalized > maximum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be finite, {qualifier}, and at most {maximum}.")
    return normalized


def _bounded_unique_text_items(
    value: object,
    field_name: str,
    *,
    maximum: int,
) -> tuple[Any, ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{field_name} must be a list or tuple of strings.")
    values = cast("list[Any] | tuple[Any, ...]", value)
    if len(values) > maximum:
        raise ValueError(f"{field_name} cannot contain more than {maximum} values.")
    items = tuple(values)
    if any(type(item) is not str for item in items):
        raise TypeError(f"{field_name} must contain only strings.")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} cannot contain duplicate values.")
    return tuple(sorted(items))


def _copy_isolated_environment(
    value: dict[str, str],
    *,
    max_bytes: int,
) -> dict[str, str]:
    if len(value) > MAX_ISOLATED_TOOL_ENVIRONMENT_ITEMS:
        raise ValueError(
            f"environment cannot contain more than {MAX_ISOLATED_TOOL_ENVIRONMENT_ITEMS} values."
        )
    copied: dict[str, str] = {}
    for name, item in value.items():
        if type(name) is not str or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("environment names must be portable identifiers.")
        if name in _FORBIDDEN_ENVIRONMENT_NAMES or name.startswith(_FORBIDDEN_ENVIRONMENT_PREFIXES):
            raise ValueError("environment variables cannot alter the child interpreter.")
        if type(item) is not str:
            raise TypeError("environment values must be strings.")
        copied[name] = require_durable_text(item, "environment value")
    sorted_environment = dict(sorted(copied.items()))
    bounded = copy_bounded_durable_json_value(
        sorted_environment,
        "environment",
        max_bytes=max_bytes,
        max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
    )
    if type(bounded) is not dict:  # pragma: no cover - local construction invariant
        raise AssertionError("Environment copy did not produce an object.")
    return bounded


__all__ = [
    "DEFAULT_ISOLATED_TOOL_MAX_REQUEST_BYTES",
    "DEFAULT_ISOLATED_TOOL_MAX_RESPONSE_BYTES",
    "DEFAULT_ISOLATED_TOOL_MAX_STDERR_BYTES",
    "DEFAULT_ISOLATED_TOOL_MAX_STDOUT_BYTES",
    "ISOLATED_TOOL_PROTOCOL_NAME",
    "ISOLATED_TOOL_PROTOCOL_VERSION",
    "MAX_ISOLATED_TOOL_JSON_NODES",
    "MAX_ISOLATED_TOOL_MESSAGE_BYTES",
    "ProcessIsolatedTool",
    "ProcessIsolatedToolContext",
    "ProcessIsolatedToolContextProjection",
    "ProcessIsolatedToolFactoryRef",
    "ProcessIsolatedToolLimits",
    "ToolExecutionBoundary",
    "ToolTimeoutStrength",
]

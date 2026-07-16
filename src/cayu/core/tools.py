from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    computed_field,
    field_serializer,
    field_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank


class ToolEffect(StrEnum):
    """Declared side-effect semantics for a tool execution.

    Classify what replay can do to externally meaningful durable state. ``NONE``
    does not mutate it, ``IDEMPOTENT`` may mutate it but a stable downstream
    identity or equivalent contract collapses replay, and ``EXTERNAL`` has a
    non-idempotent or outcome-ambiguous durable mutation. Transport, billing,
    observability, and names such as "read" do not determine the value.

    The runtime uses this as execution metadata, not as an authorization
    decision: policy still decides whether a call may run. Run
    ``cayu guide tool-effects`` for the canonical decision table.
    """

    NONE = "none"
    IDEMPOTENT = "idempotent"
    EXTERNAL = "external"


@dataclass(frozen=True)
class FrozenMapping(Mapping[str, Any]):
    """Read-only mapping used for internal immutable schema storage."""

    _items: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMapping:
        return self

    def __repr__(self) -> str:
        return repr({key: value for key, value in self._items})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(tuple((key, _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _mutable_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: _mutable_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_value(item) for item in value]
    if type(value) is dict:
        return {key: _mutable_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_mutable_value(item) for item in value]
    if isinstance(value, Mapping | list):
        raise ValueError("Tool input_schema must contain JSON-compatible values.")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError("Tool input_schema must contain JSON-compatible values.")


class _ToolSpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    parallel_safe: StrictBool = True
    effect: ToolEffect = ToolEffect.EXTERNAL

    @field_validator("input_schema", mode="before")
    @classmethod
    def copy_input_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "input_schema")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parallel_safe: StrictBool = True
    effect: ToolEffect = ToolEffect.EXTERNAL
    _input_schema: Any = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        parsed = _ToolSpecInput.model_validate(data)
        super().__init__(
            name=parsed.name,
            description=parsed.description,
            parallel_safe=parsed.parallel_safe,
            effect=parsed.effect,
        )
        object.__setattr__(self, "_input_schema", _freeze_value(parsed.input_schema))

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        return _mutable_value(self._input_schema)

    @field_serializer("effect")
    def serialize_effect(self, value: ToolEffect) -> str:
        return value.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolSpec):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> ToolSpec:
        data = self.model_dump()
        if update:
            data.update(update)
        return type(self)(**data)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> ToolSpec:
        # ToolSpec is frozen and stores its schema immutably; sharing is safe.
        return self

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        schema.setdefault("$defs", {})["ToolEffect"] = {
            "enum": [effect.value for effect in ToolEffect],
            "title": "ToolEffect",
            "type": "string",
        }
        schema.setdefault("properties", {})["input_schema"] = {
            "additionalProperties": True,
            "default": {},
            "title": "Input Schema",
            "type": "object",
        }
        schema["required"] = [
            field for field in schema.get("required", []) if field != "input_schema"
        ]
        return schema


class ToolResult(BaseModel):
    """Result returned from a tool call.

    `content` is the model-facing summary. `structured` and `artifacts` are
    for dashboards, workflows, storage, and downstream tools.

    Frozen: results own their payloads (copied at construction) and are
    treated as read-only once returned from a tool.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = ""
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    is_error: StrictBool = False

    @field_validator("structured", "artifacts", mode="before")
    @classmethod
    def copy_result_data(cls, value, info):
        return copy_json_value(value, info.field_name)


_TOOL_POLICY_DENIAL_SOURCE = "tool_policy"
_COMMAND_POLICY_DENIAL_SOURCE = "command_policy"
_POLICY_DENIAL_TEXT_MAX_BYTES = 4 * 1024
_POLICY_DENIAL_TRUNCATION_MARKER = "\n[policy denial reason truncated]"


def _bound_policy_denial_text(value: str) -> str:
    """Bound trusted policy diagnostics without splitting a Unicode scalar."""

    value = require_nonblank(value, "policy denial text")
    # Policy implementations are application code and can construct a string with
    # lone surrogates. Keep the refusal fail-closed while making its durable form
    # valid UTF-8 rather than turning the denial into an operational failure.
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= _POLICY_DENIAL_TEXT_MAX_BYTES:
        return encoded.decode("utf-8")
    marker = _POLICY_DENIAL_TRUNCATION_MARKER.encode("utf-8")
    prefix = encoded[: _POLICY_DENIAL_TEXT_MAX_BYTES - len(marker)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + _POLICY_DENIAL_TRUNCATION_MARKER


def _bound_policy_denial_result(result: ToolResult) -> ToolResult:
    """Bound the model-facing and structured reason fields of a denial result."""

    if type(result) is not ToolResult:
        raise TypeError("Policy denial results must be ToolResult instances.")
    structured = result.structured
    if structured is not None and type(structured.get("reason")) is str:
        structured = {
            **structured,
            "reason": _bound_policy_denial_text(structured["reason"]),
        }
    return ToolResult(
        content=_bound_policy_denial_text(result.content),
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.is_error,
    )


@dataclass(frozen=True)
class _PolicyDenialSignal:
    """Runtime-only attribution recorded alongside an ordinary tool result."""

    source: object
    denied_by: str
    decision: str
    reason: str
    result: ToolResult


@runtime_checkable
class WorkspaceHandle(Protocol):
    """Structural contract for the workspace handed to tools.

    Mirrors ``cayu.workspaces.Workspace`` without importing it, so custom
    workspaces only need to implement the file operations tools rely on.
    """

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> Any: ...

    async def write_bytes(self, path: str, content: bytes) -> None: ...

    async def delete(self, path: str) -> None: ...

    async def list(self, pattern: str = "**/*", *, limit: int | None = None) -> Any: ...


@runtime_checkable
class ArtifactStoreHandle(Protocol):
    """Structural contract for the artifact store handed to tools.

    Mirrors ``cayu.artifacts.ArtifactStore``.
    """

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any) -> Any: ...

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None) -> Any: ...

    async def list(self, **kwargs: Any) -> Any: ...

    async def delete(self, artifact_id: str) -> None: ...


@runtime_checkable
class RunnerHandle(Protocol):
    """Structural contract for the command runner handed to tools.

    Mirrors ``cayu.runners.Runner``.
    """

    async def exec(self, command: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class VaultHandle(Protocol):
    """Structural contract for the vault handed to tools.

    Mirrors ``cayu.vaults.Vault``.
    """

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> Any: ...

    async def resolve(self, ref: Any, *, scope: dict[str, Any] | None = None) -> Any: ...


@runtime_checkable
class CredentialProxyHandle(Protocol):
    """Structural contract for the credential proxy handed to tools.

    Mirrors ``cayu.proxies.CredentialProxy``.
    """

    async def resolve(self, ref: Any, *, scope: dict[str, Any] | None = None) -> Any: ...

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: Any = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class KnowledgeStoreHandle(Protocol):
    """Structural contract for the knowledge store handed to tools.

    Deliberately the minimal *read* surface (search/list/read) so read-only
    stores can back read-path knowledge tools. Full read/write stores such as
    ``cayu.storage.memory.KnowledgeStore`` are a superset and also satisfy it;
    write-path tools check for their extra methods at call time.
    """

    async def search(self, *args: Any, **kwargs: Any) -> Any: ...

    async def list_entries(self, *args: Any, **kwargs: Any) -> Any: ...

    async def read_chunks(self, *args: Any, **kwargs: Any) -> Any: ...


class ToolContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    session_id: str
    agent_name: str | None = None
    environment_name: str | None = None
    causal_budget_id: str | None = None
    workspace_id: str | None = None
    artifact_store_id: str | None = None
    idempotency_key: str | None = None
    workspace: WorkspaceHandle | None = Field(default=None, exclude=True)
    artifact_store: ArtifactStoreHandle | None = Field(default=None, exclude=True)
    runner: RunnerHandle | None = Field(default=None, exclude=True)
    vault: VaultHandle | None = Field(default=None, exclude=True)
    proxy: CredentialProxyHandle | None = Field(default=None, exclude=True)
    knowledge_store: KnowledgeStoreHandle | None = Field(default=None, exclude=True)
    mcp_servers: tuple[Any, ...] = Field(default_factory=tuple, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)
    _policy_denials: list[_PolicyDenialSignal] = PrivateAttr(default_factory=list)

    def _record_policy_denial(
        self,
        *,
        source: object,
        denied_by: str,
        decision: str,
        reason: str,
        result: ToolResult,
    ) -> None:
        """Record trusted control metadata without changing the public ToolResult contract."""

        if type(result) is not ToolResult:
            raise TypeError("Policy denial results must be ToolResult instances.")
        self._policy_denials.append(
            _PolicyDenialSignal(
                source=source,
                denied_by=require_clean_nonblank(denied_by, "denied_by"),
                decision=require_clean_nonblank(decision, "decision"),
                reason=require_nonblank(reason, "reason"),
                result=result,
            )
        )

    def _policy_denial_for(self, source: object) -> _PolicyDenialSignal | None:
        for denial in reversed(self._policy_denials):
            if denial.source is source:
                return denial
        return None

    def _discard_policy_denials_for(self, source: object) -> None:
        self._policy_denials[:] = [
            denial for denial in self._policy_denials if denial.source is not source
        ]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("session_id")
    @classmethod
    def validate_nonblank_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "agent_name",
        "environment_name",
        "causal_budget_id",
        "workspace_id",
        "artifact_store_id",
        "idempotency_key",
    )
    @classmethod
    def validate_optional_nonblank_ids(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def copy_mcp_servers(cls, value):
        if value is None:
            return ()
        if isinstance(value, str | bytes):
            raise TypeError("mcp_servers must be an iterable.")
        try:
            return tuple(value)
        except TypeError as exc:
            raise TypeError("mcp_servers must be an iterable.") from exc


class Tool(ABC):
    """Base class for framework-native tools."""

    spec: ToolSpec

    def __init__(self, spec: ToolSpec | None = None) -> None:
        if spec is not None:
            self.spec = spec
        else:
            class_spec = getattr(type(self), "spec", None)
            if isinstance(class_spec, ToolSpec):
                # ToolSpec is frozen and deeply immutable; instances can share
                # the class-level spec without copying.
                self.spec = class_spec
        self._validate_spec()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def schema(self) -> dict[str, Any]:
        # `input_schema` already materializes a fresh mutable dict from the
        # frozen storage; wrapping it in another copy would be redundant.
        return self.spec.input_schema

    def _validate_spec(self) -> None:
        spec = getattr(self, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise TypeError(
                f"{self.__class__.__name__} must define `spec = ToolSpec(...)` "
                "or pass a ToolSpec to Tool.__init__()."
            )
        if not spec.name.strip():
            raise ValueError("Tool spec name cannot be blank.")

    @abstractmethod
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Execute a tool call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    computed_field,
    field_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank


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
    _input_schema: Any = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        parsed = _ToolSpecInput.model_validate(data)
        super().__init__(name=parsed.name, description=parsed.description)
        object.__setattr__(self, "_input_schema", _freeze_value(parsed.input_schema))

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        return _mutable_value(self._input_schema)

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

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
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
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    is_error: StrictBool = False

    @field_validator("structured", "artifacts", mode="before")
    @classmethod
    def copy_result_data(cls, value, info):
        return copy_json_value(value, info.field_name)


class ToolContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    agent_name: str | None = None
    environment_name: str | None = None
    causal_budget_id: str | None = None
    workspace_id: str | None = None
    artifact_store_id: str | None = None
    workspace: Any = Field(default=None, exclude=True)
    artifact_store: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    vault: Any = Field(default=None, exclude=True)
    proxy: Any = Field(default=None, exclude=True)
    knowledge_store: Any = Field(default=None, exclude=True)
    mcp_servers: tuple[Any, ...] = Field(default_factory=tuple, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
            if type(class_spec) is ToolSpec:
                self.spec = class_spec.model_copy(deep=True)
        self._validate_spec()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def schema(self) -> dict[str, Any]:
        return _mutable_value(self.spec.input_schema)

    def _validate_spec(self) -> None:
        spec = getattr(self, "spec", None)
        if type(spec) is not ToolSpec:
            raise TypeError(
                f"{self.__class__.__name__} must define `spec = ToolSpec(...)` "
                "or pass a ToolSpec to Tool.__init__()."
            )
        if not spec.name.strip():
            raise ValueError("Tool spec name cannot be blank.")

    @abstractmethod
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Execute a tool call."""

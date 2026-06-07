from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_clean_nonblank_keys,
    require_nonblank,
)
from cayu.vaults import SecretRef, copy_secret_ref


class McpServerSpec(BaseModel):
    """Configuration for an external MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    secret_env: dict[str, SecretRef] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    secret_headers: dict[str, SecretRef] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("secret_env", "secret_headers", mode="before")
    @classmethod
    def validate_secret_config_keys(cls, value, info):
        return require_clean_nonblank_keys(value, info.field_name)

    @field_validator("secret_env", "secret_headers")
    @classmethod
    def copy_secret_config_data(cls, value):
        return {key: copy_secret_ref(ref) for key, ref in value.items()}

    @field_validator("env", "headers", "metadata", mode="before")
    @classmethod
    def copy_json_config_data(cls, value, info):
        copied = copy_json_value(value, info.field_name)
        if info.field_name in {"env", "headers"}:
            require_clean_nonblank_keys(copied, info.field_name)
        return copied

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("url")
    @classmethod
    def validate_nonblank_url(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("command")
    @classmethod
    def validate_command_items(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        for item in value:
            require_nonblank(item, "command")
        return value

    @model_validator(mode="after")
    def validate_transport(self) -> McpServerSpec:
        if bool(self.command) == bool(self.url):
            raise ValueError("MCP server must define exactly one of command or url.")
        return self


class McpInitializeResult(BaseModel):
    """Server metadata returned by MCP initialize."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str
    server_name: str | None = None
    server_version: str | None = None
    instructions: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("server_name", "server_version", "instructions")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("capabilities", mode="before")
    @classmethod
    def copy_capabilities(cls, value):
        return copy_json_value(value, "capabilities")


class McpToolDefinition(BaseModel):
    """Tool definition advertised by an MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("description must be a string.")
        return value

    @field_validator("input_schema", "annotations", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_json_value(value, info.field_name)


class McpToolResult(BaseModel):
    """Result returned by an MCP tools/call request."""

    model_config = ConfigDict(extra="forbid")

    content: list[dict[str, Any]] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: StrictBool = False

    @field_validator("content", "structured_content", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_json_value(value, info.field_name)


class McpResourceDefinition(BaseModel):
    """Resource definition advertised by an MCP server."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("name", "description", "mime_type")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value):
        return copy_json_value(value, "metadata")


class McpResourceResult(BaseModel):
    """Result returned by an MCP resources/read request."""

    model_config = ConfigDict(extra="forbid")

    contents: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("contents", mode="before")
    @classmethod
    def copy_contents(cls, value):
        return copy_json_value(value, "contents")


class McpSession(ABC):
    """Initialized connection to one MCP server."""

    @property
    @abstractmethod
    def initialize_result(self) -> McpInitializeResult:
        """Metadata returned by MCP initialize."""

    @abstractmethod
    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        """Return tools advertised by the server."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        """Call one server tool."""

    @abstractmethod
    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        """Return resources advertised by the server."""

    @abstractmethod
    async def read_resource(self, uri: str) -> McpResourceResult:
        """Read one server resource."""

    @abstractmethod
    async def close(self) -> None:
        """Close the server connection."""


class McpClient(ABC):
    """Factory for initialized MCP sessions."""

    @abstractmethod
    async def connect(self, server: McpServerSpec) -> McpSession:
        """Connect to one MCP server."""

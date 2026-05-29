from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    copy_json_value,
    require_nonblank,
    require_nonblank_keys,
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
        return require_nonblank_keys(value, info.field_name)

    @field_validator("secret_env", "secret_headers")
    @classmethod
    def copy_secret_config_data(cls, value):
        return {key: copy_secret_ref(ref) for key, ref in value.items()}

    @field_validator("env", "headers", "metadata", mode="before")
    @classmethod
    def copy_json_config_data(cls, value, info):
        copied = copy_json_value(value, info.field_name)
        if info.field_name in {"env", "headers"}:
            require_nonblank_keys(copied, info.field_name)
        return copied

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("url")
    @classmethod
    def validate_nonblank_url(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("command")
    @classmethod
    def validate_command_items(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        for item in value:
            require_nonblank(item, "command")
        return value

    @model_validator(mode="after")
    def validate_transport(self) -> "McpServerSpec":
        if bool(self.command) == bool(self.url):
            raise ValueError("MCP server must define exactly one of command or url.")
        return self

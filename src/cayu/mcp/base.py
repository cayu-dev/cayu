from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cayu.vaults import SecretRef


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

    @model_validator(mode="after")
    def validate_transport(self) -> "McpServerSpec":
        if bool(self.command) == bool(self.url):
            raise ValueError("MCP server must define exactly one of command or url.")
        return self

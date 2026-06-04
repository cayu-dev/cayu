from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.runtime.sessions import Session


class ToolPolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolPolicyRequest(BaseModel):
    """Input passed to a tool policy before a registered tool executes."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    environment_name: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name", "tool_call_id")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("environment_name", "workspace_id")
    @classmethod
    def validate_optional_names(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)


class ToolPolicyResult(BaseModel):
    """Authorization decision for one tool call."""

    model_config = ConfigDict(extra="forbid")

    decision: ToolPolicyDecision
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class ToolPolicy(ABC):
    """Authorizes registered tool calls before execution."""

    @abstractmethod
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        """Return whether this tool call may execute."""


class AllowAllToolPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class StaticToolPolicy(ToolPolicy):
    """Simple allow/deny policy for static tool scopes.

    Deny rules win over allow rules. When `allow` is omitted, all tools are
    allowed except explicitly denied tools. When `allow` is provided, only those
    tools are allowed.
    """

    def __init__(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
    ) -> None:
        self.allow = _copy_tool_name_set(allow, "allow") if allow is not None else None
        self.deny = _copy_tool_name_set(deny, "deny") if deny is not None else frozenset()

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name in self.deny:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"Tool denied by policy: {request.tool_name}",
            )

        if self.allow is not None and request.tool_name not in self.allow:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"Tool not allowed by policy: {request.tool_name}",
            )

        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def _copy_tool_name_set(value: Iterable[str], field_name: str) -> frozenset[str]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of tool names.")
    try:
        names = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of tool names.") from exc

    copied: set[str] = set()
    for index, name in enumerate(names):
        copied.add(require_nonblank(name, f"{field_name}[{index}]"))
    return frozenset(copied)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cayu.core.agents import AgentSpec
from cayu.core.events import Event
from cayu.core.tools import Tool, ToolResult
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider
from cayu.runtime.approvals import PendingToolApproval
from cayu.runtime.context import ContextPolicy
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.tool_policy import ToolPolicy, ToolPolicyResult


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]


@dataclass(frozen=True)
class RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    context_policy: ContextPolicy
    tool_policy: ToolPolicy
    runtime_hooks: tuple[RuntimeHook, ...]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: dict[str, Any]
    tool: Tool


@dataclass(frozen=True)
class RegisteredProvider:
    name: str
    provider: ModelProvider


@dataclass(frozen=True)
class RegisteredEnvironment:
    spec: EnvironmentSpec
    environment: Environment


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallOutcome:
    call: ToolCallRequest
    result: ToolResult


@dataclass(frozen=True)
class ToolCallPolicyOutcome:
    call: ToolCallRequest
    result: ToolPolicyResult | None


@dataclass(frozen=True)
class ToolRoundPolicyPlan:
    outcomes: list[ToolCallPolicyOutcome]
    pending_approval: tuple[PendingToolApproval, Event] | None

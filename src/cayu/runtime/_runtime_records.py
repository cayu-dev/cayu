from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu.core.agents import AgentSpec
from cayu.core.tools import Tool, ToolEffect, ToolResult
from cayu.environments import BoundWorkspace, Environment, EnvironmentFactory, EnvironmentSpec
from cayu.providers import ModelProvider
from cayu.runtime.context import ContextPolicy
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.tool_policy import ToolPolicy, ToolPolicyResult

if TYPE_CHECKING:
    from cayu.runtime.loop_policies import LoopPolicy


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]


@dataclass(frozen=True)
class RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    context_policy: ContextPolicy
    context_overflow_policy: ContextPolicy | None
    tool_policy: ToolPolicy
    runtime_hooks: tuple[RuntimeHook, ...]
    loop_policies: tuple[LoopPolicy, ...]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: dict[str, Any]
    parallel_safe: bool
    effect: ToolEffect
    tool: Tool


@dataclass(frozen=True)
class RegisteredProvider:
    name: str
    provider: ModelProvider
    model_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisteredEnvironment:
    spec: EnvironmentSpec
    environment: Environment
    factory: EnvironmentFactory | None = None
    bound_workspace: BoundWorkspace | None = None
    binding_payload: dict[str, Any] | None = None


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
class PendingToolApprovalPlan:
    call: ToolCallRequest
    calls: list[ToolCallRequest]
    policy_outcomes: list[ToolCallPolicyOutcome]
    policy_result: ToolPolicyResult


@dataclass(frozen=True)
class ToolRoundPolicyPlan:
    outcomes: list[ToolCallPolicyOutcome]
    pending_approval: PendingToolApprovalPlan | None

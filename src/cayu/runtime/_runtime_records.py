from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu._validation import require_durable_clean_nonblank
from cayu._workspace_mutation import WorkspaceMutationProcessFence
from cayu.core.agents import AgentSpec
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import Tool, ToolEffect, ToolResult
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ExecutionRequirements,
)
from cayu.providers import ModelProvider, UsageDialect
from cayu.runtime._child_session_identity import ChildSessionRecoveryMatcher
from cayu.runtime._policy_evidence import ToolPolicyEvidence
from cayu.runtime.context import ContextPolicy
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.tool_policy import ToolPolicy, ToolPolicyResult

if TYPE_CHECKING:
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.runtime.tool_exposure import RegisteredToolCapability


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]


@dataclass(frozen=True)
class RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    tool_capabilities: tuple[RegisteredToolCapability, ...]
    context_policy: ContextPolicy
    context_policy_execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    context_overflow_policy: ContextPolicy | None
    context_overflow_policy_execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    tool_policy: ToolPolicy
    tool_policy_execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    runtime_hooks: tuple[RegisteredRuntimeHook, ...]
    loop_policies: tuple[LoopPolicy, ...]
    loop_policy_execution_profile_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...]
    execution_requirements: ExecutionRequirements
    context_behavior_execution_profile_identities: Mapping[
        int, ExecutionProfileBehaviorIdentity | None
    ] = field(default_factory=dict)
    registration_source: str | None = None
    registration_symbol: str | None = None


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: dict[str, Any]
    parallel_safe: bool
    effect: ToolEffect
    publish_arguments: bool
    workspace_mutation: bool
    execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    command_policy_execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    tool: Tool
    child_session_recovery: ChildSessionRecoveryMatcher | None = None


@dataclass(frozen=True)
class RegisteredRuntimeHook:
    """A runtime hook paired with its validated, registration-time identity."""

    name: str
    execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    hook: RuntimeHook

    def __post_init__(self) -> None:
        if not isinstance(self.hook, RuntimeHook):
            raise TypeError("Registered runtime hook must contain a RuntimeHook instance.")
        object.__setattr__(
            self,
            "name",
            require_durable_clean_nonblank(self.name, "runtime_hook.name"),
        )


@dataclass(frozen=True)
class RegisteredProvider:
    name: str
    provider: ModelProvider
    execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None
    model_patterns: tuple[str, ...] = ()
    registration_source: str | None = None
    registration_symbol: str | None = None
    usage_dialect: UsageDialect = UsageDialect.AUTO


@dataclass(frozen=True)
class RegisteredEnvironment:
    spec: EnvironmentSpec
    environment: Environment
    runner_execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None
    factory_execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None
    factory: EnvironmentFactory | None = None
    # Capability provenance survives factory materialization without retaining
    # the live factory as part of the session-owned environment lifecycle.
    factory_backed: bool = False
    bound_workspace: BoundWorkspace | None = None
    binding_payload: dict[str, Any] | None = None
    execution_candidate: str | None = None
    unclaimed_factory_result: EnvironmentFactoryResult | None = None
    preserve_factory_allocation: bool = False
    registration_source: str | None = None
    registration_symbol: str | None = None
    # Runtime-owned authority for one concrete environment/binding generation.
    # Factory materialization receives a fresh value; copies and the subsequent
    # binding transfer retain it. A fresh process therefore cannot accidentally
    # claim attribution for a prior process's live workspace handle.
    binding_generation_id: str = field(
        default_factory=lambda: f"wbind_{uuid4().hex}",
        compare=False,
    )
    workspace_mutation_fence: WorkspaceMutationProcessFence = field(
        default_factory=WorkspaceMutationProcessFence,
        compare=False,
        repr=False,
    )


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
    evidence: ToolPolicyEvidence


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
    # Active taint labels PER tool call (keyed by tool_call_id), captured at that call's authorize
    # point so it includes earlier same-round source labels. Tool execution and pause/resume reuse
    # the exact set the policy gated the call with, instead of rescanning or a pre-round snapshot.
    active_taint_labels: Mapping[str, frozenset[str]] = field(default_factory=dict)

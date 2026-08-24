from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu._validation import copy_durable_json_object, require_durable_clean_nonblank
from cayu._workspace_mutation import WorkspaceMutationProcessFence
from cayu.core.agents import AgentSpec
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import DurableToolRecovery, Tool, ToolEffect, ToolResult
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ExecutionRequirements,
)
from cayu.providers import ModelProvider, UsageDialect
from cayu.providers.hosted import OpenAIWebSearch
from cayu.runtime._child_session_identity import ChildSessionRecoveryMatcher
from cayu.runtime._policy_evidence import ToolPolicyEvidence
from cayu.runtime.context import ContextPolicy
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.tool_catalogue import CALL_TOOL_NAME
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_TRANSCRIPT_REFERENCE,
    RejectedTargetedToolInvocation,
    ResolvedTargetedToolInvocation,
    validate_targeted_tool_digest,
)
from cayu.runtime.tool_policy import ToolPolicy, ToolPolicyResult

if TYPE_CHECKING:
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.runtime.tool_catalogue import ToolCatalogSnapshot
    from cayu.runtime.tool_exposure import (
        RegisteredToolCapability,
        ResolvedToolExposure,
        ToolExposurePolicy,
    )


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    hosted_tools: tuple[OpenAIWebSearch, ...] = ()


@dataclass(frozen=True)
class RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    tool_catalogue: ToolCatalogSnapshot
    tool_capabilities: tuple[RegisteredToolCapability, ...]
    all_registered_tool_exposure: ResolvedToolExposure
    tool_exposure_policy: ToolExposurePolicy
    tool_exposure_policy_execution_profile_identity: ExecutionProfileBehaviorIdentity | None
    tool_gateway_enabled: bool
    hosted_tools: tuple[OpenAIWebSearch, ...]
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
    durable_tool_recovery: DurableToolRecovery | None = None


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
    # A successfully bound virtual-egress result remains the exact live owner
    # that can be parked at an invocation boundary for governed adoption.
    retained_factory_result: EnvironmentFactoryResult | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    preserve_factory_allocation: bool = False
    # Opaque identity for one recoverable process-external allocation. This is
    # stable across a factory reconnect but absent for process-local/static
    # environments, which must not claim browser continuity after restart.
    live_allocation_fingerprint: str | None = None
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
    targeted_tool_grant_id: str | None = None
    model_tool_name: str | None = None
    targeted_tool_invocation: ResolvedTargetedToolInvocation | None = None
    targeted_tool_rejection: RejectedTargetedToolInvocation | None = None

    def __post_init__(self) -> None:
        if self.targeted_tool_grant_id is not None:
            validate_targeted_tool_digest(
                self.targeted_tool_grant_id,
                "targeted_tool_grant_id",
            )
            if self.name != CALL_TOOL_NAME and self.model_tool_name != CALL_TOOL_NAME:
                raise ValueError("Targeted grant selection requires a call_tool model call.")
        if (self.targeted_tool_invocation is not None) and (
            self.targeted_tool_rejection is not None
        ):
            raise ValueError("A tool call cannot be both resolved and rejected.")
        if self.targeted_tool_invocation is not None:
            invocation = self.targeted_tool_invocation
            if (
                self.model_tool_name != invocation.model_tool_name
                or self.name != invocation.effective_tool_name
                or self.id != invocation.outer_tool_call_id
            ):
                raise ValueError("Resolved targeted invocation conflicts with its tool call.")
            if (
                self.targeted_tool_grant_id is not None
                and self.targeted_tool_grant_id != invocation.grant_id
            ):
                raise ValueError("Targeted grant selection conflicts with its resolved invocation.")
        if self.targeted_tool_rejection is not None:
            rejection = self.targeted_tool_rejection
            if self.model_tool_name != rejection.model_tool_name or self.name != "call_tool":
                raise ValueError("Rejected targeted invocation conflicts with its model call.")
        if self.model_tool_name is not None and (
            self.targeted_tool_invocation is None and self.targeted_tool_rejection is None
        ):
            raise ValueError("A model tool alias requires targeted invocation evidence.")

    @property
    def transcript_tool_name(self) -> str:
        return self.model_tool_name or self.name

    @property
    def transcript_arguments(self) -> dict[str, Any]:
        if self.targeted_tool_invocation is None:
            projected = copy_durable_json_object(self.arguments, "tool_call.arguments")
            if self.targeted_tool_grant_id is not None and "tool_ref" in projected:
                projected["tool_ref"] = TARGETED_TOOL_TRANSCRIPT_REFERENCE
            return projected
        return {
            "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
            "arguments": copy_durable_json_object(
                self.arguments,
                "tool_call.arguments",
            ),
        }


def copy_tool_call_request(
    value: ToolCallRequest,
    *,
    arguments: dict[str, Any] | None = None,
) -> ToolCallRequest:
    """Return a detached copy without losing gateway dual identity."""

    if type(value) is not ToolCallRequest:
        raise TypeError("value must be a ToolCallRequest.")
    return ToolCallRequest(
        id=value.id,
        name=value.name,
        arguments=copy_durable_json_object(
            value.arguments if arguments is None else arguments,
            "tool_call.arguments",
        ),
        targeted_tool_grant_id=value.targeted_tool_grant_id,
        model_tool_name=value.model_tool_name,
        targeted_tool_invocation=(
            None
            if value.targeted_tool_invocation is None
            else ResolvedTargetedToolInvocation.model_validate(
                value.targeted_tool_invocation.model_dump(mode="python")
            )
        ),
        targeted_tool_rejection=(
            None
            if value.targeted_tool_rejection is None
            else RejectedTargetedToolInvocation.model_validate(
                value.targeted_tool_rejection.model_dump(mode="python")
            )
        ),
    )


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

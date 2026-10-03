"""Application registration validation, tool snapshots and catalogue descriptors."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from copy import deepcopy
from types import MappingProxyType

from cayu._validation import copy_durable_metadata, copy_json_value, require_clean_nonblank
from cayu.agents import AgentSpec
from cayu.context.base import ContextPolicy
from cayu.environments.base import EnvironmentSpec
from cayu.mcp.tools import McpToolAdapter
from cayu.observability.hooks import RuntimeHook
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._isolated_tool_process import (
    isolated_tool_execution_contract,
    validate_process_isolated_tool_registration,
)
from cayu.runtime.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.tools.base import DurableToolRecovery, Tool, ToolSpec
from cayu.tools.catalogue import (
    SEARCH_TOOLS_NAME,
    ToolDescriptor,
    ToolDescriptorProvenance,
    ToolExecutionContract,
    build_tool_descriptor,
    mcp_source_tool_fingerprint,
    validate_application_tool_name,
)
from cayu.tools.isolated import ProcessIsolatedTool
from cayu.vaults.redaction import SecretRedactor


def _copy_registered_tool(tool: runtime_records.RegisteredTool) -> runtime_records.RegisteredTool:
    return runtime_records.RegisteredTool(
        name=tool.name,
        description=tool.description,
        schema=deepcopy(tool.schema),
        parallel_safe=tool.parallel_safe,
        effect=tool.effect,
        publish_arguments=tool.publish_arguments,
        retain_arguments_for_model=tool.retain_arguments_for_model,
        workspace_mutation=tool.workspace_mutation,
        execution_contract=copy_json_value(
            tool.execution_contract,
            "registered_tool.execution_contract",
        ),
        execution_profile_identity=copy_execution_profile_behavior_identity(
            tool.execution_profile_identity
        ),
        command_policy_execution_profile_identity=copy_execution_profile_behavior_identity(
            tool.command_policy_execution_profile_identity
        ),
        tool=tool.tool,
        execution_requirements=ToolSpec(
            name=tool.name, execution_requirements=tool.execution_requirements
        ).execution_requirements,
        auxiliary_inference=ToolSpec(
            name=tool.name, auxiliary_inference=tool.auxiliary_inference
        ).auxiliary_inference,
        child_session_recovery=tool.child_session_recovery,
        durable_tool_recovery=tool.durable_tool_recovery,
        effect_reconciler=tool.effect_reconciler,
    )


def _validate_provider_model_patterns(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise TypeError("Provider model_patterns must be an iterable of strings.")
    try:
        patterns = tuple(value)
    except TypeError as exc:
        raise TypeError("Provider model_patterns must be an iterable of strings.") from exc
    return tuple(
        require_clean_nonblank(pattern, f"model_patterns[{index}]")
        for index, pattern in enumerate(patterns)
    )


def _validate_registered_tool(
    tool: Tool,
    *,
    redactor: SecretRedactor,
    framework_owned: bool = False,
    timeout_seconds: float | None,
) -> runtime_records.RegisteredTool:
    spec = getattr(tool, "spec", None)
    if type(spec) is not ToolSpec:
        raise TypeError("Agent tools must define ToolSpec instances.")
    raw_name = require_clean_nonblank(spec.name, "name")
    if framework_owned:
        if raw_name != SEARCH_TOOLS_NAME:
            raise ValueError(f"Unsupported Cayu runtime tool: {raw_name}")
        name = raw_name
    else:
        name = validate_application_tool_name(raw_name)
    if not inspect.iscoroutinefunction(tool.run):
        raise TypeError(
            f"{type(tool).__name__}.run must be declared with `async def` and return a ToolResult."
        )
    schema = copy_json_value(tool.schema, "schema")
    if type(schema) is not dict:
        raise TypeError(f"{type(tool).__name__}.schema must return a JSON Schema object.")
    publish_arguments = tool._publish_arguments
    retain_arguments_for_model = tool.retain_arguments_for_model
    if type(retain_arguments_for_model) is not bool:
        raise TypeError("Tool model argument retention policy must be a bool.")
    if type(publish_arguments) is not bool:
        raise TypeError(f"{type(tool).__name__} argument publication policy must be a bool.")
    validated_spec = ToolSpec(
        name=name,
        description=spec.description,
        input_schema=schema,
        parallel_safe=spec.parallel_safe,
        effect=spec.effect,
        workspace_mutation=spec.workspace_mutation,
        max_terminal_payload_bytes=spec.max_terminal_payload_bytes,
        execution_requirements=spec.execution_requirements,
        auxiliary_inference=spec.auxiliary_inference,
    )
    command_policy = getattr(tool, "command_policy", None)
    if isinstance(tool, ProcessIsolatedTool):
        if validated_spec.auxiliary_inference is not None:
            raise ValueError(
                "Process-isolated tools cannot request in-process auxiliary inference."
            )
        if validated_spec.workspace_mutation:
            raise ValueError(
                "Process-isolated tools cannot request Cayu workspace mutation authority."
            )
        validate_process_isolated_tool_registration(tool, redactor=redactor)
        execution_contract = isolated_tool_execution_contract(
            tool,
            runtime_timeout_seconds=timeout_seconds,
        )
    else:
        execution_contract = ToolExecutionContract(
            timeout_strength=("cooperative_in_process" if timeout_seconds is not None else "none")
        ).model_dump(mode="json")
    execution_contract = (
        ToolExecutionContract.model_validate(execution_contract)
        .model_copy(
            update={"max_terminal_payload_bytes": validated_spec.max_terminal_payload_bytes}
        )
        .model_dump(mode="json")
    )
    return runtime_records.RegisteredTool(
        name=validated_spec.name,
        description=validated_spec.description,
        schema=validated_spec.input_schema,
        parallel_safe=validated_spec.parallel_safe,
        effect=validated_spec.effect,
        publish_arguments=publish_arguments,
        retain_arguments_for_model=retain_arguments_for_model,
        workspace_mutation=validated_spec.workspace_mutation,
        execution_contract=execution_contract,
        execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
            tool.execution_profile_identity,
            redactor=redactor,
            field_name=f"tools[{name!r}].execution_profile_identity",
        ),
        command_policy_execution_profile_identity=(
            copy_secret_free_execution_profile_behavior_identity(
                None
                if command_policy is None
                else getattr(command_policy, "execution_profile_identity", None),
                redactor=redactor,
                field_name=f"tools[{name!r}].command_policy.execution_profile_identity",
            )
        ),
        tool=tool,
        execution_requirements=validated_spec.execution_requirements,
        auxiliary_inference=validated_spec.auxiliary_inference,
        child_session_recovery=(
            tool if isinstance(tool, runtime_records.ChildSessionRecoveryMatcher) else None
        ),
        durable_tool_recovery=(tool if isinstance(tool, DurableToolRecovery) else None),
    )


def _registered_tool_descriptor(
    tool: runtime_records.RegisteredTool,
) -> ToolDescriptor:
    """Derive one canonical callable-free descriptor from admitted registration state."""

    registered_tool = tool.tool
    if isinstance(registered_tool, McpToolAdapter):
        binding = registered_tool._manifest_binding
        provenance = ToolDescriptorProvenance(
            kind="mcp",
            source_id=registered_tool.toolset.manifest_identity,
            source_tool_fingerprint=mcp_source_tool_fingerprint(binding.manifest_mcp_name),
            source_contract_fingerprint=binding.manifest_contract_hash,
        )
    else:
        provenance = ToolDescriptorProvenance()
    return build_tool_descriptor(
        name=tool.name,
        description=tool.description,
        input_schema=tool.schema,
        parallel_safe=tool.parallel_safe,
        effect=tool.effect,
        publishes_arguments=tool.publish_arguments,
        workspace_mutation=tool.workspace_mutation,
        execution_contract=ToolExecutionContract.model_validate(tool.execution_contract),
        execution_requirements=tool.execution_requirements,
        auxiliary_inference=tool.auxiliary_inference,
        provenance=provenance,
    )


def _validate_agent_spec(spec: AgentSpec) -> AgentSpec:
    if type(spec) is not AgentSpec:
        raise TypeError("Agent registration requires an AgentSpec.")
    return AgentSpec(
        name=spec.name,
        model=spec.model,
        provider_name=spec.provider_name,
        system_prompt=spec.system_prompt,
        workflow_tool_names=spec.workflow_tool_names,
        authoring_state=spec.authoring_state,
        metadata=copy_durable_metadata(spec.metadata),
        provider_options=copy_json_value(spec.provider_options, "provider_options"),
        thinking=spec.thinking,
    )


def _validate_environment_spec(
    spec: EnvironmentSpec,
    *,
    redactor: SecretRedactor,
) -> EnvironmentSpec:
    if type(spec) is not EnvironmentSpec:
        raise TypeError("Environment registration requires an EnvironmentSpec.")
    if type(spec.name) is not str:
        raise ValueError("`name` must be a string.")
    return EnvironmentSpec(
        name=spec.name,
        metadata=copy_durable_metadata(spec.metadata),
        execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
            spec.execution_profile_identity,
            redactor=redactor,
            field_name="environment_spec.execution_profile_identity",
        ),
        lifecycle_policy=spec.lifecycle_policy,
        workspace_checkpoint_policy=spec.workspace_checkpoint_policy,
    )


def _validate_runtime_hooks(
    hooks: Iterable[RuntimeHook] | None,
    *,
    field_name: str,
    redactor: SecretRedactor,
) -> tuple[runtime_records.RegisteredRuntimeHook, ...]:
    if hooks is None:
        return ()
    if isinstance(hooks, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.")
    try:
        hook_list = list(hooks)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.") from exc
    registered_hooks: list[runtime_records.RegisteredRuntimeHook] = []
    for index, hook in enumerate(hook_list):
        if not isinstance(hook, RuntimeHook):
            raise TypeError(f"{field_name} must contain RuntimeHook instances.")
        registered_hooks.append(
            runtime_records.RegisteredRuntimeHook(
                name=hook.name,
                execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                    hook.execution_profile_identity,
                    redactor=redactor,
                    field_name=(f"{field_name}[{index}].execution_profile_identity"),
                ),
                hook=hook,
            )
        )
    return tuple(registered_hooks)


def _snapshot_context_behavior_execution_profile_identities(
    context_policy: ContextPolicy,
    context_overflow_policy: ContextPolicy | None,
    *,
    redactor: SecretRedactor,
) -> Mapping[int, ExecutionProfileBehaviorIdentity | None]:
    """Copy declarations reachable through Cayu-owned context wrappers."""

    from cayu.context.base import (
        CheckpointCompactionContextPolicy,
        ModelCompactor,
        PromptCacheCompactor,
        UsageTriggeredContextPolicy,
    )
    from cayu.memory.context import AutomaticRecallContextPolicy

    snapshots: dict[int, ExecutionProfileBehaviorIdentity | None] = {}

    def visit(policy: ContextPolicy, *, field_name: str) -> None:
        policy_id = id(policy)
        if policy_id in snapshots:
            return
        snapshots[policy_id] = copy_secret_free_execution_profile_behavior_identity(
            policy.execution_profile_identity,
            redactor=redactor,
            field_name=f"{field_name}.execution_profile_identity",
        )
        if type(policy) is AutomaticRecallContextPolicy:
            visit(policy.base_policy, field_name=f"{field_name}.base_policy")
            return
        if type(policy) is UsageTriggeredContextPolicy:
            visit(policy.base_policy, field_name=f"{field_name}.base_policy")
            visit(policy.triggered_policy, field_name=f"{field_name}.triggered_policy")
            return
        if type(policy) is not CheckpointCompactionContextPolicy:
            return

        visit_compactor(
            policy.compactor,
            field_name=f"{field_name}.compactor",
        )

    def visit_compactor(compactor: object, *, field_name: str) -> None:
        compactor_id = id(compactor)
        if compactor_id in snapshots:
            return
        snapshots[compactor_id] = copy_secret_free_execution_profile_behavior_identity(
            getattr(compactor, "execution_profile_identity", None),
            redactor=redactor,
            field_name=f"{field_name}.execution_profile_identity",
        )
        if type(compactor) is ModelCompactor:
            provider = compactor.provider
            snapshots[id(provider)] = copy_secret_free_execution_profile_behavior_identity(
                provider.execution_profile_identity,
                redactor=redactor,
                field_name=f"{field_name}.provider.execution_profile_identity",
            )
            return
        if type(compactor) is PromptCacheCompactor:
            provider = compactor.provider
            snapshots[id(provider)] = copy_secret_free_execution_profile_behavior_identity(
                provider.execution_profile_identity,
                redactor=redactor,
                field_name=f"{field_name}.provider.execution_profile_identity",
            )
            visit_compactor(
                compactor._fallback,
                field_name=f"{field_name}.fallback_compactor",
            )

    visit(context_policy, field_name="context_policy")
    if context_overflow_policy is not None:
        visit(context_overflow_policy, field_name="context_overflow_policy")
    return MappingProxyType(snapshots)

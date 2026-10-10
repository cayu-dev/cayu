"""Resolve complete execution profiles for session operations."""

from __future__ import annotations

from typing import Any, cast

from cayu._validation import (
    copy_json_value,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    budget_limits_for_session,
    request_budget_execution_profile_ids,
)
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import (
    StructuredOutputSpec,
)
from cayu.context.thinking import ThinkingConfig, thinking_config_payload
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _execution_profile_continuation as execution_profile_continuation
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_continuation import (
    _execution_profile_provider_options,
    _execution_profile_structured_output,
)
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._tool_completion import (
    require_registered_completion_tools,
)
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import (
    LoopPolicy,
)
from cayu.runtime.tool_completion import (
    ToolCompletionPolicy,
    copy_tool_completion_policy,
)
from cayu.sessions.records import (
    Session,
    SessionRuntimeIdentity,
    copy_session_runtime_identity,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.vaults.redaction import SecretRedactor


def _session_tool_capability_ceiling(
    session: Session,
) -> ToolCapabilityCeiling:
    """Load the required durable application-tool authority."""

    return tool_capability_ceiling_from_session_metadata(session.metadata)


def _execution_profile_identity(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    redactor: SecretRedactor,
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
    process_identity: str = "standalone-profile-builder",
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
    loop_policies: tuple[LoopPolicy, ...] = (),
    loop_policy_execution_profile_identities: tuple[
        ExecutionProfileBehaviorIdentity | None, ...
    ] = (),
    request_loop_policies: tuple[LoopPolicy, ...] = (),
    request_loop_policy_instance_identities: tuple[str | None, ...] = (),
    registered_provider: runtime_records.RegisteredProvider | None = None,
    budget_policy: BudgetPolicy | None = None,
    request_budget_limits: tuple[BudgetLimit, ...] = (),
    causal_budget_id: str | None = None,
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    max_steps: int | None = None,
    limits: RunLimits | None = None,
    retry_policy: RetryPolicy | None = None,
    finalization_material: dict[str, Any] | None = None,
    tool_completion: ToolCompletionPolicy | None = None,
    tool_capability_ceiling: ToolCapabilityCeiling | None = None,
    runtime_identity: SessionRuntimeIdentity | None = None,
) -> ExecutionProfileIdentity:
    tool_completion = copy_tool_completion_policy(tool_completion)
    require_registered_completion_tools(tool_completion, registered_agent)
    if tool_completion is not None and structured_output is not None:
        raise ValueError("tool_completion and structured_output cannot be combined.")
    if tool_completion is not None:
        material = tool_completion.model_dump(mode="json")
        if redactor.redact_json(material) != material:
            raise ValueError("tool_completion names must be free of workload secrets.")
    if finalization_material is None and max_steps is None:
        raise ValueError("Execution profile requires resolved invocation max_steps.")
    runtime_identity = (
        SessionRuntimeIdentity(
            runtime_name="cayu",
            runtime_version=execution_profile_continuation._runtime_version(),
            runtime_build_provenance=current_runtime_build_provenance(),
        )
        if runtime_identity is None
        else copy_session_runtime_identity(runtime_identity)
    )
    provider_options, provider_options_process_local = _execution_profile_provider_options(
        registered_agent.spec.provider_options,
        provider=registered_provider.provider if registered_provider is not None else None,
        model=model,
        process_identity=process_identity,
    )
    effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
    app_limit_ids: tuple[str, ...] = ()
    request_limit_ids: tuple[str, ...] = ()
    if causal_budget_id is not None:
        app_limit_ids = tuple(
            limit.budget_limit_id
            for limit in budget_limits_for_session(
                policy=budget_policy,
                agent_name=registered_agent.spec.name,
                causal_budget_id=causal_budget_id,
            )
        )
        request_limit_ids = request_budget_execution_profile_ids(
            limits=request_budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=causal_budget_id,
        )
    return execution_profile_admission.resolve_execution_profile_identity(
        registered_agent=registered_agent,
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        runtime_name=runtime_identity.runtime_name,
        runtime_version=runtime_identity.runtime_version,
        runtime_build_provenance=runtime_identity.runtime_build_provenance,
        redactor=redactor,
        registered_environment=registered_environment,
        process_identity=process_identity,
        runtime_hooks=runtime_hooks,
        loop_policies=loop_policies,
        loop_policy_identities=loop_policy_execution_profile_identities,
        invocation_loop_policies=request_loop_policies,
        invocation_loop_policy_identities=tuple(
            copy_secret_free_execution_profile_behavior_identity(
                policy.execution_profile_identity,
                redactor=redactor,
                field_name=f"request.loop_policies[{index}].execution_profile_identity",
            )
            for index, policy in enumerate(request_loop_policies)
        ),
        invocation_loop_policy_instance_identities=(request_loop_policy_instance_identities),
        registered_provider=registered_provider,
        provider_options=provider_options,
        provider_options_process_local=provider_options_process_local,
        thinking=(
            None if effective_thinking is None else thinking_config_payload(effective_thinking)
        ),
        app_budget_limit_ids=app_limit_ids,
        request_budget_limit_ids=request_limit_ids,
        structured_output=_execution_profile_structured_output(structured_output),
        finalization=(
            execution_profile_admission.model_finalization_material(
                max_steps=cast("int", max_steps),
                limits=copy_run_limits(limits),
                retry_policy=copy_retry_policy(retry_policy),
                tool_completion=tool_completion,
            )
            if finalization_material is None
            else copy_json_value(finalization_material, "finalization_material")
        ),
        tool_capability_ceiling=(
            None if tool_capability_ceiling is None else tool_capability_ceiling.tool_names
        ),
    )

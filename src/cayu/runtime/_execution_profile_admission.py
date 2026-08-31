"""Deep execution-profile admission rules shared by resume and recovery."""

from __future__ import annotations

import hmac
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from threading import Lock
from typing import Any, cast
from uuid import uuid4
from weakref import ReferenceType, ref

from cayu._validation import canonical_durable_json_bytes
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.egress.authority import EgressAuthorityIdentity, _copy_egress_authority_identity
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.build_provenance import RuntimeBuildProvenance
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_with_component,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import Session
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.targeted_tool_projection import (
    TargetedToolProjectionKind,
    resolve_targeted_tool_projection,
)
from cayu.runtime.tool_discovery import (
    ToolDiscoveryProjectionKind,
    resolve_tool_discovery_projection,
    tool_discovery_execution_profile_material,
)
from cayu.runtime.tool_gateway import call_tool_gateway_execution_profile_material
from cayu.runtime.user_input import pending_user_input_from_checkpoint
from cayu.vaults import SecretRedactor

_MODEL_FINALIZATION_MATERIAL_KIND = "cayu:model-finalization:v2"
_MODEL_COMPACTOR_MATERIAL_VERSION = 2
_PROMPT_CACHE_COMPACTOR_MATERIAL_VERSION = 2
_EGRESS_AUTHORITY_SCHEMA_KEYS = frozenset(
    {
        "allowed_destinations",
        "authority_scope",
        "authority_source",
        "bindings",
        "comparison_available",
        "credential_authority_fingerprint",
        "credential_kind",
        "cutover_strategy",
        "denied_path_prefixes",
        "destination",
        "fingerprint",
        "generation",
        "kind",
        "match",
        "method",
        "name",
        "operations",
        "path",
        "policies",
        "policy_name",
        "policy_version",
        "record_type",
        "runner_kind",
        "schema_version",
    }
)


def _egress_authority_application_text_values(
    identity: EgressAuthorityIdentity,
) -> tuple[str, ...]:
    """Return only configurable text, excluding validated protocol controls."""

    values = [
        identity.authority_source,
        identity.authority_scope,
        identity.policy_version,
        identity.runner_kind,
    ]
    for policy in identity.policies:
        values.extend((policy.name, *policy.allowed_destinations))
        for operation in policy.operations:
            values.extend((operation.method, operation.path))
        values.extend(policy.denied_path_prefixes)
    for binding in identity.bindings:
        values.extend(
            (
                binding.destination,
                binding.policy_name,
                binding.credential_kind,
            )
        )
    return tuple(values)


@dataclass(frozen=True)
class ExecutionProfileContinuationPlan:
    """A reconstructed continuation and any component drift it exposes."""

    snapshot: ActiveInvocationExecutionProfile
    candidate_profile: ExecutionProfileIdentity
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...]


def model_finalization_material(
    *,
    max_steps: int,
    limits: RunLimits,
    retry_policy: RetryPolicy,
) -> dict[str, Any]:
    """Return the versioned structural identity for model-loop finalization."""

    if type(max_steps) is not int:
        raise TypeError("max_steps must be an integer.")
    if type(limits) is not RunLimits:
        raise TypeError("limits must be a RunLimits instance.")
    if type(retry_policy) is not RetryPolicy:
        raise TypeError("retry_policy must be a RetryPolicy instance.")
    return {
        "kind": _MODEL_FINALIZATION_MATERIAL_KIND,
        "max_steps": max_steps,
        "limits": limits.model_dump(mode="json"),
        "retry_policy": retry_policy.model_dump(mode="json"),
    }


class ProcessLocalBehaviorIdentityRegistry:
    """Assign opaque identities to exact live behavior objects without retaining them."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._identities: dict[int, tuple[ReferenceType[object], str]] = {}

    def identity_for(self, value: object) -> str:
        """Return one stable token for ``value`` while that exact object remains live."""

        object_id = id(value)
        with self._lock:
            existing = self._identities.get(object_id)
            if existing is not None and existing[0]() is value:
                return existing[1]

            identity = uuid4().hex

            def discard(reference: ReferenceType[object]) -> None:
                with self._lock:
                    current = self._identities.get(object_id)
                    if current is not None and current[0] is reference:
                        self._identities.pop(object_id, None)

            try:
                reference = ref(value, discard)
            except TypeError as exc:  # pragma: no cover - LoopPolicy supports weak references
                raise TypeError(
                    "Process-local behavior identity requires a weak-referenceable object."
                ) from exc
            self._identities[object_id] = (reference, identity)
            return identity


def _process_scope_identity(process_identity: str) -> str:
    """Return a public commitment without exposing the private process HMAC key."""

    return sha256(process_identity.encode("utf-8")).hexdigest()


def resolve_execution_profile_identity(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    runtime_name: str,
    runtime_version: str | None,
    redactor: SecretRedactor,
    runtime_build_provenance: RuntimeBuildProvenance | None = None,
    process_identity: str = "standalone-profile-builder",
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
    loop_policies: tuple[Any, ...] = (),
    loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policies: tuple[Any, ...] = (),
    invocation_loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policy_instance_identities: tuple[str | None, ...] = (),
    registered_provider: runtime_records.RegisteredProvider | None = None,
    provider_options: Mapping[str, Any] | None = None,
    provider_options_process_local: bool = False,
    thinking: Mapping[str, Any] | None = None,
    app_budget_limit_ids: tuple[str, ...] = (),
    request_budget_limit_ids: tuple[str, ...] = (),
    structured_output: Mapping[str, Any] | None = None,
    finalization: Mapping[str, Any] | None = None,
    tool_capability_ceiling: tuple[str, ...] | None = None,
) -> ExecutionProfileIdentity:
    """Resolve one registered runtime body into its durable profile identity."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("Execution-profile resolution requires a SecretRedactor.")

    descriptors_by_name = {
        descriptor.name: descriptor for descriptor in registered_agent.tool_catalogue.descriptors
    }
    if set(descriptors_by_name) != set(registered_agent.tools):
        raise RuntimeError("Registered tool catalogue conflicts with admitted agent tools.")
    direct_tools = [
        descriptors_by_name[name].execution_profile_material() for name in registered_agent.tools
    ]
    registered_tool_names = tuple(registered_agent.tools)
    if tool_capability_ceiling is None:
        effective_tool_capability_ceiling = registered_tool_names
    else:
        if type(tool_capability_ceiling) is not tuple:
            raise TypeError("tool_capability_ceiling must be a tuple or None.")
        if len(tool_capability_ceiling) != len(set(tool_capability_ceiling)):
            raise ValueError("tool_capability_ceiling must contain unique tool names.")
        ceiling_set = frozenset(tool_capability_ceiling)
        if any(name not in registered_tool_names for name in tool_capability_ceiling):
            raise ValueError("tool_capability_ceiling contains an unregistered tool.")
        effective_tool_capability_ceiling = tuple(
            name for name in registered_tool_names if name in ceiling_set
        )
        if effective_tool_capability_ceiling != tool_capability_ceiling:
            raise ValueError("tool_capability_ceiling must preserve registration order.")
    tool_implementations = []
    tool_implementations_process_local = False
    tool_implementations_application_versioned = False
    for index, tool in enumerate(registered_agent.tools.values()):
        entry, process_local = _behavior_identity_material(
            identity=tool.execution_profile_identity,
            value=tool.tool,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"tool:{index}:{tool.name}",
            cayu_owned_material=_secret_safe_cayu_owned_material(
                _cayu_tool_material(tool.tool),
                redactor=redactor,
            ),
        )
        tool_implementations.append({"implementation": entry})
        tool_implementations_process_local |= process_local
        tool_implementations_application_versioned |= tool.execution_profile_identity is not None

    command_policy_material = []
    execution_policies_process_local = False
    execution_policies_application_versioned = False
    for tool in registered_agent.tools.values():
        command_policy = getattr(tool.tool, "command_policy", None)
        if command_policy is None:
            continue
        entry, process_local = _behavior_identity_material(
            identity=tool.command_policy_execution_profile_identity,
            value=command_policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"command-policy:{tool.name}",
            cayu_owned_material=_secret_safe_cayu_owned_material(
                _cayu_policy_material(command_policy),
                redactor=redactor,
            ),
        )
        command_policy_material.append({"tool": tool.name, "policy": entry})
        execution_policies_process_local |= process_local
        execution_policies_application_versioned |= (
            tool.command_policy_execution_profile_identity is not None
        )

    tool_policy_entry, tool_policy_process_local = _behavior_identity_material(
        identity=registered_agent.tool_policy_execution_profile_identity,
        value=registered_agent.tool_policy,
        runtime_version=runtime_version,
        process_identity=process_identity,
        slot="tool-policy",
        cayu_owned_material=_secret_safe_cayu_owned_material(
            _cayu_policy_material(registered_agent.tool_policy),
            redactor=redactor,
        ),
    )
    execution_policies_process_local |= tool_policy_process_local
    execution_policies_application_versioned |= (
        registered_agent.tool_policy_execution_profile_identity is not None
    )

    # Preserve the historical expose-all profile byte-for-byte. Any other
    # exposure policy changes provider-visible behavior and therefore belongs
    # in the existing execution-policies identity component.
    from cayu.runtime.tool_exposure import AllRegisteredToolsExposurePolicy

    tool_exposure_policy_entry: dict[str, Any] | None = None
    if type(registered_agent.tool_exposure_policy) is not AllRegisteredToolsExposurePolicy:
        tool_exposure_policy_entry, exposure_policy_process_local = _behavior_identity_material(
            identity=(registered_agent.tool_exposure_policy_execution_profile_identity),
            value=registered_agent.tool_exposure_policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot="tool-exposure-policy",
            cayu_owned_material=_secret_safe_cayu_owned_material(
                _cayu_policy_material(registered_agent.tool_exposure_policy),
                redactor=redactor,
            ),
        )
        execution_policies_process_local |= exposure_policy_process_local
        execution_policies_application_versioned |= (
            registered_agent.tool_exposure_policy_execution_profile_identity is not None
        )

    combined_loop_policies = (*loop_policies, *registered_agent.loop_policies)
    combined_loop_identities = (
        *loop_policy_identities,
        *registered_agent.loop_policy_execution_profile_identities,
    )
    if len(combined_loop_policies) != len(combined_loop_identities):
        raise RuntimeError("Loop-policy execution-profile identities are inconsistent.")
    loop_policy_material = []
    for index, (policy, identity) in enumerate(
        zip(combined_loop_policies, combined_loop_identities, strict=True)
    ):
        entry, process_local = _behavior_identity_material(
            identity=identity,
            value=policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"loop-policy:{index}",
        )
        loop_policy_material.append(entry)
        execution_policies_process_local |= process_local
        execution_policies_application_versioned |= identity is not None

    if len(invocation_loop_policies) != len(invocation_loop_policy_identities):
        raise RuntimeError("Invocation loop-policy execution-profile identities are inconsistent.")
    invocation_policy_material = []
    invocation_policies_process_local = False
    invocation_policies_application_versioned = False
    if not invocation_loop_policy_instance_identities:
        invocation_loop_policy_instance_identities = (None,) * len(invocation_loop_policies)
    if len(invocation_loop_policies) != len(invocation_loop_policy_instance_identities):
        raise RuntimeError("Invocation loop-policy instance identities are inconsistent.")
    for index, (policy, identity, instance_identity) in enumerate(
        zip(
            invocation_loop_policies,
            invocation_loop_policy_identities,
            invocation_loop_policy_instance_identities,
            strict=True,
        )
    ):
        if identity is None and instance_identity is None:
            raise RuntimeError(
                "Opaque invocation loop policy is missing its process-local instance identity."
            )
        entry, process_local = _behavior_identity_material(
            identity=identity,
            value=policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"invocation-loop-policy:{index}",
            process_local_instance_identity=instance_identity,
        )
        invocation_policy_material.append(entry)
        invocation_policies_process_local |= process_local
        invocation_policies_application_versioned |= identity is not None

    combined_hooks = (*runtime_hooks, *registered_agent.runtime_hooks)
    runtime_hook_material = []
    runtime_hooks_process_local = False
    runtime_hooks_application_versioned = False
    for index, hook in enumerate(combined_hooks):
        entry, process_local = _behavior_identity_material(
            identity=hook.execution_profile_identity,
            value=hook.hook,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"runtime-hook:{index}:{hook.name}",
        )
        runtime_hook_material.append({"name": hook.name, "behavior": entry})
        runtime_hooks_process_local |= process_local
        runtime_hooks_application_versioned |= hook.execution_profile_identity is not None

    (
        environment_material,
        environment_process_local,
        environment_application_versioned,
    ) = _environment_identity_material(
        registered_environment=registered_environment,
        execution_requirements=registered_agent.execution_requirements.model_dump(mode="json"),
        runtime_version=runtime_version,
        process_identity=process_identity,
        redactor=redactor,
    )
    environment = None if registered_environment is None else registered_environment.environment
    egress_authority: EgressAuthorityIdentity | None = None
    if registered_environment is not None and registered_environment.factory is not None:
        proposed_egress_authority = registered_environment.factory.egress_authority_identity
        if proposed_egress_authority is not None:
            if type(proposed_egress_authority) is not EgressAuthorityIdentity:
                raise TypeError(
                    "Environment factory egress_authority_identity must be an "
                    "EgressAuthorityIdentity or None."
                )
            egress_authority = _copy_egress_authority_identity(proposed_egress_authority)
            egress_authority_material = egress_authority.model_dump(mode="json")
            redactor.require_no_secret_keys(
                egress_authority_material,
                field_name="environment factory egress authority identity",
                preserve_keys=_EGRESS_AUTHORITY_SCHEMA_KEYS,
                match_short_substrings=True,
            )
            if any(
                redactor.redact_text(value) != value
                for value in _egress_authority_application_text_values(egress_authority)
            ):
                raise ValueError(
                    "Environment factory egress authority identity contains a workload "
                    "secret and cannot be used as durable execution authority."
                )
    credential_authority = (
        "factory_managed"
        if registered_environment is not None and registered_environment.factory_backed
        else (
            "none"
            if environment is None or (environment.vault is None and environment.proxy is None)
            else (
                "brokered"
                if environment.proxy is not None and environment.vault is None
                else ("direct_and_brokered" if environment.proxy is not None else "direct")
            )
        )
    )
    runner_authority = (
        "factory_managed"
        if registered_environment is not None and registered_environment.factory_backed
        else ("present" if environment is not None and environment.runner is not None else "none")
    )
    context_components = _context_component_materials(
        registered_agent=registered_agent,
        runtime_version=runtime_version,
        process_identity=process_identity,
        redactor=redactor,
    )
    provider_entry: dict[str, Any]
    provider_process_local = False
    provider_application_versioned = False
    if registered_provider is None:
        provider_entry = {
            "kind": "structural_target_only",
            "provider_name": provider_name,
        }
    else:
        provider_material = _cayu_provider_material(registered_provider.provider)
        safe_provider_material = _secret_safe_cayu_owned_material(
            provider_material,
            redactor=redactor,
        )
        if provider_material is not None and safe_provider_material is None:
            provider_entry = _process_local_private_material(
                provider_material,
                value=registered_provider.provider,
                process_identity=process_identity,
                slot=f"model-provider:{registered_provider.name}",
            )
            provider_process_local = True
        else:
            provider_entry, provider_process_local = _behavior_identity_material(
                identity=registered_provider.execution_profile_identity,
                value=registered_provider.provider,
                runtime_version=runtime_version,
                process_identity=process_identity,
                slot=f"model-provider:{registered_provider.name}",
                cayu_owned_material=safe_provider_material,
            )
        provider_application_versioned = registered_provider.execution_profile_identity is not None
    provider_request_process_local = provider_options_process_local
    if provider_options_process_local:
        safe_provider_options = _validated_private_provider_options_material(
            provider_options,
            redactor=redactor,
        )
    else:
        safe_provider_options = _secret_safe_cayu_owned_material(
            {} if provider_options is None else dict(provider_options),
            redactor=redactor,
        )
    if safe_provider_options is None:
        provider_request_process_local = True
        safe_provider_options = _process_local_private_material(
            {} if provider_options is None else dict(provider_options),
            value=registered_agent.spec,
            process_identity=process_identity,
            slot="provider-options",
        )
    safe_thinking = _secret_safe_cayu_owned_material(
        {} if thinking is None else dict(thinking),
        redactor=redactor,
    )
    if safe_thinking is None:
        provider_request_process_local = True
        safe_thinking = _process_local_private_material(
            {} if thinking is None else dict(thinking),
            value=registered_agent.spec,
            process_identity=process_identity,
            slot="thinking",
        )
    targeted_tool_projection = (
        None
        if registered_agent.targeted_tool_mode is None or registered_provider is None
        else resolve_targeted_tool_projection(
            registered_agent.targeted_tool_mode,
            provider=registered_provider.provider,
            model=model,
        )
    )
    tool_discovery_projection = (
        None
        if registered_agent.tool_discovery_mode is None or registered_provider is None
        else resolve_tool_discovery_projection(
            registered_agent.tool_discovery_mode,
            provider=registered_provider.provider,
            model=model,
        )
    )
    targeted_tool_delivery_material = (
        None
        if registered_agent.targeted_tool_mode is None
        else {
            "kind": "cayu:targeted-tool-delivery",
            "schema_version": 2,
            "configured_mode": registered_agent.targeted_tool_mode.value,
            "resolved_projection": (
                None if targeted_tool_projection is None else targeted_tool_projection.value
            ),
            **(
                {
                    "call_tool_core": {
                        **call_tool_gateway_execution_profile_material(),
                        "callable": (
                            targeted_tool_projection is TargetedToolProjectionKind.CALL_TOOL
                            or tool_discovery_projection is ToolDiscoveryProjectionKind.SEARCH_TOOLS
                        ),
                    }
                }
                if targeted_tool_projection is not None
                else {}
            ),
        }
    )
    tool_discovery_material = (
        None
        if registered_agent.tool_discovery_mode is None
        else {
            **tool_discovery_execution_profile_material(),
            "configured_mode": registered_agent.tool_discovery_mode.value,
            "resolved_projection": (
                None if tool_discovery_projection is None else tool_discovery_projection.value
            ),
            "call_tool_core": {
                **call_tool_gateway_execution_profile_material(),
                "callable": (tool_discovery_projection is ToolDiscoveryProjectionKind.SEARCH_TOOLS),
            },
        }
    )
    mcp_tool_source_material = (
        None
        if not registered_agent.mcp_toolsets
        else {
            "kind": "cayu:mcp-tool-sources",
            "schema_version": 1,
            "sources": [
                {
                    "manifest_identity": toolset.manifest_identity,
                    "registration_mode": "complete",
                }
                for toolset in sorted(
                    registered_agent.mcp_toolsets,
                    key=lambda item: item.manifest_identity,
                )
            ],
        }
    )
    return build_execution_profile_identity(
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        runtime_build_provenance=runtime_build_provenance,
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        direct_tools=direct_tools,
        tool_catalogue_revision=registered_agent.tool_catalogue.revision,
        tool_implementations=tool_implementations,
        tool_implementations_process_local=tool_implementations_process_local,
        tool_implementations_application_versioned=(tool_implementations_application_versioned),
        tool_view_grants={
            "view_kind": "direct",
            "generation": 1,
            "grant_baseline": list(effective_tool_capability_ceiling),
        },
        execution_policies={
            "tool_policy": tool_policy_entry,
            "command_policies": command_policy_material,
            "loop_policies": loop_policy_material,
            **(
                {}
                if targeted_tool_delivery_material is None
                else {"targeted_tool_delivery": targeted_tool_delivery_material}
            ),
            **(
                {}
                if tool_discovery_material is None
                else {"tool_discovery": tool_discovery_material}
            ),
            **(
                {}
                if mcp_tool_source_material is None
                else {"mcp_tool_sources": mcp_tool_source_material}
            ),
            **(
                {}
                if tool_exposure_policy_entry is None
                else {"tool_exposure_policy": tool_exposure_policy_entry}
            ),
        },
        execution_policies_process_local=execution_policies_process_local,
        execution_policies_application_versioned=execution_policies_application_versioned,
        invocation_policies=invocation_policy_material,
        invocation_policies_process_local=invocation_policies_process_local,
        invocation_policies_application_versioned=invocation_policies_application_versioned,
        runtime_hooks=runtime_hook_material,
        runtime_hooks_process_local=runtime_hooks_process_local,
        runtime_hooks_application_versioned=runtime_hooks_application_versioned,
        execution_environment=environment_material,
        execution_environment_process_local=environment_process_local,
        execution_environment_application_versioned=environment_application_versioned,
        effect_authority={
            "tool_effects": [
                {
                    "name": tool.name,
                    "effect": tool.effect.value,
                    "workspace_mutation": tool.workspace_mutation,
                    "publishes_arguments": tool.publish_arguments,
                }
                for tool in registered_agent.tools.values()
            ],
            "provider_hosted_tools": [
                tool.model_dump(mode="json") for tool in registered_agent.hosted_tools
            ],
            "credential_authority": credential_authority,
            "egress_authority": registered_agent.execution_requirements.network_access,
            "real_secret_visibility": (
                registered_agent.execution_requirements.real_secret_visibility
            ),
            "runner_authority": runner_authority,
        },
        egress_authority=egress_authority,
        context_selection=context_components.selection,
        context_selection_process_local=context_components.selection_process_local,
        context_selection_application_versioned=(
            context_components.selection_application_versioned
        ),
        automatic_recall=context_components.recall,
        automatic_recall_process_local=context_components.recall_process_local,
        automatic_recall_application_versioned=(context_components.recall_application_versioned),
        context_compaction=context_components.compaction,
        context_compaction_process_local=context_components.compaction_process_local,
        context_compaction_application_versioned=(
            context_components.compaction_application_versioned
        ),
        live_state_projection={"kind": "none", "version": 1},
        provider_adapter=provider_entry,
        provider_adapter_process_local=provider_process_local,
        provider_adapter_application_versioned=provider_application_versioned,
        provider_request_policy={
            "provider_options": safe_provider_options,
            "thinking": safe_thinking,
        },
        provider_request_policy_process_local=provider_request_process_local,
        application_budget_policy={"limit_ids": list(app_budget_limit_ids)},
        invocation_budget_policy={"limit_ids": list(request_budget_limit_ids)},
        structured_output=(
            {"kind": "none", "version": 1} if structured_output is None else structured_output
        ),
        finalization=(
            model_finalization_material(
                max_steps=16,
                limits=RunLimits(),
                retry_policy=RetryPolicy(),
            )
            if finalization is None
            else finalization
        ),
    )


def prepare_execution_profile_continuation(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    runtime_version: str | None,
    redactor: SecretRedactor,
    runtime_build_provenance: RuntimeBuildProvenance | None = None,
    process_identity: str = "standalone-profile-builder",
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
    loop_policies: tuple[Any, ...] = (),
    loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policies: tuple[Any, ...] | None = None,
    invocation_loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policy_instance_identities: tuple[str | None, ...] = (),
    additional_profile_fingerprints: Iterable[str | None] = (),
    frozen_candidate_profile: ExecutionProfileIdentity | None = None,
    provider_options: Mapping[str, Any] | None = None,
    provider_options_process_local: bool = False,
    thinking: Mapping[str, Any] | None = None,
    app_budget_limit_ids: tuple[str, ...] = (),
    request_budget_limit_ids: tuple[str, ...] = (),
    structured_output: Mapping[str, Any] | None = None,
    finalization: Mapping[str, Any] | None = None,
    invocation_semantics_available: bool = False,
    tool_capability_ceiling: tuple[str, ...] | None = None,
) -> ExecutionProfileContinuationPlan:
    """Reconstruct a pending invocation and fail closed on invalid authority."""

    if invocation_loop_policies is None and (
        invocation_loop_policy_identities or invocation_loop_policy_instance_identities
    ):
        raise ValueError(
            "Invocation loop-policy identities require invocation loop-policy objects."
        )
    snapshot = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if snapshot is None:
        raise RuntimeError(
            "Pending recovery state has no durable active invocation execution profile."
        )
    if not active_invocation_execution_profile_matches_session_epoch(
        snapshot,
        session_id=session.id,
        run_epoch=session.run_epoch,
    ):
        raise RuntimeError(
            "Active invocation execution profile does not match the session run epoch: "
            f"snapshot={snapshot.run_epoch}, session={session.run_epoch}."
        )
    pending_profile_fingerprints = {
        pending.execution_profile_fingerprint
        for pending in (
            tool_round_recovery.pending_tool_round_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
            approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
            pending_user_input_from_checkpoint(
                checkpoint,
                redactor=redactor,
                consume_on_rejection=True,
            ),
        )
        if pending is not None
    }
    pending_profile_fingerprints.update(additional_profile_fingerprints)
    if pending_profile_fingerprints and pending_profile_fingerprints != {
        snapshot.profile.fingerprint
    }:
        raise RuntimeError(
            "Pending recovery state does not reference the active invocation execution profile."
        )
    if frozen_candidate_profile is None:
        candidate = resolve_execution_profile_identity(
            registered_agent=registered_agent,
            provider_name=registered_provider.name,
            model=session.model,
            durable_system_prompt=None,
            runtime_name="cayu",
            runtime_version=runtime_version,
            runtime_build_provenance=runtime_build_provenance,
            redactor=redactor,
            process_identity=process_identity,
            registered_environment=registered_environment,
            runtime_hooks=runtime_hooks,
            loop_policies=loop_policies,
            loop_policy_identities=loop_policy_identities,
            invocation_loop_policies=(
                () if invocation_loop_policies is None else invocation_loop_policies
            ),
            invocation_loop_policy_identities=invocation_loop_policy_identities,
            invocation_loop_policy_instance_identities=(invocation_loop_policy_instance_identities),
            registered_provider=registered_provider,
            provider_options=provider_options,
            provider_options_process_local=provider_options_process_local,
            thinking=thinking,
            app_budget_limit_ids=app_budget_limit_ids,
            request_budget_limit_ids=request_budget_limit_ids,
            structured_output=structured_output,
            finalization=finalization,
            tool_capability_ceiling=tool_capability_ceiling,
        )
    elif type(frozen_candidate_profile) is not ExecutionProfileIdentity:
        raise TypeError("frozen_candidate_profile must be an ExecutionProfileIdentity or None.")
    else:
        candidate = frozen_candidate_profile
    if (
        frozen_candidate_profile is None
        and invocation_loop_policies is None
        and snapshot.profile.schema_version >= 2
    ):
        candidate = execution_profile_with_component(
            candidate,
            snapshot.profile.component(ExecutionProfileComponentClass.INVOCATION_POLICIES),
        )
    if frozen_candidate_profile is None:
        candidate = execution_profile_with_component(
            candidate,
            snapshot.profile.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION),
        )
        if snapshot.profile.schema_version >= 4 and not invocation_semantics_available:
            for component_class in (
                ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
                ExecutionProfileComponentClass.FINALIZATION,
            ):
                candidate = execution_profile_with_component(
                    candidate,
                    snapshot.profile.component(component_class),
                )
    return ExecutionProfileContinuationPlan(
        snapshot=snapshot,
        candidate_profile=candidate,
        changed_component_classes=changed_execution_profile_components(
            snapshot.profile,
            candidate,
        ),
    )


def _behavior_identity_material(
    *,
    identity: ExecutionProfileBehaviorIdentity | None,
    value: object,
    runtime_version: str | None,
    process_identity: str,
    slot: str,
    cayu_owned_material: dict[str, Any] | None = None,
    process_local_instance_identity: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if identity is not None:
        return {"kind": "application_versioned", **identity.model_dump(mode="json")}, False
    if cayu_owned_material is not None and runtime_version is not None:
        return (
            {
                "kind": "cayu_versioned",
                "component": _qualified_type_name(value),
                "behavior_version": "1",
                "implementation_version": "1",
                "configuration": cayu_owned_material,
            },
            False,
        )
    return (
        {
            "kind": "process_local_unverifiable",
            "component": _qualified_type_name(value),
            "process_scope": (
                _process_scope_identity(process_identity)
                if process_local_instance_identity is None
                else "exact_live_object"
            ),
            "slot": slot,
            **(
                {"instance_identity": process_local_instance_identity}
                if process_local_instance_identity is not None
                else {}
            ),
        },
        True,
    )


def _secret_safe_cayu_owned_material(
    material: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any] | None:
    """Keep inspectable material only when it is outside the workload-secret boundary."""

    if material is None:
        return None
    if redactor.redact_json(material) != material:
        # Hashing a redacted replacement would collapse distinct authority into
        # one identity. Fall back to exact app-local behavior identity instead.
        return None
    return material


def _validated_private_provider_options_material(
    material: Mapping[str, Any] | None,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Detach the internal opaque-options commitment without trusting raw values."""

    if material is None or set(material) != {
        "public_projection",
        "private_configuration_hmac_sha256",
        "process_scope",
    }:
        raise ValueError("Private provider-options material is malformed.")
    public_projection = material["public_projection"]
    private_commitment = material["private_configuration_hmac_sha256"]
    process_scope = material["process_scope"]
    if type(public_projection) is not dict:
        raise TypeError("Private provider-options public projection must be a dictionary.")
    if not _is_lower_sha256(private_commitment) or not _is_lower_sha256(process_scope):
        raise ValueError("Private provider-options commitments are malformed.")
    safe_public_projection = _secret_safe_cayu_owned_material(
        public_projection,
        redactor=redactor,
    )
    return {
        **({} if safe_public_projection is None else {"public_projection": safe_public_projection}),
        "private_configuration_hmac_sha256": private_commitment,
        "process_scope": process_scope,
    }


def _is_lower_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}:{value_type.__qualname__}"


_ExecutionProfileMaterial = dict[str, Any] | None
_ExecutionProfileMaterialExtractor = Callable[[Any], _ExecutionProfileMaterial]


def _material_from_exact_type(
    value: object,
    extractors: dict[type[object], _ExecutionProfileMaterialExtractor],
) -> _ExecutionProfileMaterial:
    """Invoke only an extractor registered for the value's exact built-in type."""

    extractor = extractors.get(type(value))
    if extractor is None:
        return None
    return extractor(value)


@lru_cache(maxsize=1)
def _cayu_tool_material_extractors() -> dict[type[object], _ExecutionProfileMaterialExtractor]:
    # Imports stay lazy so this deep runtime module does not participate in the
    # public tool modules' import graph. Each unbound extractor is owned by the
    # component it describes; exact-type lookup prevents subclasses from
    # inheriting Cayu-versioned trust.
    from cayu.tools.browser import ScreenshotPageTool
    from cayu.tools.browser_session import BrowserSessionTool
    from cayu.tools.commands import ExecCommandTool
    from cayu.tools.files import (
        DeleteFileTool,
        EditFileTool,
        ListArtifactsTool,
        ListFilesTool,
        ReadFileTool,
        WriteFileTool,
    )
    from cayu.tools.git import GitChangesTool
    from cayu.tools.knowledge import (
        ListKnowledgeTool,
        ReadKnowledgeTool,
        RememberKnowledgeTool,
        SearchKnowledgeTool,
    )
    from cayu.tools.named_checks import RunCheckTool
    from cayu.tools.patches import ApplyPatchTool
    from cayu.tools.search import SearchTextTool
    from cayu.tools.structured_commands import RunCommandTool
    from cayu.tools.user_input import UserInputTool
    from cayu.tools.web import WebFetchTool
    from cayu.tools.web_access import WebAccessRoutingTool

    tool_types = (
        ExecCommandTool,
        RunCheckTool,
        ApplyPatchTool,
        RunCommandTool,
        DeleteFileTool,
        EditFileTool,
        ListArtifactsTool,
        ListFilesTool,
        ReadFileTool,
        WriteFileTool,
        GitChangesTool,
        ListKnowledgeTool,
        ReadKnowledgeTool,
        RememberKnowledgeTool,
        SearchKnowledgeTool,
        SearchTextTool,
        BrowserSessionTool,
        ScreenshotPageTool,
        UserInputTool,
        WebFetchTool,
        WebAccessRoutingTool,
    )
    return {tool_type: tool_type._execution_profile_material for tool_type in tool_types}


def _cayu_tool_material(tool: object) -> dict[str, Any] | None:
    """Return bounded built-in behavior material, failing closed on opaque code."""

    # Subagent tools and any future built-in wrappers may embed application
    # runtimes, stores, or other executable collaborators. They remain
    # process-local until the application supplies a ToolSpec identity or a
    # bounded extractor is deliberately registered here and owned there.
    return _material_from_exact_type(tool, _cayu_tool_material_extractors())


@lru_cache(maxsize=1)
def _cayu_runner_material_extractors() -> dict[type[object], _ExecutionProfileMaterialExtractor]:
    from cayu.runners.docker import DockerRunner
    from cayu.runners.local import LocalRunner

    return {
        LocalRunner: LocalRunner._execution_profile_material,
        DockerRunner: DockerRunner._execution_profile_material,
    }


def _cayu_runner_material(runner: object) -> dict[str, Any] | None:
    """Return bounded adapter configuration for runners Cayu can identify safely."""

    # Remote SDK runners wrap opaque clients, transports, or provider modules.
    # Their application identity property remains the explicit portability seam.
    return _material_from_exact_type(runner, _cayu_runner_material_extractors())


@lru_cache(maxsize=1)
def _cayu_policy_material_extractors() -> dict[type[object], _ExecutionProfileMaterialExtractor]:
    from cayu.runtime.tool_exposure import (
        AllRegisteredToolsExposurePolicy,
        StaticToolExposurePolicy,
    )
    from cayu.runtime.tool_policy import (
        AllowAllToolPolicy,
        AlwaysRequireApprovalToolPolicy,
        ParameterConstrainedToolPolicy,
        StaticToolPolicy,
        TaintAwareToolPolicy,
    )
    from cayu.tools.command_policy import ProcessCommandPolicy
    from cayu.tools.git_command_policy import GitCommandPolicy
    from cayu.tools.structured_commands import StructuredCommandToolPolicy

    policy_types = (
        AllowAllToolPolicy,
        AlwaysRequireApprovalToolPolicy,
        ParameterConstrainedToolPolicy,
        StaticToolPolicy,
        TaintAwareToolPolicy,
        ProcessCommandPolicy,
        GitCommandPolicy,
        StructuredCommandToolPolicy,
        AllRegisteredToolsExposurePolicy,
        StaticToolExposurePolicy,
    )
    return {policy_type: policy_type._execution_profile_material for policy_type in policy_types}


def _cayu_policy_material(policy: object) -> dict[str, Any] | None:
    return _material_from_exact_type(policy, _cayu_policy_material_extractors())


@dataclass(frozen=True)
class _ContextComponentMaterials:
    selection: dict[str, Any]
    recall: dict[str, Any]
    compaction: dict[str, Any]
    selection_process_local: bool = False
    recall_process_local: bool = False
    compaction_process_local: bool = False
    selection_application_versioned: bool = False
    recall_application_versioned: bool = False
    compaction_application_versioned: bool = False


def _context_component_materials(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    runtime_version: str | None,
    process_identity: str,
    redactor: SecretRedactor,
) -> _ContextComponentMaterials:
    primary = _project_context_policy(
        registered_agent.context_policy,
        identity=registered_agent.context_policy_execution_profile_identity,
        behavior_identities=(registered_agent.context_behavior_execution_profile_identities),
        runtime_version=runtime_version,
        process_identity=process_identity,
        slot="context-policy",
        redactor=redactor,
    )
    overflow_policy = registered_agent.context_overflow_policy
    if overflow_policy is None:
        overflow = _ContextComponentMaterials(
            selection={"kind": "none", "version": 1},
            recall={"kind": "none", "version": 1},
            compaction={"kind": "none", "version": 1},
        )
    else:
        overflow = _project_context_policy(
            overflow_policy,
            identity=registered_agent.context_overflow_policy_execution_profile_identity,
            behavior_identities=(registered_agent.context_behavior_execution_profile_identities),
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot="context-overflow-policy",
            redactor=redactor,
        )
    return _ContextComponentMaterials(
        selection={"primary": primary.selection, "overflow": overflow.selection},
        recall={"primary": primary.recall, "overflow": overflow.recall},
        compaction={"primary": primary.compaction, "overflow": overflow.compaction},
        selection_process_local=(
            primary.selection_process_local or overflow.selection_process_local
        ),
        recall_process_local=(primary.recall_process_local or overflow.recall_process_local),
        compaction_process_local=(
            primary.compaction_process_local or overflow.compaction_process_local
        ),
        selection_application_versioned=(
            primary.selection_application_versioned or overflow.selection_application_versioned
        ),
        recall_application_versioned=(
            primary.recall_application_versioned or overflow.recall_application_versioned
        ),
        compaction_application_versioned=(
            primary.compaction_application_versioned or overflow.compaction_application_versioned
        ),
    )


def _project_context_policy(
    policy: object,
    *,
    identity: ExecutionProfileBehaviorIdentity | None,
    behavior_identities: Mapping[int, ExecutionProfileBehaviorIdentity | None],
    runtime_version: str | None,
    process_identity: str,
    slot: str,
    redactor: SecretRedactor,
) -> _ContextComponentMaterials:
    if identity is not None:
        entry, _ = _behavior_identity_material(
            identity=identity,
            value=policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=slot,
        )
        material = {"behavior": entry}
        return _ContextComponentMaterials(
            selection=material,
            recall=material,
            compaction=material,
            selection_application_versioned=True,
            recall_application_versioned=True,
            compaction_application_versioned=True,
        )

    projected = _cayu_context_policy_material(
        policy,
        behavior_identities=behavior_identities,
        process_identity=process_identity,
    )
    if projected is None:
        entry, _ = _behavior_identity_material(
            identity=None,
            value=policy,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=slot,
        )
        material = {"behavior": entry}
        return _ContextComponentMaterials(
            selection=material,
            recall=material,
            compaction=material,
            selection_process_local=True,
            recall_process_local=True,
            compaction_process_local=True,
        )

    selection_material = dict(projected.selection)
    recall_material = dict(projected.recall)
    compaction_material = dict(projected.compaction)
    if projected.selection_process_local:
        selection_material["process_scope"] = _process_scope_identity(process_identity)
        selection_material["slot"] = f"{slot}:selection"
    if projected.recall_process_local:
        recall_material["process_scope"] = _process_scope_identity(process_identity)
        recall_material["slot"] = f"{slot}:recall"
    if projected.compaction_process_local:
        compaction_material["process_scope"] = _process_scope_identity(process_identity)
        compaction_material["slot"] = f"{slot}:compaction"
    safe_selection = _secret_safe_cayu_owned_material(selection_material, redactor=redactor)
    safe_recall = _secret_safe_cayu_owned_material(recall_material, redactor=redactor)
    safe_compaction = _secret_safe_cayu_owned_material(compaction_material, redactor=redactor)
    private_selection = _process_local_private_material(
        selection_material,
        value=policy,
        process_identity=process_identity,
        slot=f"{slot}:selection",
    )
    private_recall = _process_local_private_material(
        recall_material,
        value=policy,
        process_identity=process_identity,
        slot=f"{slot}:recall",
    )
    private_compaction = _process_local_private_material(
        compaction_material,
        value=policy,
        process_identity=process_identity,
        slot=f"{slot}:compaction",
    )
    return _ContextComponentMaterials(
        selection=(private_selection if safe_selection is None else safe_selection),
        recall=(private_recall if safe_recall is None else safe_recall),
        compaction=(private_compaction if safe_compaction is None else safe_compaction),
        selection_process_local=(projected.selection_process_local or safe_selection is None),
        recall_process_local=(projected.recall_process_local or safe_recall is None),
        compaction_process_local=(projected.compaction_process_local or safe_compaction is None),
        selection_application_versioned=projected.selection_application_versioned,
        recall_application_versioned=projected.recall_application_versioned,
        compaction_application_versioned=projected.compaction_application_versioned,
    )


def _cayu_context_policy_material(
    policy: object,
    *,
    behavior_identities: Mapping[int, ExecutionProfileBehaviorIdentity | None],
    process_identity: str,
) -> _ContextComponentMaterials | None:
    from cayu.runtime.context import (
        _DEFAULT_CHECKPOINT_COMPACTION_SUMMARY_PREFIX,
        CheckpointCompactionContextPolicy,
        DefaultContextPolicy,
        MessageWindowContextPolicy,
        RecentTurnsContextPolicy,
        TranscriptDigestCompactor,
        UsageTriggeredContextPolicy,
    )
    from cayu.runtime.memory_context import AutomaticRecallContextPolicy

    declared_identity = behavior_identities.get(id(policy))
    if declared_identity is not None:
        material = {
            "behavior": {
                "kind": "application_versioned",
                **declared_identity.model_dump(mode="json"),
            }
        }
        return _ContextComponentMaterials(
            selection=material,
            recall=material,
            compaction=material,
            selection_application_versioned=True,
            recall_application_versioned=True,
            compaction_application_versioned=True,
        )

    if type(policy) is DefaultContextPolicy:
        return _ContextComponentMaterials(
            selection={
                "kind": "default",
                "version": 1,
                "max_attachment_results": policy.max_attachment_results,
            },
            recall={"kind": "none", "version": 1},
            compaction={"kind": "none", "version": 1},
        )
    if type(policy) is MessageWindowContextPolicy:
        return _ContextComponentMaterials(
            selection={
                "kind": "message_window",
                "version": 1,
                "max_messages": policy.max_messages,
                "preserve_system": policy.preserve_system,
                "max_attachment_results": policy.max_attachment_results,
            },
            recall={"kind": "none", "version": 1},
            compaction={"kind": "none", "version": 1},
        )
    if type(policy) is RecentTurnsContextPolicy:
        return _ContextComponentMaterials(
            selection={
                "kind": "recent_turns",
                "version": 1,
                "max_user_turns": policy.max_user_turns,
                "preserve_system": policy.preserve_system,
                "max_attachment_results": policy.max_attachment_results,
            },
            recall={"kind": "none", "version": 1},
            compaction={"kind": "none", "version": 1},
        )
    if type(policy) is AutomaticRecallContextPolicy:
        base = _cayu_context_policy_material(
            policy.base_policy,
            behavior_identities=behavior_identities,
            process_identity=process_identity,
        )
        if base is None:
            return None
        return _ContextComponentMaterials(
            selection=base.selection,
            recall={"base": base.recall, "automatic": policy.configuration_material()},
            compaction=base.compaction,
            selection_process_local=base.selection_process_local,
            recall_process_local=base.recall_process_local,
            compaction_process_local=base.compaction_process_local,
            selection_application_versioned=base.selection_application_versioned,
            recall_application_versioned=base.recall_application_versioned,
            compaction_application_versioned=base.compaction_application_versioned,
        )
    if type(policy) is UsageTriggeredContextPolicy:
        base = _cayu_context_policy_material(
            policy.base_policy,
            behavior_identities=behavior_identities,
            process_identity=process_identity,
        )
        triggered = _cayu_context_policy_material(
            policy.triggered_policy,
            behavior_identities=behavior_identities,
            process_identity=process_identity,
        )
        if base is None or triggered is None:
            return None
        selection = {
            "kind": "usage_triggered",
            "version": 1,
            "base": base.selection,
            "triggered": triggered.selection,
            "min_input_tokens": policy.min_input_tokens,
            "trigger_estimated_context_tokens": policy.trigger_estimated_context_tokens,
            "reserved_output_tokens": policy.reserved_output_tokens,
            "verify_estimate_with_provider_count": policy.verify_estimate_with_provider_count,
            "provider_count_threshold_ratio": policy.provider_count_threshold_ratio,
            "provider_count_min_delta_tokens": policy.provider_count_min_delta_tokens,
            "min_total_tokens": policy.min_total_tokens,
            "sticky": policy.sticky,
        }
        return _ContextComponentMaterials(
            selection=selection,
            recall={"base": base.recall, "triggered": triggered.recall},
            compaction={"base": base.compaction, "triggered": triggered.compaction},
            selection_process_local=(
                base.selection_process_local or triggered.selection_process_local
            ),
            recall_process_local=(base.recall_process_local or triggered.recall_process_local),
            compaction_process_local=(
                base.compaction_process_local or triggered.compaction_process_local
            ),
            selection_application_versioned=(
                base.selection_application_versioned or triggered.selection_application_versioned
            ),
            recall_application_versioned=(
                base.recall_application_versioned or triggered.recall_application_versioned
            ),
            compaction_application_versioned=(
                base.compaction_application_versioned or triggered.compaction_application_versioned
            ),
        )
    if type(policy) is CheckpointCompactionContextPolicy:
        compactor = policy.compactor
        private_summary_prefix = (
            policy.summary_prefix != _DEFAULT_CHECKPOINT_COMPACTION_SUMMARY_PREFIX
        )
        compactor_identity = behavior_identities.get(id(compactor))
        if type(compactor) is TranscriptDigestCompactor:
            compaction = {
                "kind": "transcript_digest",
                "version": 2,
                "max_summary_chars": compactor.max_summary_chars,
            }
            compaction_process_local = False
        elif (
            built_in_compactor := _cayu_compactor_material(
                compactor,
                behavior_identities=behavior_identities,
                process_identity=process_identity,
            )
        ) is not None:
            compaction = built_in_compactor
            compaction_process_local = _material_contains_process_local_identity(compaction)
        else:
            if compactor_identity is None:
                compaction, _ = _behavior_identity_material(
                    identity=None,
                    value=compactor,
                    runtime_version=None,
                    process_identity=process_identity,
                    slot="context-compactor",
                )
                compaction_process_local = True
            else:
                compaction = {
                    "kind": "application_versioned",
                    **compactor_identity.model_dump(mode="json"),
                }
                compaction_process_local = False
        selection: dict[str, Any] = {
            "kind": "checkpoint_compaction",
            "version": 2,
            "max_user_turns": policy.max_user_turns,
            "compact_after_messages": policy.compact_after_messages,
            "compact_after_estimated_context_tokens": (
                policy.compact_after_estimated_context_tokens
            ),
            "max_recent_context_tokens": policy.max_recent_context_tokens,
            "reserved_output_tokens": policy.reserved_output_tokens,
            "summary_prefix": (
                {
                    "kind": "process_local_private_configuration",
                    "configuration_hmac_sha256": (
                        _process_local_configuration_commitment(
                            {"summary_prefix": policy.summary_prefix},
                            process_identity=process_identity,
                            field_name="checkpoint_summary_prefix_private_configuration",
                        )
                    ),
                }
                if private_summary_prefix
                else {"kind": "cayu_default", "version": 1}
            ),
            "max_attachment_results": policy.max_attachment_results,
        }
        return _ContextComponentMaterials(
            selection=selection,
            recall={"kind": "none", "version": 1},
            compaction=compaction,
            selection_process_local=private_summary_prefix,
            compaction_process_local=compaction_process_local,
            compaction_application_versioned=(
                compactor_identity is not None
                or _material_contains_application_identity(compaction)
            ),
        )
    return None


def _material_contains_application_identity(value: object) -> bool:
    if type(value) is dict:
        material = cast("dict[object, object]", value)
        if material.get("kind") == "application_versioned":
            return True
        return any(_material_contains_application_identity(item) for item in material.values())
    if type(value) is list:
        return any(_material_contains_application_identity(item) for item in value)
    return False


def _material_contains_process_local_identity(value: object) -> bool:
    if type(value) is dict:
        material = cast("dict[object, object]", value)
        if material.get("kind") in {
            "process_local_private_configuration",
            "process_local_unverifiable",
        }:
            return True
        return any(_material_contains_process_local_identity(item) for item in material.values())
    if type(value) is list:
        return any(_material_contains_process_local_identity(item) for item in value)
    return False


def _process_local_configuration_commitment(
    value: object,
    *,
    process_identity: str,
    field_name: str,
) -> str:
    return hmac.new(
        process_identity.encode("utf-8"),
        canonical_durable_json_bytes(value, field_name),
        sha256,
    ).hexdigest()


def _process_local_private_material(
    material: object,
    *,
    value: object,
    process_identity: str,
    slot: str,
) -> dict[str, Any]:
    """Bind private configuration exactly without retaining its raw representation."""

    return {
        "kind": "process_local_private_configuration",
        "component": _qualified_type_name(value),
        "process_scope": _process_scope_identity(process_identity),
        "slot": slot,
        "configuration_hmac_sha256": _process_local_configuration_commitment(
            material,
            process_identity=process_identity,
            field_name=f"{slot}_private_configuration",
        ),
    }


def _process_local_object_material(
    value: object,
    *,
    process_identity: str,
    slot: str,
) -> dict[str, Any]:
    return {
        "kind": "process_local_unverifiable",
        "component": _qualified_type_name(value),
        "process_scope": _process_scope_identity(process_identity),
        "slot": slot,
        "instance_hmac_sha256": _process_local_configuration_commitment(
            {"object_id": str(id(value))},
            process_identity=process_identity,
            field_name=f"{slot}_instance",
        ),
    }


def _cayu_compactor_material(
    compactor: object,
    *,
    behavior_identities: Mapping[int, ExecutionProfileBehaviorIdentity | None],
    process_identity: str,
) -> dict[str, Any] | None:
    """Identify transparent built-in compactors without retaining prompt content."""

    from cayu.runtime.context import (
        ModelCompactor,
        PromptCacheCompactor,
        TranscriptDigestCompactor,
        default_compaction_prompt,
    )

    if type(compactor) is TranscriptDigestCompactor:
        return {
            "kind": "transcript_digest",
            "version": 2,
            "max_summary_chars": compactor.max_summary_chars,
        }
    if type(compactor) is ModelCompactor:
        default_prompt = (
            "You summarize prior agent session context for a future model call. "
            "Return only the compact summary. Do not call tools."
        )
        provider = _nested_provider_material(
            compactor.provider,
            behavior_identities=behavior_identities,
            process_identity=process_identity,
            slot="model-compactor-provider",
        )
        material = {
            "kind": "model_compactor",
            "version": _MODEL_COMPACTOR_MATERIAL_VERSION,
            "provider": provider,
            "provider_name": compactor._provider_snapshot.provider_name,
            "pricing_provider_name": compactor._provider_snapshot.pricing_provider_name,
            "usage_dialect": compactor._provider_snapshot.usage_dialect.value,
            "model": compactor.model,
            "max_input_chars": compactor.max_input_chars,
            "max_hierarchy_calls": compactor.max_hierarchy_calls,
            "retry_policy": compactor.retry_policy.model_dump(mode="json"),
        }
        if (
            compactor.system_prompt != default_prompt
            or compactor.options
            or compactor.prompt_builder not in (None, default_compaction_prompt)
        ):
            material["private_configuration"] = {
                "kind": "process_local_private_configuration",
                "configuration_hmac_sha256": _process_local_configuration_commitment(
                    {
                        "system_prompt": compactor.system_prompt,
                        "options": compactor.options,
                        "prompt_builder": (
                            None
                            if compactor.prompt_builder is None
                            else {
                                "component": _qualified_type_name(compactor.prompt_builder),
                                "object_id": str(id(compactor.prompt_builder)),
                            }
                        ),
                    },
                    process_identity=process_identity,
                    field_name="model_compactor_private_configuration",
                ),
            }
        return material
    if type(compactor) is PromptCacheCompactor:
        default_instruction = (
            "Summarize the conversation above so a future agent step can continue "
            "with the important context. Preserve concrete user requests, decisions, "
            "files or resources mentioned, tool results, errors, and pending work. "
            "Do not invent facts. Keep the summary concise but specific. "
            "Do not call tools. Return only the summary text."
        )
        provider = _nested_provider_material(
            compactor.provider,
            behavior_identities=behavior_identities,
            process_identity=process_identity,
            slot="prompt-cache-compactor-provider",
        )
        fallback_identity = behavior_identities.get(id(compactor._fallback))
        fallback = (
            {
                "kind": "application_versioned",
                **fallback_identity.model_dump(mode="json"),
            }
            if fallback_identity is not None
            else _cayu_compactor_material(
                compactor._fallback,
                behavior_identities=behavior_identities,
                process_identity=process_identity,
            )
        )
        if fallback is None:
            fallback = _process_local_object_material(
                compactor._fallback,
                process_identity=process_identity,
                slot="prompt-cache-fallback",
            )
        material = {
            "kind": "prompt_cache_compactor",
            "version": _PROMPT_CACHE_COMPACTOR_MATERIAL_VERSION,
            "provider": provider,
            "provider_name": compactor._provider_snapshot.provider_name,
            "pricing_provider_name": compactor._provider_snapshot.pricing_provider_name,
            "usage_dialect": compactor._provider_snapshot.usage_dialect.value,
            "fallback": fallback,
            "retry_policy": compactor.retry_policy.model_dump(mode="json"),
        }
        if (
            compactor.options
            or compactor.model is not None
            or compactor.compaction_instruction != default_instruction
        ):
            material["private_configuration"] = {
                "kind": "process_local_private_configuration",
                "configuration_hmac_sha256": _process_local_configuration_commitment(
                    {
                        "options": compactor.options,
                        "model": compactor.model,
                        "compaction_instruction": compactor.compaction_instruction,
                    },
                    process_identity=process_identity,
                    field_name="prompt_cache_compactor_private_configuration",
                ),
            }
        return material
    return None


def _nested_provider_material(
    provider: object,
    *,
    behavior_identities: Mapping[int, ExecutionProfileBehaviorIdentity | None],
    process_identity: str,
    slot: str,
) -> dict[str, Any]:
    identity = behavior_identities.get(id(provider))
    if identity is not None:
        return {
            "kind": "application_versioned",
            **identity.model_dump(mode="json"),
        }
    built_in = _cayu_provider_material(provider)
    if built_in is not None:
        return built_in
    return _process_local_object_material(
        provider,
        process_identity=process_identity,
        slot=slot,
    )


def _cayu_provider_material(provider: object) -> dict[str, Any] | None:
    """Return bounded behavior material only for transparent built-in adapters."""

    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers.anthropic import (
        DEFAULT_ANTHROPIC_BASE_URL,
        AnthropicProvider,
        HttpxAnthropicTransport,
    )
    from cayu.providers.bedrock import BedrockProvider
    from cayu.providers.chat_completions import (
        DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV,
        DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER,
        DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX,
        DEFAULT_CHAT_COMPLETIONS_BASE_URL,
        ChatCompletionsProvider,
        HttpxChatCompletionsTransport,
    )
    from cayu.providers.openai import (
        DEFAULT_OPENAI_BASE_URL,
        HttpxOpenAITransport,
        OpenAIProvider,
    )
    from cayu.providers.openai_subscription import OpenAISubscriptionProvider

    if type(provider) is ScriptedModelProvider:
        return {
            "adapter": "scripted-model-provider",
            "version": 1,
            "background": provider.provider_operations is not None,
        }

    if type(provider) is OpenAIProvider:
        if type(provider.transport) is not HttpxOpenAITransport or provider.extra_headers:
            return None
        return {
            "adapter": "openai-responses",
            "version": 5,
            "base_url": provider.base_url,
            "default_route": provider.base_url == DEFAULT_OPENAI_BASE_URL,
            "reasoning_state": provider.reasoning_state,
            "background": provider.background,
            "additional_tools_models": sorted(provider.additional_tools_models),
            "client_tool_search_models": sorted(provider.client_tool_search_models),
            "hosted_tool_search_models": sorted(provider.hosted_tool_search_models),
            "timeout_s": provider.timeout_s,
            "stream_idle_timeout_s": provider.stream_idle_timeout_s,
        }
    if type(provider) is ChatCompletionsProvider:
        if type(provider.transport) is not HttpxChatCompletionsTransport or provider.extra_headers:
            return None
        return {
            "adapter": "chat-completions",
            "version": 2,
            "base_url": provider.base_url,
            "endpoint_url": provider.endpoint_url,
            "api_key_env": provider.api_key_env,
            "auth_header": provider.auth_header,
            "auth_value_prefix": provider.auth_value_prefix,
            "allow_http": provider.allow_http,
            "stream_include_usage": provider.stream_include_usage,
            "timeout_s": provider.timeout_s,
            "stream_idle_timeout_s": provider.stream_idle_timeout_s,
            "api_version": provider.api_version,
            "default_route": bool(
                provider.base_url == DEFAULT_CHAT_COMPLETIONS_BASE_URL
                and provider.endpoint_url is None
                and provider.api_key_env == DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV
                and provider.auth_header == DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER
                and provider.auth_value_prefix == DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX
                and not provider.allow_http
                and provider.api_version is None
                and provider.openrouter_http_referer is None
                and provider.openrouter_app_title is None
                and not provider.openrouter_router_metadata
            ),
            "clean_schemas": provider.clean_schemas,
            "strip_additional_properties": provider.strip_additional_properties,
            "document_encoding": provider.document_encoding,
            "usage_dialect": provider.usage_dialect.value,
            # Attribution values are request headers, not durable runtime
            # evidence. Only their presence affects inspectable behavior
            # material so the values never enter profile events or metadata.
            "openrouter_http_referer_configured": (provider.openrouter_http_referer is not None),
            "openrouter_app_title_configured": provider.openrouter_app_title is not None,
            "openrouter_router_metadata": provider.openrouter_router_metadata,
        }
    if type(provider) is AnthropicProvider:
        if (
            type(provider.transport) is not HttpxAnthropicTransport
            or provider.extra_headers
            or provider.credential_proxy is not None
        ):
            return None
        return {
            "adapter": "anthropic-messages",
            "version": 1,
            "base_url": provider.base_url,
            "default_route": provider.base_url == DEFAULT_ANTHROPIC_BASE_URL,
            "credential_mode": ("brokered" if provider.api_key_ref is not None else "direct"),
            "anthropic_version": provider.anthropic_version,
            "max_tokens": provider.max_tokens,
            "timeout_s": provider.timeout_s,
            "stream_idle_timeout_s": provider.stream_idle_timeout_s,
            "cache_policy": (
                None
                if provider.cache_policy is None
                else provider.cache_policy.model_dump(mode="json")
            ),
        }
    if type(provider) is BedrockProvider:
        if any(
            (
                not provider._owns_client,
                provider.region_name is None,
            )
        ):
            return None
        return {
            "adapter": "bedrock-converse-stream",
            "version": 1,
            "region_name": provider.region_name,
            "profile_name": provider.profile_name,
            "endpoint_url": provider.endpoint_url,
            "max_tokens": provider.max_tokens,
            "stream_idle_timeout_s": provider.stream_idle_timeout_s,
            "stream_close_timeout_s": provider.stream_close_timeout_s,
        }
    if type(provider) is OpenAISubscriptionProvider:
        # Authentication and transport collaborators are opaque unless the app
        # declares a stable provider identity.
        return None
    # Vertex configuration includes project and credential-routing authority;
    # it likewise requires an application identity for cross-process reuse.
    return None


def _environment_identity_material(
    *,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    execution_requirements: dict[str, Any],
    runtime_version: str | None,
    process_identity: str,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], bool, bool]:
    if registered_environment is None:
        return (
            {
                "environment": None,
                "execution_requirements": execution_requirements,
            },
            False,
            False,
        )
    environment = registered_environment.environment
    factory_managed = registered_environment.factory_backed
    application_versioned = registered_environment.spec.execution_profile_identity is not None
    environment_entry, process_local = _behavior_identity_material(
        identity=registered_environment.spec.execution_profile_identity,
        value=environment,
        runtime_version=runtime_version,
        process_identity=process_identity,
        slot=f"environment:{registered_environment.spec.name}",
    )
    runner_entry = None
    if environment.runner is not None:
        runner_entry, runner_process_local = _behavior_identity_material(
            identity=registered_environment.runner_execution_profile_identity,
            value=environment.runner,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"runner:{registered_environment.spec.name}",
            cayu_owned_material=_secret_safe_cayu_owned_material(
                _cayu_runner_material(environment.runner),
                redactor=redactor,
            ),
        )
        process_local |= runner_process_local
        application_versioned |= (
            registered_environment.runner_execution_profile_identity is not None
        )
    factory_entry = None
    if registered_environment.factory_backed:
        factory = registered_environment.factory or environment
        factory_entry, factory_process_local = _behavior_identity_material(
            identity=registered_environment.factory_execution_profile_identity,
            value=factory,
            runtime_version=runtime_version,
            process_identity=process_identity,
            slot=f"environment-factory:{registered_environment.spec.name}",
        )
        process_local |= factory_process_local
        application_versioned |= (
            registered_environment.factory_execution_profile_identity is not None
        )
    return (
        {
            "environment": {
                "name": registered_environment.spec.name,
                "behavior": environment_entry,
                "factory_backed": registered_environment.factory_backed,
                "factory": factory_entry,
                "runner": runner_entry,
                "workspace_presence": (
                    "factory_managed"
                    if factory_managed
                    else ("present" if environment.workspace is not None else "none")
                ),
                "binding_presence": (
                    "factory_managed"
                    if factory_managed
                    else ("present" if environment.binding is not None else "none")
                ),
                "runner_presence": (
                    "factory_managed"
                    if factory_managed
                    else ("present" if environment.runner is not None else "none")
                ),
            },
            "execution_requirements": execution_requirements,
        },
        process_local,
        application_versioned,
    )

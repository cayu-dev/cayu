"""Deep execution-profile admission rules shared by resume and recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any
from uuid import uuid4
from weakref import ReferenceType, ref

from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
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
from cayu.runtime.sessions import Session
from cayu.runtime.user_input import pending_user_input_from_checkpoint
from cayu.vaults import SecretRedactor


@dataclass(frozen=True)
class ExecutionProfileContinuationPlan:
    """A reconstructed continuation and any component drift it exposes."""

    snapshot: ActiveInvocationExecutionProfile
    candidate_profile: ExecutionProfileIdentity
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...]


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


def resolve_execution_profile_identity(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    runtime_name: str,
    runtime_version: str | None,
    redactor: SecretRedactor,
    process_identity: str = "standalone-profile-builder",
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
    runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...] = (),
    loop_policies: tuple[Any, ...] = (),
    loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policies: tuple[Any, ...] = (),
    invocation_loop_policy_identities: tuple[ExecutionProfileBehaviorIdentity | None, ...] = (),
    invocation_loop_policy_instance_identities: tuple[str | None, ...] = (),
) -> ExecutionProfileIdentity:
    """Resolve one registered runtime body into its durable profile identity."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("Execution-profile resolution requires a SecretRedactor.")

    direct_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.input_schema,
            "parallel_safe": tool.parallel_safe,
            "effect": tool.effect.value,
            **({"workspace_mutation": True} if tool.workspace_mutation else {}),
        }
        for tool in registered_agent.tool_capabilities
    ]
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
    return build_execution_profile_identity(
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        provider_name=provider_name,
        model=model,
        durable_system_prompt=durable_system_prompt,
        direct_tools=direct_tools,
        tool_implementations=tool_implementations,
        tool_implementations_process_local=tool_implementations_process_local,
        tool_implementations_application_versioned=(tool_implementations_application_versioned),
        tool_view_grants={
            "view_kind": "direct",
            "generation": 1,
            "grant_baseline": list(registered_agent.tools),
        },
        execution_policies={
            "tool_policy": tool_policy_entry,
            "command_policies": command_policy_material,
            "loop_policies": loop_policy_material,
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
            "credential_authority": credential_authority,
            "egress_authority": registered_agent.execution_requirements.network_access,
            "real_secret_visibility": (
                registered_agent.execution_requirements.real_secret_visibility
            ),
            "runner_authority": runner_authority,
        },
    )


def prepare_execution_profile_continuation(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    runtime_version: str | None,
    redactor: SecretRedactor,
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
            "process_identity": process_identity,
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
    from cayu.tools.search import SearchTextTool
    from cayu.tools.user_input import UserInputTool
    from cayu.tools.web import WebFetchTool

    tool_types = (
        ExecCommandTool,
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
        ScreenshotPageTool,
        UserInputTool,
        WebFetchTool,
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
    from cayu.runtime.tool_policy import (
        AllowAllToolPolicy,
        AlwaysRequireApprovalToolPolicy,
        ParameterConstrainedToolPolicy,
        StaticToolPolicy,
        TaintAwareToolPolicy,
    )
    from cayu.tools.command_policy import ProcessCommandPolicy
    from cayu.tools.git_command_policy import GitCommandPolicy

    policy_types = (
        AllowAllToolPolicy,
        AlwaysRequireApprovalToolPolicy,
        ParameterConstrainedToolPolicy,
        StaticToolPolicy,
        TaintAwareToolPolicy,
        ProcessCommandPolicy,
        GitCommandPolicy,
    )
    return {policy_type: policy_type._execution_profile_material for policy_type in policy_types}


def _cayu_policy_material(policy: object) -> dict[str, Any] | None:
    return _material_from_exact_type(policy, _cayu_policy_material_extractors())


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

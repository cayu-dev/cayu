"""Runtime profile construction, policy interfaces and admission diagnostics."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from cayu._validation import (
    canonical_durable_json_bytes,
)
from cayu.build_provenance import (
    RuntimeBuildProvenance,
    legacy_runtime_build_provenance,
)
from cayu.egress.authority import (
    EgressAuthorityIdentity,
)
from cayu.execution_profiles import _AUTHORITY_COMPONENT_CLASSES as _AUTHORITY_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_COMPONENT_CLASSES as _SCHEMA_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_V1_COMPONENT_CLASSES as _SCHEMA_V1_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_V2_COMPONENT_CLASSES as _SCHEMA_V2_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_V4_COMPONENT_CLASSES as _SCHEMA_V4_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_V5_COMPONENT_CLASSES as _SCHEMA_V5_COMPONENT_CLASSES
from cayu.execution_profiles import _SCHEMA_V6_COMPONENT_CLASSES as _SCHEMA_V6_COMPONENT_CLASSES

# Preserve established imports and legacy pickle paths with identical objects.
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS as EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS,
)
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS as EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
)
from cayu.execution_profiles import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD as EXECUTION_PROFILE_FINGERPRINT_FIELD,
)
from cayu.execution_profiles import (
    EXECUTION_PROFILE_SCHEMA_VERSION as EXECUTION_PROFILE_SCHEMA_VERSION,
)
from cayu.execution_profiles import ExecutionProfileAdoptionIntent as ExecutionProfileAdoptionIntent
from cayu.execution_profiles import (
    ExecutionProfileAuthorityDecision as ExecutionProfileAuthorityDecision,
)
from cayu.execution_profiles import ExecutionProfileComponentClass as ExecutionProfileComponentClass
from cayu.execution_profiles import (
    ExecutionProfileComponentIdentity as ExecutionProfileComponentIdentity,
)
from cayu.execution_profiles import ExecutionProfileDecision as ExecutionProfileDecision
from cayu.execution_profiles import ExecutionProfileDecisionKind as ExecutionProfileDecisionKind
from cayu.execution_profiles import ExecutionProfileIdentity as ExecutionProfileIdentity
from cayu.execution_profiles import (
    ExecutionProfileIdentityAvailability as ExecutionProfileIdentityAvailability,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentityStrength as ExecutionProfileIdentityStrength,
)
from cayu.execution_profiles import ExecutionProfilePolicyAction as ExecutionProfilePolicyAction
from cayu.execution_profiles import ExecutionProfilePolicyRequest as ExecutionProfilePolicyRequest
from cayu.execution_profiles import ExecutionProfilePolicyResult as ExecutionProfilePolicyResult
from cayu.execution_profiles import (
    ExecutionProfileRejectionResult as ExecutionProfileRejectionResult,
)
from cayu.execution_profiles import ModelFailoverCandidateProfile as ModelFailoverCandidateProfile
from cayu.execution_profiles import ModelFailoverProfileBinding as ModelFailoverProfileBinding
from cayu.execution_profiles import _available_component as _available_component
from cayu.execution_profiles import _egress_authority_component as _egress_authority_component
from cayu.execution_profiles import _profile_fingerprint as _profile_fingerprint
from cayu.execution_profiles import _unavailable_component as _unavailable_component
from cayu.execution_profiles import (
    changed_execution_profile_components as changed_execution_profile_components,
)
from cayu.execution_profiles import (
    copy_execution_profile_adoption_intent as copy_execution_profile_adoption_intent,
)
from cayu.execution_profiles import (
    copy_execution_profile_decision as copy_execution_profile_decision,
)
from cayu.execution_profiles import (
    copy_execution_profile_policy_result as copy_execution_profile_policy_result,
)
from cayu.execution_profiles import (
    direct_tool_capability_ceiling_component as direct_tool_capability_ceiling_component,
)
from cayu.execution_profiles import (
    event_with_execution_profile_authority as event_with_execution_profile_authority,
)
from cayu.execution_profiles import (
    event_with_execution_profile_fingerprint_authority as event_with_execution_profile_fingerprint_authority,
)
from cayu.execution_profiles import (
    execution_profile_changes_authority as execution_profile_changes_authority,
)
from cayu.execution_profiles import (
    execution_profile_decision_payload as execution_profile_decision_payload,
)
from cayu.execution_profiles import (
    execution_profile_egress_authority_change as execution_profile_egress_authority_change,
)
from cayu.execution_profiles import (
    execution_profile_provider_target_component as execution_profile_provider_target_component,
)
from cayu.execution_profiles import (
    execution_profile_runtime_component as execution_profile_runtime_component,
)
from cayu.execution_profiles import (
    execution_profile_with_component as execution_profile_with_component,
)
from cayu.execution_profiles import (
    execution_profile_with_durable_system_projection_digest as execution_profile_with_durable_system_projection_digest,
)
from cayu.execution_profiles import (
    execution_profile_with_egress_authority as execution_profile_with_egress_authority,
)
from cayu.execution_profiles import (
    execution_profile_with_model_failover as execution_profile_with_model_failover,
)
from cayu.execution_profiles import (
    execution_profile_with_tool_capability_ceiling as execution_profile_with_tool_capability_ceiling,
)
from cayu.execution_profiles import (
    inherited_execution_profile_component_changes as inherited_execution_profile_component_changes,
)
from cayu.execution_profiles import (
    unavailable_execution_profile_components as unavailable_execution_profile_components,
)
from cayu.sessions._execution_profile_checkpoint import (
    _ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE as _ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE,
)
from cayu.sessions._execution_profile_checkpoint import (
    _ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION as _ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION,
)
from cayu.sessions._execution_profile_checkpoint import (
    _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION as _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION,
)
from cayu.sessions._execution_profile_checkpoint import (
    _EXECUTION_PROFILE_RECORD_TYPE as _EXECUTION_PROFILE_RECORD_TYPE,
)
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY as EXECUTION_PROFILE_METADATA_KEY,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile as ActiveInvocationExecutionProfile,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint as active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_is_released as active_invocation_execution_profile_is_released,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_matches_session_epoch as active_invocation_execution_profile_matches_session_epoch,
)
from cayu.sessions._execution_profile_checkpoint import (
    checkpoint_with_active_invocation_execution_profile as checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_baseline_from_session_metadata as execution_profile_baseline_from_session_metadata,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_from_session_metadata as execution_profile_from_session_metadata,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_metadata_after_adoption as execution_profile_metadata_after_adoption,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_session_metadata as execution_profile_session_metadata,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY as ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
)
from cayu.tools.catalogue import (
    TOOL_CATALOGUE_MAX_TOOLS,
    validate_canonical_tool_id,
    validate_tool_catalogue_revision,
    validate_tool_descriptor_version,
)


class ExecutionProfilePolicy(ABC):
    """Application-owned compatibility and adoption authority."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable, versioned, non-secret policy identity."""

    @abstractmethod
    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        """Classify one non-equal profile before governed work begins."""


_RUNTIME_EXECUTION_PROFILE_DECISIONS: dict[
    int,
    tuple[weakref.ReferenceType[ExecutionProfileDecision], str],
] = {}


def _execution_profile_decision_authority_digest(
    decision: ExecutionProfileDecision,
) -> str:
    material = decision.model_dump(mode="json", warnings="error")
    return sha256(
        canonical_durable_json_bytes(material, "runtime_execution_profile_decision")
    ).hexdigest()


def _with_runtime_execution_profile_decision_authority(
    decision: ExecutionProfileDecision,
) -> ExecutionProfileDecision:
    """Attest one decision produced by the runtime policy boundary."""

    if type(decision) is not ExecutionProfileDecision:
        raise TypeError("decision must be an ExecutionProfileDecision.")
    identity = id(decision)

    def forget(reference: weakref.ReferenceType[ExecutionProfileDecision]) -> None:
        current = _RUNTIME_EXECUTION_PROFILE_DECISIONS.get(identity)
        if current is not None and current[0] is reference:
            _RUNTIME_EXECUTION_PROFILE_DECISIONS.pop(identity, None)

    reference = weakref.ref(decision, forget)
    _RUNTIME_EXECUTION_PROFILE_DECISIONS[identity] = (
        reference,
        _execution_profile_decision_authority_digest(decision),
    )
    return decision


def _has_runtime_execution_profile_decision_authority(
    decision: ExecutionProfileDecision,
) -> bool:
    """Return authority that cannot be copied from the public model itself."""

    attestation = _RUNTIME_EXECUTION_PROFILE_DECISIONS.get(id(decision))
    if attestation is None or attestation[0]() is not decision:
        return False
    try:
        observed_digest = _execution_profile_decision_authority_digest(decision)
    except Exception:
        return False
    return observed_digest == attestation[1]


class _ExecutionProfileAdmissionRequestRejected(RuntimeError):
    """Private signal for deterministic request rejection before admission."""


class ExecutionProfileDifference(BaseModel):
    """Bounded class-level evidence; profile digests do not identify members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_class: ExecutionProfileComponentClass
    category: Literal["opaque_identity", "other_or_unknown"]


def _profile_differences(
    changed: tuple[ExecutionProfileComponentClass, ...],
    expected: ExecutionProfileIdentity | None,
    candidate: ExecutionProfileIdentity | None,
    expected_fingerprint: str,
    candidate_fingerprint: str,
) -> tuple[ExecutionProfileDifference, ...]:
    # Only exact runtime value objects may participate. Never inspect tools,
    # services, configuration, reprs, or user-provided identity hooks here.
    profiles = (expected, candidate)
    trusted = (
        all(type(profile) is ExecutionProfileIdentity for profile in profiles)
        and expected is not None
        and candidate is not None
        and expected.fingerprint == expected_fingerprint
        and candidate.fingerprint == candidate_fingerprint
    )
    opaque = set()
    if trusted:
        for profile in profiles:
            assert profile is not None
            for component in profile.components:
                if type(component) is ExecutionProfileComponentIdentity and (
                    component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
                ):
                    opaque.add(component.component_class)
    return tuple(
        ExecutionProfileDifference(
            component_class=component,
            category="opaque_identity" if component in opaque else "other_or_unknown",
        )
        for component in changed
    )


class ExecutionProfileMismatchError(RuntimeError):
    """Raised after durable evidence rejects a changed execution profile."""

    def __init__(
        self,
        *,
        session_id: str,
        expected_profile_fingerprint: str,
        candidate_profile_fingerprint: str,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
        expected_profile: ExecutionProfileIdentity | None = None,
        candidate_profile: ExecutionProfileIdentity | None = None,
    ) -> None:
        self.session_id = session_id
        self.expected_profile_fingerprint = expected_profile_fingerprint
        self.candidate_profile_fingerprint = candidate_profile_fingerprint
        self.changed_component_classes = changed_component_classes
        changed = ", ".join(component.value for component in changed_component_classes) or (
            "decision-bearing authority outside the structural profile"
        )
        self.differences = _profile_differences(
            changed_component_classes,
            expected_profile,
            candidate_profile,
            expected_profile_fingerprint,
            candidate_profile_fingerprint,
        )
        guidance = (
            " Class-level digest evidence cannot identify individual components, "
            "changed declared versions, or additions/removals. Inspect the persisted "
            "execution-profile decision and your component declarations; "
            "see `cayu guide durable-service-tools`."
        )
        opaque = ", ".join(
            item.component_class.value
            for item in self.differences
            if item.category == "opaque_identity"
        )
        if opaque:
            guidance += (
                f" Process-local (opaque) identity is present in: {opaque}. "
                "Reconstructed components need explicit stable behavior and implementation "
                "identities declared from the first run. Adding a declaration does not "
                "repair an already persisted opaque baseline."
            )
        if ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS in changed_component_classes:
            guidance += (
                " If you added or edited a tool: give every custom tool "
                "`execution_profile_identity=ExecutionProfileBehaviorIdentity(name=..., "
                'behavior_version="1", implementation_version="1")` in its ToolSpec, as '
                "`cayu generate tool` does, bump behavior_version when its behavior "
                "changes, and start a new session for the changed tool set."
            )
        super().__init__(self._message(session_id=session_id, changed=changed) + guidance)

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution profile changed in: {changed}. "
            "Pending recovery requires a restored compatible registration and a fresh recovery plan. "
            "Start a new session or, at a clean resume boundary, request explicit profile adoption."
        )


class ExecutionProfileAdoptionRejected(ExecutionProfileMismatchError):
    """Raised after policy durably rejects an explicit adoption request."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution-profile adoption was rejected for changes in: "
            f"{changed}. Inspect the durable decision evidence or start a new session."
        )


class ExecutionProfileMigrationRequired(ExecutionProfileMismatchError):
    """Raised after policy records that an explicit migration is required."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} requires an execution-profile migration before changes in: "
            f"{changed}. No resumed work was admitted."
        )


class ExecutionProfilePolicyError(RuntimeError):
    """Raised when configured profile policy cannot produce an authoritative decision."""


def _aggregate_identity_strength(
    *,
    process_local: bool,
    application_versioned: bool,
) -> ExecutionProfileIdentityStrength:
    if process_local:
        return ExecutionProfileIdentityStrength.PROCESS_LOCAL
    if application_versioned:
        return ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    return ExecutionProfileIdentityStrength.STRUCTURAL


def _canonical_direct_tool_material(
    direct_tools: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(direct_tools, str | bytes | bytearray | Mapping | BaseModel):
        raise TypeError("direct_tools must be an iterable of canonical tool identities.")
    material: list[dict[str, str]] = []
    for index, item in enumerate(direct_tools):
        if index >= TOOL_CATALOGUE_MAX_TOOLS:
            raise ValueError(
                f"direct_tools cannot contain more than {TOOL_CATALOGUE_MAX_TOOLS} items."
            )
        if not isinstance(item, Mapping):
            raise TypeError(f"direct_tools[{index}] must be a mapping.")
        if set(item) != {"tool_id", "descriptor_version"}:
            raise ValueError(
                f"direct_tools[{index}] must contain exactly tool_id and descriptor_version."
            )
        material.append(
            {
                "tool_id": validate_canonical_tool_id(
                    item["tool_id"],
                    f"direct_tools[{index}].tool_id",
                ),
                "descriptor_version": validate_tool_descriptor_version(
                    item["descriptor_version"],
                    f"direct_tools[{index}].descriptor_version",
                ),
            }
        )
    tool_ids = tuple(item["tool_id"] for item in material)
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("direct_tools must contain unique tool_id values.")
    return material


def build_execution_profile_identity(
    *,
    runtime_name: str,
    runtime_version: str | None,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    direct_tools: Iterable[Mapping[str, Any]],
    tool_catalogue_revision: str,
    tool_implementations: Iterable[Mapping[str, Any]] | None = None,
    tool_implementations_process_local: bool = False,
    tool_implementations_application_versioned: bool = False,
    tool_view_grants: Mapping[str, Any] | None = None,
    execution_policies: Mapping[str, Any] | None = None,
    execution_policies_process_local: bool = False,
    execution_policies_application_versioned: bool = False,
    invocation_policies: Iterable[Mapping[str, Any]] = (),
    invocation_policies_process_local: bool = False,
    invocation_policies_application_versioned: bool = False,
    runtime_hooks: Iterable[Mapping[str, Any]] = (),
    runtime_hooks_process_local: bool = False,
    runtime_hooks_application_versioned: bool = False,
    execution_environment: Mapping[str, Any] | None = None,
    execution_environment_process_local: bool = False,
    execution_environment_application_versioned: bool = False,
    effect_authority: Mapping[str, Any] | None = None,
    egress_authority: EgressAuthorityIdentity | None = None,
    context_selection: Mapping[str, Any] | None = None,
    context_selection_process_local: bool = False,
    context_selection_application_versioned: bool = False,
    automatic_recall: Mapping[str, Any] | None = None,
    automatic_recall_process_local: bool = False,
    automatic_recall_application_versioned: bool = False,
    context_compaction: Mapping[str, Any] | None = None,
    context_compaction_process_local: bool = False,
    context_compaction_application_versioned: bool = False,
    live_state_projection: Mapping[str, Any] | None = None,
    provider_adapter: Mapping[str, Any] | None = None,
    provider_adapter_process_local: bool = False,
    provider_adapter_application_versioned: bool = False,
    provider_request_policy: Mapping[str, Any] | None = None,
    provider_request_policy_process_local: bool = False,
    provider_request_policy_application_versioned: bool = False,
    application_budget_policy: Mapping[str, Any] | None = None,
    invocation_budget_policy: Mapping[str, Any] | None = None,
    structured_output: Mapping[str, Any] | None = None,
    finalization: Mapping[str, Any] | None = None,
    runtime_build_provenance: RuntimeBuildProvenance | None = None,
) -> ExecutionProfileIdentity:
    """Build a profile without retaining raw prompts, schemas, or tool names."""

    direct_tool_material = _canonical_direct_tool_material(direct_tools)
    tool_catalogue_revision = validate_tool_catalogue_revision(
        tool_catalogue_revision,
        "tool_catalogue_revision",
    )
    direct_tool_identity_material = {
        "kind": "cayu:catalogued-direct-tools",
        "version": 1,
        "catalogue_revision": tool_catalogue_revision,
        "tools": direct_tool_material,
    }
    if tool_view_grants is None and direct_tool_material:
        raise ValueError("tool_view_grants must be provided when direct_tools is non-empty.")
    implementation_material = (
        [{"tool_id": item["tool_id"]} for item in direct_tool_material]
        if tool_implementations is None
        else list(tool_implementations)
    )
    components = (
        _available_component(
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            direct_tool_identity_material,
        ),
        _available_component(
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
            _aggregate_identity_strength(
                process_local=tool_implementations_process_local,
                application_versioned=tool_implementations_application_versioned,
            ),
            implementation_material,
        ),
        _available_component(
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            (
                {
                    "view_kind": "direct",
                    "generation": 1,
                    "grant_baseline": [],
                }
                if tool_view_grants is None
                else tool_view_grants
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.EXECUTION_POLICIES,
            _aggregate_identity_strength(
                process_local=execution_policies_process_local,
                application_versioned=execution_policies_application_versioned,
            ),
            (
                {
                    "tool_policy": "cayu:allow-all-tool-policy:v1",
                    "command_policies": [],
                    "loop_policies": [],
                }
                if execution_policies is None
                else execution_policies
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
            _aggregate_identity_strength(
                process_local=invocation_policies_process_local,
                application_versioned=invocation_policies_application_versioned,
            ),
            list(invocation_policies),
        ),
        _available_component(
            ExecutionProfileComponentClass.RUNTIME_HOOKS,
            _aggregate_identity_strength(
                process_local=runtime_hooks_process_local,
                application_versioned=runtime_hooks_application_versioned,
            ),
            list(runtime_hooks),
        ),
        _available_component(
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
            _aggregate_identity_strength(
                process_local=execution_environment_process_local,
                application_versioned=execution_environment_application_versioned,
            ),
            (
                {
                    "environment": None,
                    "execution_requirements": {
                        "code_trust": "trusted",
                        "real_secret_visibility": "allowed",
                        "network_access": "unrestricted",
                        "guest_privilege": "unrestricted",
                        "host_filesystem": "unrestricted",
                        "cancellation": "best_effort",
                        "cleanup": "best_effort",
                        "durability": "ephemeral",
                        "minimum_evidence": "declared",
                        "evidence_overrides": [],
                        "required_executables": [],
                    },
                }
                if execution_environment is None
                else execution_environment
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            (
                {
                    "tool_effects": [
                        {
                            "name": item.get("name"),
                            "effect": item.get("effect"),
                            "workspace_mutation": bool(item.get("workspace_mutation", False)),
                        }
                        for item in direct_tool_material
                    ],
                    "credential_authority": "none",
                    "egress_authority": "unrestricted",
                }
                if effect_authority is None
                else effect_authority
            ),
        ),
        _egress_authority_component(egress_authority),
        _available_component(
            ExecutionProfileComponentClass.CONTEXT_SELECTION,
            _aggregate_identity_strength(
                process_local=context_selection_process_local,
                application_versioned=context_selection_application_versioned,
            ),
            {"kind": "cayu:default-context-selection:v1"}
            if context_selection is None
            else context_selection,
        ),
        _available_component(
            ExecutionProfileComponentClass.AUTOMATIC_RECALL,
            _aggregate_identity_strength(
                process_local=automatic_recall_process_local,
                application_versioned=automatic_recall_application_versioned,
            ),
            {"kind": "none", "version": 1} if automatic_recall is None else automatic_recall,
        ),
        _available_component(
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
            _aggregate_identity_strength(
                process_local=context_compaction_process_local,
                application_versioned=context_compaction_application_versioned,
            ),
            {"kind": "none", "version": 1} if context_compaction is None else context_compaction,
        ),
        _available_component(
            ExecutionProfileComponentClass.LIVE_STATE_PROJECTION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "none", "version": 1}
            if live_state_projection is None
            else live_state_projection,
        ),
        execution_profile_provider_adapter_component(
            provider_adapter,
            process_local=provider_adapter_process_local,
            application_versioned=provider_adapter_application_versioned,
        ),
        _available_component(
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
            _aggregate_identity_strength(
                process_local=provider_request_policy_process_local,
                application_versioned=provider_request_policy_application_versioned,
            ),
            {"kind": "provider-defaults", "version": 1}
            if provider_request_policy is None
            else provider_request_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"limit_ids": []} if application_budget_policy is None else application_budget_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"limit_ids": []} if invocation_budget_policy is None else invocation_budget_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "none", "version": 1} if structured_output is None else structured_output,
        ),
        _available_component(
            ExecutionProfileComponentClass.FINALIZATION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "cayu:default-finalization:v1"} if finalization is None else finalization,
        ),
        _available_component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"system_prompt": durable_system_prompt},
        ),
        execution_profile_provider_target_component(provider_name, model),
        execution_profile_runtime_component(
            runtime_name,
            runtime_version,
            runtime_build_provenance,
        ),
    )
    sorted_components = tuple(sorted(components, key=lambda component: component.component_class))
    return ExecutionProfileIdentity(
        fingerprint=_profile_fingerprint(
            sorted_components,
            schema_version=EXECUTION_PROFILE_SCHEMA_VERSION,
            runtime_build_provenance=(
                legacy_runtime_build_provenance()
                if runtime_build_provenance is None
                else runtime_build_provenance
            ),
        ),
        components=sorted_components,
        egress_authority=egress_authority,
        runtime_build_provenance=(
            legacy_runtime_build_provenance()
            if runtime_build_provenance is None
            else runtime_build_provenance
        ),
    )


def execution_profile_provider_adapter_component(
    material: Mapping[str, Any] | None,
    *,
    process_local: bool,
    application_versioned: bool,
) -> ExecutionProfileComponentIdentity:
    """Build the same provider component for admission and live dispatch checks."""

    return _available_component(
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        _aggregate_identity_strength(
            process_local=process_local, application_versioned=application_versioned
        ),
        {"kind": "provider-defaults", "version": 1} if material is None else material,
    )

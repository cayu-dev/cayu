"""Runtime profile construction, policy interfaces and admission diagnostics."""

from __future__ import annotations

import json
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
    """Bounded class-level evidence; profile digests do not identify members.

    ``expected_strength`` and ``candidate_strength`` report how the stored and
    current profiles identified this class. They are ``None`` when that profile
    does not record the class or is not authoritative runtime evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_class: ExecutionProfileComponentClass
    category: Literal["opaque_identity", "other_or_unknown"]
    expected_strength: ExecutionProfileIdentityStrength | None = None
    candidate_strength: ExecutionProfileIdentityStrength | None = None


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
    strengths: tuple[
        dict[ExecutionProfileComponentClass, ExecutionProfileIdentityStrength],
        dict[ExecutionProfileComponentClass, ExecutionProfileIdentityStrength],
    ] = ({}, {})
    if trusted:
        for profile, by_class in zip(profiles, strengths, strict=True):
            assert profile is not None
            for component in profile.components:
                if type(component) is not ExecutionProfileComponentIdentity:
                    continue
                by_class[component.component_class] = component.strength
                if component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL:
                    opaque.add(component.component_class)
    return tuple(
        ExecutionProfileDifference(
            component_class=component,
            category="opaque_identity" if component in opaque else "other_or_unknown",
            expected_strength=strengths[0].get(component),
            candidate_strength=strengths[1].get(component),
        )
        for component in changed
    )


# Fixed, non-secret wording only. Messages never include component names,
# declared identity strings, configuration, or object representations.
_COMPONENT_LABELS: dict[ExecutionProfileComponentClass, str] = {
    ExecutionProfileComponentClass.RUNTIME: "Cayu version or build",
    ExecutionProfileComponentClass.PROVIDER_TARGET: "provider name and model",
    ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION: "stored system prompt projection",
    ExecutionProfileComponentClass.DIRECT_TOOLS: "the agent's tool set and tool descriptors",
    ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS: "tool implementations",
    ExecutionProfileComponentClass.TOOL_VIEW_GRANTS: "tool capability ceiling and grants",
    ExecutionProfileComponentClass.EXECUTION_POLICIES: (
        "tool, command, exposure and app or agent loop policies"
    ),
    ExecutionProfileComponentClass.INVOCATION_POLICIES: "request loop_policies",
    ExecutionProfileComponentClass.RUNTIME_HOOKS: "runtime hooks",
    ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT: "environment and runner",
    ExecutionProfileComponentClass.EFFECT_AUTHORITY: "tool effects and credential authority",
    ExecutionProfileComponentClass.EGRESS_AUTHORITY: "egress authority",
    ExecutionProfileComponentClass.CONTEXT_SELECTION: "context policy",
    ExecutionProfileComponentClass.AUTOMATIC_RECALL: "automatic recall",
    ExecutionProfileComponentClass.CONTEXT_COMPACTION: "context compaction",
    ExecutionProfileComponentClass.LIVE_STATE_PROJECTION: "live state projection",
    ExecutionProfileComponentClass.PROVIDER_ADAPTER: "model provider adapter",
    ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY: (
        "provider options, thinking and failover"
    ),
    ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY: "application budget limits",
    ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY: "request budget limits",
    ExecutionProfileComponentClass.STRUCTURED_OUTPUT: "structured output",
    ExecutionProfileComponentClass.FINALIZATION: (
        "max_steps, run limits, retry policy and tool completion"
    ),
}

_STRENGTH_LABELS: dict[ExecutionProfileIdentityStrength, str] = {
    ExecutionProfileIdentityStrength.APPLICATION_VERSIONED: "declared",
    ExecutionProfileIdentityStrength.STRUCTURAL: "structural",
    ExecutionProfileIdentityStrength.PROCESS_LOCAL: "opaque",
    ExecutionProfileIdentityStrength.UNAVAILABLE: "unavailable",
}

_MAX_LISTED_DIFFERENCES = 8
_MAX_SNIPPET_SESSION_ID_CHARS = 128

_TOOL_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.DIRECT_TOOLS,
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        ExecutionProfileComponentClass.EFFECT_AUTHORITY,
    }
)
_POLICY_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.EXECUTION_POLICIES,
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ExecutionProfileComponentClass.RUNTIME_HOOKS,
    }
)
_PROVIDER_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
    }
)
_OTHER_COMPONENT_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        ExecutionProfileComponentClass.EGRESS_AUTHORITY,
        ExecutionProfileComponentClass.CONTEXT_SELECTION,
        ExecutionProfileComponentClass.AUTOMATIC_RECALL,
        ExecutionProfileComponentClass.CONTEXT_COMPACTION,
        ExecutionProfileComponentClass.LIVE_STATE_PROJECTION,
        ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileComponentClass.FINALIZATION,
        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
    }
)

_DECLARE_IDENTITY = (
    'ExecutionProfileBehaviorIdentity(name=..., behavior_version="1", implementation_version="1")'
)


def _difference_interpretation(difference: ExecutionProfileDifference) -> str:
    """Explain one class change from its two identity strengths only."""

    stored, current = difference.expected_strength, difference.candidate_strength
    if stored is None or current is None:
        return "Cayu cannot tell more from the stored evidence."
    opaque = ExecutionProfileIdentityStrength.PROCESS_LOCAL
    if ExecutionProfileIdentityStrength.UNAVAILABLE in (stored, current):
        return "Its identity could not be resolved, so it cannot be compared."
    if stored is opaque and current is opaque:
        return (
            "An undeclared component gets a new identity in every process and CayuApp "
            "instance (a request loop policy, for every new policy object), so this "
            "differs even when your code did not change."
        )
    if current is opaque:
        return "A component without a declared identity was added, or a declaration was removed."
    if stored is opaque:
        return (
            "The stored identity is opaque, so no later process can match it; declaring "
            "an identity now does not repair this session."
        )
    declared = ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    if stored is declared and current is declared:
        return (
            "A declared name or version changed, or a component was added, removed or reconfigured."
        )
    if stored is declared:
        return "A component with a declared identity was removed, or lost its declaration."
    if current is declared:
        return "A component with a declared identity was added."
    return "Its configuration changed, or a component was added or removed."


def _difference_lines(differences: tuple[ExecutionProfileDifference, ...]) -> list[str]:
    lines = []
    for difference in differences[:_MAX_LISTED_DIFFERENCES]:
        label = _COMPONENT_LABELS.get(difference.component_class, "component")
        stored, current = difference.expected_strength, difference.candidate_strength
        kinds = (
            ""
            if stored is None or current is None
            else f" {_STRENGTH_LABELS[stored]} -> {_STRENGTH_LABELS[current]}:"
        )
        lines.append(
            f"- {difference.component_class.value} ({label}):{kinds} "
            f"{_difference_interpretation(difference)}"
        )
    remaining = len(differences) - _MAX_LISTED_DIFFERENCES
    if remaining > 0:
        lines.append(f"- and {remaining} more component classes.")
    return lines


def _component_guidance(changed: frozenset[ExecutionProfileComponentClass]) -> list[str]:
    """Return class-specific fixes. Every changed class receives one line."""

    guidance = []
    if changed & _TOOL_CLASSES:
        guidance.append(
            "Tools: if you added or edited a tool, give every custom tool "
            f"`execution_profile_identity={_DECLARE_IDENTITY}` in its ToolSpec, as "
            "`cayu generate tool` does, and bump behavior_version when its behavior "
            "changes. Adding, removing or redescribing a tool changes the session's "
            "tool authority, so the changed tool set needs a new session or adoption."
        )
    if changed & _POLICY_CLASSES:
        line = (
            "Policies and hooks: a custom LoopPolicy, ToolPolicy, CommandPolicy or "
            "RuntimeHook without a declared identity is opaque. Override its "
            f"`execution_profile_identity` property to return `{_DECLARE_IDENTITY}` "
            "(built-ins such as RequireFinalTool already declare one)."
        )
        if ExecutionProfileComponentClass.INVOCATION_POLICIES in changed:
            line += (
                " Request loop_policies are part of the session's profile: pass the same "
                "policies on every RunRequest and ResumeRequest of the session, including "
                "in each simulated process of a test."
            )
        guidance.append(line)
    if changed & _PROVIDER_CLASSES:
        guidance.append(
            "Provider: built-in providers and `ScriptedModelProvider` built from "
            "positional event batches keep the same identity in every process. "
            "`ScriptedModelProvider(response_factory=...)`, provider subclasses and "
            "custom ModelProvider classes are opaque unless they override "
            "`execution_profile_identity`. In a test that simulates several processes, "
            "build the provider the same way in each one, with the same options, "
            "thinking and failover settings."
        )
    if ExecutionProfileComponentClass.PROVIDER_TARGET in changed:
        guidance.append(
            "Model: the session's provider or model differs from what the agent resolves "
            "to now. Register the same provider name and model, or switch explicitly with "
            "`ResumeRequest(..., target=ModelTarget(provider_name=..., model=...))` "
            "(`from cayu.sessions import ModelTarget`)."
        )
    if ExecutionProfileComponentClass.RUNTIME in changed:
        guidance.append(
            "Cayu: the Cayu version or build changed. When that is the only change and "
            "no ExecutionProfilePolicy is configured, resume adopts it automatically at "
            "a clean boundary."
        )
    if changed & _OTHER_COMPONENT_CLASSES:
        guidance.append(
            "Other components: custom environments, runners, environment factories and "
            "context policies declare `execution_profile_identity` the same way. "
            "Structural settings such as budget limits, structured output, max_steps, "
            "run limits and egress must match the stored session."
        )
    return guidance


def _snippet_session_id(session_id: object) -> str:
    if (
        type(session_id) is str
        and 0 < len(session_id) <= _MAX_SNIPPET_SESSION_ID_CHARS
        and session_id.isprintable()
    ):
        return json.dumps(session_id)
    return "..."


ExecutionProfileMismatchBoundary = Literal["resume", "continuation"]


class ExecutionProfileMismatchError(RuntimeError):
    """Raised after durable evidence rejects a changed execution profile.

    ``boundary`` is ``"resume"`` when the rejected attempt was an ordinary resume
    with nothing pending, ``"continuation"`` when pending work had to continue
    under its recorded profile, and ``None`` when the raising path does not know.
    ``application_policy_configured`` reports whether the ``CayuApp`` had an
    ``ExecutionProfilePolicy``, or ``None`` when unknown. Both only shape the
    guidance; they never change admission.
    """

    def __init__(
        self,
        *,
        session_id: str,
        expected_profile_fingerprint: str,
        candidate_profile_fingerprint: str,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
        expected_profile: ExecutionProfileIdentity | None = None,
        candidate_profile: ExecutionProfileIdentity | None = None,
        boundary: ExecutionProfileMismatchBoundary | None = None,
        application_policy_configured: bool | None = None,
    ) -> None:
        self.session_id = session_id
        self.expected_profile_fingerprint = expected_profile_fingerprint
        self.candidate_profile_fingerprint = candidate_profile_fingerprint
        self.changed_component_classes = changed_component_classes
        self.boundary = boundary if boundary in ("resume", "continuation") else None
        self.application_policy_configured = (
            application_policy_configured if type(application_policy_configured) is bool else None
        )
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
        lines = [self._message(session_id=session_id, changed=changed)]
        if self.differences:
            lines.append(
                "What changed (stored -> current identity). Class-level digest evidence "
                "cannot name the individual tool, policy or version:"
            )
            lines.extend(_difference_lines(self.differences))
        opaque = ", ".join(
            item.component_class.value
            for item in self.differences
            if item.category == "opaque_identity"
        )
        if opaque:
            lines.append(
                f"Process-local (opaque) identity is present in: {opaque}. Reconstructed "
                "components need explicit stable behavior and implementation identities "
                "declared from the first run. Adding a declaration does not repair an "
                "already persisted opaque baseline. Adopting a profile that is still "
                "opaque only lasts until the next process or CayuApp instance."
            )
        lines.extend(_component_guidance(frozenset(changed_component_classes)))
        lines.extend(self._next_steps(session_id=session_id))
        lines.append(
            "Details: `cayu guide diagnostics#execution-profile-mismatch`; component "
            "identities: `cayu guide durable-service-tools`."
        )
        super().__init__("\n".join(lines))

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution profile changed in: {changed}. "
            "No resumed work was admitted."
        )

    def _new_session_step(self) -> str:
        return (
            "- Start a new session: "
            'app.run(RunRequest(agent_name=..., session_id="<new id>", messages=[...])).'
        )

    def _adoption_steps(self, *, session_id: str) -> list[str]:
        steps = [
            "- Or keep this session: at a clean resume boundary (nothing pending), adopt "
            "the current profile with an ExecutionProfilePolicy on "
            "CayuApp(execution_profile_policy=...) that adopts explicit requests, then "
            f"app.resume(ResumeRequest(session_id={_snippet_session_id(session_id)}, "
            "messages=[...], profile_adoption=ExecutionProfileAdoptionIntent("
            'idempotency_key="adopt-1", reason="...", requested_by=ResolutionActor('
            'subject="...", source=ResolutionActorSource.REQUEST)))).'
        ]
        if self.application_policy_configured is False:
            steps.append(
                "- This CayuApp has no ExecutionProfilePolicy, so an adoption request is "
                "rejected until you configure one."
            )
        return steps

    def _next_steps(self, *, session_id: str) -> list[str]:
        if self.boundary == "continuation":
            return [
                "Next steps: this session has pending work (an approval, user input, tool "
                "round or unfinished invocation) that must continue under the profile it "
                "started with, so adoption is not available here.",
                "- Restore the previous code or identities and finish the pending work; a "
                "later resume can then adopt the new profile. Or",
                self._new_session_step(),
            ]
        steps = [
            "Next steps:",
            self._new_session_step(),
            *self._adoption_steps(session_id=session_id),
        ]
        if self.boundary is None:
            steps.append(
                "- If the session has pending work, adoption is not available: pending "
                "recovery requires a restored compatible registration and a fresh "
                "recovery plan."
            )
        return steps


class ExecutionProfileAdoptionRejected(ExecutionProfileMismatchError):
    """Raised after policy durably rejects an explicit adoption request."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution-profile adoption was rejected for changes in: "
            f"{changed}. Inspect the durable decision evidence or start a new session."
        )

    def _next_steps(self, *, session_id: str) -> list[str]:
        if self.application_policy_configured is False:
            reason = (
                "- This CayuApp has no ExecutionProfilePolicy, and without one Cayu rejects "
                "every adoption request. Configure CayuApp(execution_profile_policy=...) "
                "with a policy that returns adopt for explicit requests (and "
                "authority_decision=authorized when authority changes), then retry with a "
                "new idempotency_key."
            )
        elif self.application_policy_configured is True:
            reason = (
                "- The configured ExecutionProfilePolicy rejected it. Adoption needs "
                "action=adopt, plus authority_decision=authorized when authority changes."
            )
        else:
            reason = (
                "- Adoption needs an ExecutionProfilePolicy that returns adopt, plus "
                "authority_decision=authorized when authority changes."
            )
        return ["Next steps:", reason, self._new_session_step()]


class ExecutionProfileMigrationRequired(ExecutionProfileMismatchError):
    """Raised after policy records that an explicit migration is required."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} requires an execution-profile migration before changes in: "
            f"{changed}. No resumed work was admitted."
        )

    def _next_steps(self, *, session_id: str) -> list[str]:
        return [
            "Next steps: the application's ExecutionProfilePolicy classified this change "
            "as needing a migration. Follow the application's migration path, or",
            self._new_session_step(),
        ]


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

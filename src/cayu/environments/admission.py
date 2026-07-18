"""Capability-based admission for explicitly selected execution environments."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from cayu.capabilities import (
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityIdentity,
    CapabilityObservation,
    CapabilityState,
)

EXECUTION_CAPABILITY_EVIDENCE_SCHEMA = "cayu.execution_capabilities.v1"
EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS = 300

ExecutionCodeTrust = Literal["trusted", "untrusted"]
ExecutionSecretVisibility = Literal["allowed", "non_possession"]
ExecutionNetworkAccess = Literal["unrestricted", "deny_by_default", "brokered_egress"]
ExecutionGuestPrivilege = Literal["unrestricted", "contained", "unprivileged"]
ExecutionHostFilesystem = Literal["unrestricted", "isolated", "read_only_inputs"]
ExecutionCancellation = Literal["best_effort", "confirmed"]
ExecutionCleanup = Literal["best_effort", "confirmed"]
ExecutionDurability = Literal["ephemeral", "reconnectable"]
MinimumExecutionEvidence = Literal["declared", "available", "live_verified"]
ExecutionAdmissionStage = Literal["pre_create", "pre_exposure"]
ExecutionAdmissionStatus = Literal["admitted", "refused"]
ExecutionAdmissionRefusalCode = Literal[
    "malformed_evidence",
    "evidence_candidate_mismatch",
    "unclaimed_evidence",
    "missing_capability",
    "unverified_capability",
    "unsupported_capability",
    "insufficient_evidence",
    "stale_evidence",
    "future_evidence",
    "overlong_evidence",
    "contradictory_evidence",
]
ExecutionObservedCapabilityState = (
    CapabilityState
    | Literal[
        "missing",
        "unclaimed",
        "stale",
        "malformed",
        "mismatched",
    ]
)
_CAPABILITY_IDENTITY_ADAPTER = TypeAdapter(CapabilityIdentity)
_MAX_EVIDENCE_CLOCK_SKEW = timedelta(seconds=30)
_MAX_LIVE_EVIDENCE_TTL = timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
_POSITIVE_EVIDENCE_RANK: dict[str, int] = {
    "declared": 1,
    "available": 2,
    "live_verified": 3,
}
_REQUIRED_LIVE_OBSERVATIONS: dict[str, CapabilityObservation] = {
    "untrusted_code_isolation": "supported",
    "real_credential_non_possession": "supported",
    "deny_by_default_network": "denied",
    "brokered_egress": "reachable",
    "guest_privilege_containment": "supported",
    "unprivileged_guest": "supported",
    "host_filesystem_isolation": "supported",
    "read_only_host_inputs": "supported",
    "confirmed_cancellation": "supported",
    "confirmed_cleanup": "supported",
    "reconnect": "supported",
}


class ExecutionCapabilityClaim(CapabilityClaim):
    """One bounded capability fact used by execution admission."""

    @classmethod
    def declared(cls, capability: str) -> Self:
        return cls(
            capability=capability,
            state="declared",
            proof_source="integration_declaration",
            observation="supported",
        )

    @classmethod
    def available(cls, capability: str) -> Self:
        return cls(
            capability=capability,
            state="available",
            proof_source="integration_validation",
            observation="available",
        )

    @classmethod
    def live_verified(
        cls,
        capability: str,
        *,
        observation: Literal["denied", "reachable", "supported"],
        observed_at: datetime,
        valid_until: datetime,
    ) -> Self:
        return cls(
            capability=capability,
            state="live_verified",
            proof_source="runtime_preflight",
            observation=observation,
            observed_at=observed_at,
            valid_until=valid_until,
        )

    @classmethod
    def unverified(
        cls,
        capability: str,
        *,
        reason_code: str,
        remediation_code: str,
    ) -> Self:
        return cls(
            capability=capability,
            state="unverified",
            proof_source="operator_opt_out",
            observation="not_probed",
            reason_code=reason_code,
            remediation_code=remediation_code,
        )

    @classmethod
    def unsupported(
        cls,
        capability: str,
        *,
        reason_code: str,
        remediation_code: str,
    ) -> Self:
        return cls(
            capability=capability,
            state="unsupported",
            proof_source="integration_declaration",
            observation="unavailable",
            reason_code=reason_code,
            remediation_code=remediation_code,
        )


class ExecutionCapabilityEvidence(CapabilityEvidence):
    """Versioned evidence for one explicitly selected execution candidate."""

    schema_version: Literal["cayu.execution_capabilities.v1"] = Field(
        default=EXECUTION_CAPABILITY_EVIDENCE_SCHEMA,
        alias="schema",
    )
    claims: tuple[ExecutionCapabilityClaim, ...] = Field(
        default_factory=tuple,
        max_length=64,
    )

    def claim_for(self, capability: str) -> ExecutionCapabilityClaim | None:
        return next(
            (claim for claim in self.claims if claim.capability == capability),
            None,
        )


class ExecutionAdmissionCandidate(BaseModel):
    """Explicit execution identity and evidence supplied by its integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: CapabilityIdentity
    evidence: ExecutionCapabilityEvidence

    @model_validator(mode="after")
    def validate_evidence_subject(self) -> Self:
        if self.evidence.subject != self.candidate:
            raise ValueError("Execution admission evidence must describe its named candidate.")
        return self


class ExecutionEvidenceOverride(BaseModel):
    """A capability-specific minimum that overrides the workload default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: CapabilityIdentity
    minimum_evidence: MinimumExecutionEvidence


class ExecutionRequirements(BaseModel):
    """Provider-neutral security and lifecycle requirements for one workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_trust: ExecutionCodeTrust = "trusted"
    real_secret_visibility: ExecutionSecretVisibility = "allowed"
    network_access: ExecutionNetworkAccess = "unrestricted"
    guest_privilege: ExecutionGuestPrivilege = "unrestricted"
    host_filesystem: ExecutionHostFilesystem = "unrestricted"
    cancellation: ExecutionCancellation = "best_effort"
    cleanup: ExecutionCleanup = "best_effort"
    durability: ExecutionDurability = "ephemeral"
    minimum_evidence: MinimumExecutionEvidence = "declared"
    evidence_overrides: tuple[ExecutionEvidenceOverride, ...] = Field(
        default_factory=tuple,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_evidence_overrides(self) -> Self:
        capabilities = [override.capability for override in self.evidence_overrides]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Execution evidence overrides must name unique capabilities.")
        required_capabilities = set(self.required_capabilities())
        unused = sorted(set(capabilities) - required_capabilities)
        if unused:
            raise ValueError(
                "Execution evidence overrides must name required capabilities: " + ", ".join(unused)
            )
        return self

    @classmethod
    def trusted(cls, **overrides: object) -> Self:
        """Build requirements for an explicitly trusted workload."""

        return cls.model_validate(overrides)

    @classmethod
    def untrusted(cls, **overrides: object) -> Self:
        """Build fail-closed defaults for model-authored or otherwise untrusted code."""

        values: dict[str, object] = {
            "code_trust": "untrusted",
            "real_secret_visibility": "non_possession",
            "network_access": "deny_by_default",
            "guest_privilege": "contained",
            "host_filesystem": "isolated",
            "cancellation": "confirmed",
            "cleanup": "confirmed",
            "durability": "ephemeral",
            "minimum_evidence": "available",
        }
        values.update(overrides)
        return cls.model_validate(values)

    def required_capabilities(self) -> tuple[str, ...]:
        """Project the configured workload policy into explicit capability names."""

        capabilities: list[str] = []
        if self.code_trust == "untrusted":
            capabilities.append("untrusted_code_isolation")
        if self.real_secret_visibility == "non_possession":
            capabilities.append("real_credential_non_possession")
        if self.network_access in {"deny_by_default", "brokered_egress"}:
            capabilities.append("deny_by_default_network")
        if self.network_access == "brokered_egress":
            capabilities.append("brokered_egress")
        if self.guest_privilege in {"contained", "unprivileged"}:
            capabilities.append("guest_privilege_containment")
        if self.guest_privilege == "unprivileged":
            capabilities.append("unprivileged_guest")
        if self.host_filesystem in {"isolated", "read_only_inputs"}:
            capabilities.append("host_filesystem_isolation")
        if self.host_filesystem == "read_only_inputs":
            capabilities.append("read_only_host_inputs")
        if self.cancellation == "confirmed":
            capabilities.append("confirmed_cancellation")
        if self.cleanup == "confirmed":
            capabilities.append("confirmed_cleanup")
        if self.durability == "reconnectable":
            capabilities.append("reconnect")
        return tuple(capabilities)

    def minimum_evidence_for(self, capability: str) -> MinimumExecutionEvidence:
        """Return the configured evidence minimum for one required capability."""

        override = next(
            (
                item.minimum_evidence
                for item in self.evidence_overrides
                if item.capability == capability
            ),
            None,
        )
        return self.minimum_evidence if override is None else override


class ExecutionAdmissionRefusal(BaseModel):
    """One stable reason an explicit execution candidate was not admitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ExecutionAdmissionRefusalCode
    capability: CapabilityIdentity | None = None
    required_state: MinimumExecutionEvidence | None = None
    observed_state: ExecutionObservedCapabilityState | None = None
    reason_code: CapabilityIdentity | None = None
    remediation_code: CapabilityIdentity | None = None


class ExecutionAdmissionDecision(BaseModel):
    """Complete admitted/refused result for one explicit candidate and stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ExecutionAdmissionStatus
    stage: ExecutionAdmissionStage
    candidate: CapabilityIdentity
    requirements: ExecutionRequirements
    evidence_schema: str | None = None
    evidence: ExecutionCapabilityEvidence | None = None
    refusals: tuple[ExecutionAdmissionRefusal, ...] = ()

    @model_validator(mode="after")
    def validate_status_refusals(self) -> Self:
        if self.status == "admitted" and self.refusals:
            raise ValueError("Admitted admission status cannot include refusals.")
        if self.status == "refused" and not self.refusals:
            raise ValueError("Refused admission status requires at least one refusal.")
        return self

    def require_admitted(self) -> Self:
        """Return an admitted decision or raise its structured refusal."""

        if self.status == "refused":
            raise ExecutionAdmissionError(self)
        return self


class ExecutionAdmissionError(RuntimeError):
    """An execution candidate failed capability-based admission."""

    def __init__(self, decision: ExecutionAdmissionDecision) -> None:
        self.decision = decision
        capabilities = ", ".join(
            refusal.capability for refusal in decision.refusals if refusal.capability is not None
        )
        detail = f": {capabilities}" if capabilities else ""
        super().__init__(
            f"Execution candidate {decision.candidate!r} was refused during "
            f"{decision.stage}{detail}."
        )


def evaluate_execution_admission(
    *,
    candidate: str,
    requirements: ExecutionRequirements,
    evidence: ExecutionCapabilityEvidence | Mapping[str, object] | None,
    stage: ExecutionAdmissionStage = "pre_exposure",
    now: datetime | None = None,
) -> ExecutionAdmissionDecision:
    """Evaluate one explicit candidate without selecting or falling back to another."""

    validated_candidate = _CAPABILITY_IDENTITY_ADAPTER.validate_python(candidate)
    if not isinstance(requirements, ExecutionRequirements):
        raise TypeError("requirements must be ExecutionRequirements.")
    if stage not in {"pre_create", "pre_exposure"}:
        raise ValueError("stage must be 'pre_create' or 'pre_exposure'.")
    required_capabilities = requirements.required_capabilities()
    if evidence is None:
        if not required_capabilities:
            return ExecutionAdmissionDecision(
                status="admitted",
                stage=stage,
                candidate=validated_candidate,
                requirements=requirements,
            )
        return _refused_for_each_requirement(
            candidate=validated_candidate,
            requirements=requirements,
            stage=stage,
            code="missing_capability",
            observed_state="missing",
        )

    try:
        if isinstance(evidence, ExecutionCapabilityEvidence):
            validated_evidence = ExecutionCapabilityEvidence.model_validate(
                evidence.model_dump(mode="python", by_alias=True)
            )
        elif isinstance(evidence, Mapping):
            validated_evidence = ExecutionCapabilityEvidence.model_validate(dict(evidence))
        else:
            raise TypeError("evidence must be ExecutionCapabilityEvidence, a mapping, or None.")
    except (TypeError, ValidationError, ValueError):
        return _refused_for_each_requirement(
            candidate=validated_candidate,
            requirements=requirements,
            stage=stage,
            code="malformed_evidence",
            observed_state="malformed",
        )

    if validated_evidence.subject != validated_candidate:
        return _refused_for_each_requirement(
            candidate=validated_candidate,
            requirements=requirements,
            stage=stage,
            code="evidence_candidate_mismatch",
            observed_state="mismatched",
            evidence_schema=validated_evidence.schema_version,
            evidence=validated_evidence,
        )

    if not required_capabilities:
        return ExecutionAdmissionDecision(
            status="admitted",
            stage=stage,
            candidate=validated_candidate,
            requirements=requirements,
            evidence_schema=validated_evidence.schema_version,
            evidence=validated_evidence,
        )

    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise ValueError("Admission time must include a timezone.")
    refusals: list[ExecutionAdmissionRefusal] = []
    for capability in required_capabilities:
        required_state: MinimumExecutionEvidence = (
            "declared" if stage == "pre_create" else requirements.minimum_evidence_for(capability)
        )
        claim = validated_evidence.claim_for(capability)
        if claim is None:
            refusals.append(
                ExecutionAdmissionRefusal(
                    code=(
                        "unclaimed_evidence"
                        if validated_evidence.unclaimed_reason_code is not None
                        else "missing_capability"
                    ),
                    capability=capability,
                    required_state=required_state,
                    observed_state=(
                        "unclaimed"
                        if validated_evidence.unclaimed_reason_code is not None
                        else "missing"
                    ),
                    reason_code=validated_evidence.unclaimed_reason_code,
                )
            )
            continue
        if claim.state == "unverified":
            refusals.append(
                _claim_refusal(
                    claim,
                    code="unverified_capability",
                    required_state=required_state,
                )
            )
            continue
        if claim.state == "unsupported":
            refusals.append(
                _claim_refusal(
                    claim,
                    code="unsupported_capability",
                    required_state=required_state,
                )
            )
            continue
        if (
            claim.state == "live_verified"
            and claim.observed_at is not None
            and claim.observed_at > checked_at + _MAX_EVIDENCE_CLOCK_SKEW
        ):
            refusals.append(
                _claim_refusal(
                    claim,
                    code="future_evidence",
                    required_state=required_state,
                )
            )
            continue
        if (
            claim.state == "live_verified"
            and claim.valid_until is not None
            and claim.valid_until <= checked_at
        ):
            refusals.append(
                _claim_refusal(
                    claim,
                    code="stale_evidence",
                    required_state=required_state,
                    observed_state="stale",
                )
            )
            continue
        if (
            claim.state == "live_verified"
            and claim.valid_until is not None
            and claim.observed_at is not None
            and claim.valid_until - claim.observed_at > _MAX_LIVE_EVIDENCE_TTL
        ):
            refusals.append(
                _claim_refusal(
                    claim,
                    code="overlong_evidence",
                    required_state=required_state,
                )
            )
            continue
        if claim.state == "available" and claim.observation not in {
            "available",
            "supported",
        }:
            # Availability proves that an integration path exists; observations
            # such as reachable or denied assert a concrete runtime condition
            # and therefore require live-verified evidence. In particular,
            # reachability cannot prove deny-by-default networking.
            refusals.append(
                _claim_refusal(
                    claim,
                    code="contradictory_evidence",
                    required_state=required_state,
                )
            )
            continue
        required_observation = _REQUIRED_LIVE_OBSERVATIONS[capability]
        if claim.state == "live_verified" and claim.observation != required_observation:
            refusals.append(
                _claim_refusal(
                    claim,
                    code="contradictory_evidence",
                    required_state=required_state,
                )
            )
            continue
        if _POSITIVE_EVIDENCE_RANK[claim.state] < _POSITIVE_EVIDENCE_RANK[required_state]:
            refusals.append(
                _claim_refusal(
                    claim,
                    code="insufficient_evidence",
                    required_state=required_state,
                )
            )

    return ExecutionAdmissionDecision(
        status="refused" if refusals else "admitted",
        stage=stage,
        candidate=validated_candidate,
        requirements=requirements,
        evidence_schema=validated_evidence.schema_version,
        evidence=validated_evidence,
        refusals=tuple(refusals),
    )


def _claim_refusal(
    claim: ExecutionCapabilityClaim,
    *,
    code: ExecutionAdmissionRefusalCode,
    required_state: MinimumExecutionEvidence,
    observed_state: ExecutionObservedCapabilityState | None = None,
) -> ExecutionAdmissionRefusal:
    return ExecutionAdmissionRefusal(
        code=code,
        capability=claim.capability,
        required_state=required_state,
        observed_state=observed_state or claim.state,
        reason_code=claim.reason_code,
        remediation_code=claim.remediation_code,
    )


def _refused_for_each_requirement(
    *,
    candidate: str,
    requirements: ExecutionRequirements,
    stage: ExecutionAdmissionStage,
    code: ExecutionAdmissionRefusalCode,
    observed_state: ExecutionObservedCapabilityState,
    evidence_schema: str | None = None,
    evidence: ExecutionCapabilityEvidence | None = None,
) -> ExecutionAdmissionDecision:
    refusals = tuple(
        ExecutionAdmissionRefusal(
            code=code,
            capability=capability,
            required_state=(
                "declared"
                if stage == "pre_create"
                else requirements.minimum_evidence_for(capability)
            ),
            observed_state=observed_state,
        )
        for capability in requirements.required_capabilities()
    )
    if not refusals:
        refusals = (ExecutionAdmissionRefusal(code=code, observed_state=observed_state),)
    return ExecutionAdmissionDecision(
        status="refused",
        stage=stage,
        candidate=candidate,
        requirements=requirements,
        evidence_schema=evidence_schema,
        evidence=evidence,
        refusals=refusals,
    )

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

EGRESS_CAPABILITY_EVIDENCE_SCHEMA = "cayu.egress_capabilities.v1"
MAX_EGRESS_CAPABILITY_CLAIMS = 64
MAX_SAFE_JSON_INTEGER = 2**53 - 1
MIN_SAFE_JSON_INTEGER = -MAX_SAFE_JSON_INTEGER

EgressCapabilityState = Literal[
    "verified",
    "unverified",
    "unsupported",
    "unclaimed",
]
EgressCapabilityClaimState = Literal[
    "verified",
    "unverified",
    "unsupported",
]
EgressCapabilityProofSource = Literal[
    "adapter_declaration",
    "agent_preflight",
    "external_live_verification",
    "operator_opt_out",
]
EgressCapabilityObservation = Literal[
    "denied",
    "not_probed",
    "reachable",
    "supported",
    "unavailable",
]
EgressCapabilityReasonCode = Literal[
    "capability_unsupported",
    "guest_process_boundary_unverified",
]
EgressCapabilityRemediationCode = Literal[
    "supply_enforceable_guest_boundary",
    "use_supported_configuration",
]
EgressUnclaimedReasonCode = Literal["adapter_capabilities_unclaimed"]
_ALLOWED_CLAIM_PROOFS: dict[
    EgressCapabilityClaimState,
    frozenset[tuple[EgressCapabilityProofSource, EgressCapabilityObservation]],
] = {
    "verified": frozenset(
        (source, observation)
        for source in ("agent_preflight", "external_live_verification")
        for observation in ("denied", "reachable", "supported")
    ),
    "unverified": frozenset({("operator_opt_out", "not_probed")}),
    "unsupported": frozenset({("adapter_declaration", "unavailable")}),
}
EvidenceToken = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=96,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]


def _reject_secret_shaped_identity(value: str) -> str:
    parts = set(value.replace("-", "_").split("_"))
    if value.startswith(("ghp_", "github_pat_", "pk_", "sk_", "xox")) or parts.intersection(
        {"password", "secret"}
    ):
        raise ValueError("Evidence identities cannot contain secret-shaped values.")
    return value


EvidenceIdentity = Annotated[EvidenceToken, AfterValidator(_reject_secret_shaped_identity)]
SafeJsonInteger = Annotated[
    StrictInt,
    Field(ge=MIN_SAFE_JSON_INTEGER, le=MAX_SAFE_JSON_INTEGER),
]
EgressCapabilityDetailValue = StrictBool | SafeJsonInteger


class EgressCapabilityDetail(BaseModel):
    """One bounded adapter-specific fact attached to a capability claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: EvidenceIdentity
    value: EgressCapabilityDetailValue


class EgressCapabilityClaim(BaseModel):
    """One bounded, secret-safe runtime capability observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: EvidenceIdentity
    state: EgressCapabilityClaimState
    proof_source: EgressCapabilityProofSource
    observation: EgressCapabilityObservation | None = None
    reason_code: EgressCapabilityReasonCode | None = None
    remediation_code: EgressCapabilityRemediationCode | None = None
    adapter_details: tuple[EgressCapabilityDetail, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_state_fields(self) -> Self:
        detail_names = [detail.name for detail in self.adapter_details]
        if len(detail_names) != len(set(detail_names)):
            raise ValueError("Capability adapter details must have unique names.")
        proof = (self.proof_source, self.observation)
        if proof not in _ALLOWED_CLAIM_PROOFS[self.state]:
            raise ValueError(
                f"{self.state} capability claims cannot combine proof_source "
                f"{self.proof_source!r} with observation {self.observation!r}."
            )
        if self.state == "verified":
            if self.reason_code is not None or self.remediation_code is not None:
                raise ValueError(
                    "Verified capability claims cannot define reason or remediation codes."
                )
            return self
        if self.reason_code is None:
            raise ValueError(f"{self.state} capability claims require a reason_code.")
        if self.state == "unsupported" and self.remediation_code is None:
            raise ValueError("Unsupported capability claims require a remediation_code.")
        return self


class EgressCapabilityEvidence(BaseModel):
    """Versioned runtime proof published by one virtual-egress adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["cayu.egress_capabilities.v1"] = Field(
        default=EGRESS_CAPABILITY_EVIDENCE_SCHEMA,
        alias="schema",
    )
    adapter: EvidenceIdentity
    claims: tuple[EgressCapabilityClaim, ...] = Field(
        default_factory=tuple,
        max_length=MAX_EGRESS_CAPABILITY_CLAIMS,
    )
    unclaimed_reason_code: EgressUnclaimedReasonCode | None = None

    @model_validator(mode="after")
    def validate_claims(self) -> Self:
        capabilities = [claim.capability for claim in self.claims]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Egress capability claims must have unique capability names.")
        if self.claims and self.unclaimed_reason_code is not None:
            raise ValueError("Claimed evidence cannot define unclaimed_reason_code.")
        if not self.claims and self.unclaimed_reason_code is None:
            raise ValueError("Evidence without claims requires unclaimed_reason_code.")
        return self

    @classmethod
    def unclaimed(
        cls,
        adapter: str,
        *,
        reason_code: EgressUnclaimedReasonCode = "adapter_capabilities_unclaimed",
    ) -> EgressCapabilityEvidence:
        return cls(adapter=adapter, unclaimed_reason_code=reason_code)

    def claim_for(self, capability: str) -> EgressCapabilityClaim | None:
        return next(
            (claim for claim in self.claims if claim.capability == capability),
            None,
        )

    def state_for(self, capability: str) -> EgressCapabilityState:
        claim = self.claim_for(capability)
        return "unclaimed" if claim is None else claim.state

    def to_metadata(self) -> dict[str, object]:
        claims: list[dict[str, object]] = []
        for claim in sorted(self.claims, key=lambda item: item.capability):
            claim_metadata = claim.model_dump(
                mode="json",
                exclude={"adapter_details"},
                exclude_none=True,
            )
            if claim.adapter_details:
                claim_metadata["adapter_details"] = [
                    detail.model_dump(mode="json")
                    for detail in sorted(claim.adapter_details, key=lambda item: item.name)
                ]
            claims.append(claim_metadata)
        metadata: dict[str, object] = {
            "schema": self.schema_version,
            "adapter": self.adapter,
            "claims": claims,
        }
        if self.unclaimed_reason_code is not None:
            metadata["unclaimed_reason_code"] = self.unclaimed_reason_code
        return metadata

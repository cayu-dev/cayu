"""Shared, bounded capability-evidence vocabulary.

Domain-specific evidence envelopes, including virtual egress and execution
admission, build on these primitives so they retain one secret-safe identity,
detail, proof-state, and freshness discipline.
"""

from __future__ import annotations

from datetime import datetime
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

MAX_CAPABILITY_CLAIMS = 64
MAX_CAPABILITY_DETAILS = 16
MAX_SAFE_JSON_INTEGER = 2**53 - 1
MIN_SAFE_JSON_INTEGER = -MAX_SAFE_JSON_INTEGER

CapabilityState = Literal[
    "declared",
    "available",
    "live_verified",
    "unverified",
    "unsupported",
]
CapabilityProofSource = Literal[
    "integration_declaration",
    "integration_validation",
    "process_preflight",
    "runtime_preflight",
    "external_live_verification",
    "operator_opt_out",
]
CapabilityObservation = Literal[
    "supported",
    "available",
    "denied",
    "reachable",
    "not_probed",
    "unavailable",
]
_ALLOWED_PROOFS: dict[
    CapabilityState,
    frozenset[tuple[CapabilityProofSource, CapabilityObservation]],
] = {
    "declared": frozenset({("integration_declaration", "supported")}),
    "available": frozenset(
        {
            ("integration_validation", "available"),
            ("process_preflight", "available"),
            ("process_preflight", "reachable"),
            ("process_preflight", "supported"),
        }
    ),
    "live_verified": frozenset(
        (source, observation)
        for source in ("runtime_preflight", "external_live_verification")
        for observation in ("denied", "reachable", "supported")
    ),
    "unverified": frozenset({("operator_opt_out", "not_probed")}),
    "unsupported": frozenset({("integration_declaration", "unavailable")}),
}

CapabilityToken = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=96,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
CapabilitySchema = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=96,
        pattern=r"^cayu\.[a-z][a-z0-9_.-]*\.v[1-9][0-9]*$",
    ),
]


def _reject_secret_shaped_identity(value: str) -> str:
    parts = set(value.replace("-", "_").split("_"))
    if value.startswith(("ghp_", "github_pat_", "pk_", "sk_", "xox")) or parts.intersection(
        {"password", "secret"}
    ):
        raise ValueError("Evidence identities cannot contain secret-shaped values.")
    return value


CapabilityIdentity = Annotated[
    CapabilityToken,
    AfterValidator(_reject_secret_shaped_identity),
]
CapabilitySafeJsonInteger = Annotated[
    StrictInt,
    Field(ge=MIN_SAFE_JSON_INTEGER, le=MAX_SAFE_JSON_INTEGER),
]
CapabilityDetailValue = StrictBool | CapabilitySafeJsonInteger


class CapabilityDetail(BaseModel):
    """One bounded integration-specific boolean or integer proof fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: CapabilityIdentity
    value: CapabilityDetailValue


class CapabilityClaimBase(BaseModel):
    """Shared bounded fields for domain-specific capability claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: CapabilityIdentity
    adapter_details: tuple[CapabilityDetail, ...] = Field(
        default_factory=tuple,
        max_length=MAX_CAPABILITY_DETAILS,
    )

    @model_validator(mode="after")
    def validate_detail_names(self) -> Self:
        detail_names = [detail.name for detail in self.adapter_details]
        if len(detail_names) != len(set(detail_names)):
            raise ValueError("Capability adapter details must have unique names.")
        return self


class CapabilityClaim(CapabilityClaimBase):
    """One declared, available, live-verified, or negative capability fact."""

    state: CapabilityState
    proof_source: CapabilityProofSource
    observation: CapabilityObservation
    reason_code: CapabilityIdentity | None = None
    remediation_code: CapabilityIdentity | None = None
    observed_at: datetime | None = None
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_state_fields(self) -> Self:
        proof = (self.proof_source, self.observation)
        if proof not in _ALLOWED_PROOFS[self.state]:
            raise ValueError(
                f"{self.state} capability claims cannot combine proof_source "
                f"{self.proof_source!r} with observation {self.observation!r}."
            )
        if self.state in {"declared", "available", "live_verified"}:
            if self.reason_code is not None or self.remediation_code is not None:
                raise ValueError(
                    f"{self.state} capability claims cannot define reason or remediation codes."
                )
        elif self.reason_code is None or self.remediation_code is None:
            raise ValueError(
                f"{self.state} capability claims require reason and remediation codes."
            )

        if self.state == "live_verified":
            if self.observed_at is None or self.valid_until is None:
                raise ValueError(
                    "Live-verified capability claims require observed_at and valid_until."
                )
            if self.observed_at.tzinfo is None or self.valid_until.tzinfo is None:
                raise ValueError("Capability evidence timestamps must include a timezone.")
            if self.valid_until <= self.observed_at:
                raise ValueError("Capability evidence valid_until must follow observed_at.")
        elif self.observed_at is not None or self.valid_until is not None:
            raise ValueError(
                f"{self.state} capability claims cannot define live-verification timestamps."
            )
        return self


class CapabilityEvidence(BaseModel):
    """Versioned evidence for one explicit integration or execution candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: CapabilitySchema = Field(
        default="cayu.capabilities.v1",
        alias="schema",
    )
    subject: CapabilityIdentity
    claims: tuple[CapabilityClaim, ...] = Field(
        default_factory=tuple,
        max_length=MAX_CAPABILITY_CLAIMS,
    )
    unclaimed_reason_code: CapabilityIdentity | None = None

    @model_validator(mode="after")
    def validate_claims(self) -> Self:
        capabilities = [claim.capability for claim in self.claims]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Capability evidence claims must have unique capability names.")
        if self.claims and self.unclaimed_reason_code is not None:
            raise ValueError("Claimed evidence cannot define unclaimed_reason_code.")
        if not self.claims and self.unclaimed_reason_code is None:
            raise ValueError("Evidence without claims requires unclaimed_reason_code.")
        return self

    @classmethod
    def unclaimed(
        cls,
        subject: str,
        *,
        reason_code: str = "capabilities_unclaimed",
    ) -> Self:
        return cls(subject=subject, unclaimed_reason_code=reason_code)

    def claim_for(self, capability: str) -> CapabilityClaim | None:
        return next(
            (claim for claim in self.claims if claim.capability == capability),
            None,
        )

    def state_for(self, capability: str) -> CapabilityState | Literal["unclaimed"]:
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
            "subject": self.subject,
            "claims": claims,
        }
        if self.unclaimed_reason_code is not None:
            metadata["unclaimed_reason_code"] = self.unclaimed_reason_code
        return metadata

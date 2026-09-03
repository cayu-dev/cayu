"""Content-free protected control-plane diagnostics contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS = 64
MAX_SYSTEM_DEPLOYMENT_NAME_CHARS = 128
MAX_SYSTEM_PRICING_METADATA_CHARS = 256

SystemAccessKind = Literal["open", "authenticated"]
SystemDiagnosticTextStatus = Literal["available", "not_provided", "omitted"]
CapabilityUnavailableReason = Literal["not_configured", "unsupported"]
ConfiguredStoreRole = Literal["session", "task", "knowledge", "artifact"]
EvalsReadinessState = Literal["ready", "gated", "unsupported"]
EvalsReadinessReasonCode = Literal[
    "evaluation_promotion_not_configured",
    "terminal_evidence_not_supported",
    "session_lineage_not_supported",
    "eval_store_not_configured",
    "eval_target_not_configured",
    "captured_result_persistence_not_available",
    "scenario_v2_not_available",
]

_GATED_EVALS_READINESS_REASONS = frozenset(
    {
        "evaluation_promotion_not_configured",
        "eval_store_not_configured",
        "eval_target_not_configured",
    }
)
_UNSUPPORTED_EVALS_READINESS_REASONS = frozenset(
    {
        "terminal_evidence_not_supported",
        "session_lineage_not_supported",
        "captured_result_persistence_not_available",
        "scenario_v2_not_available",
    }
)


class _DiagnosticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SystemDeploymentDiagnostics(_DiagnosticsModel):
    """Bounded resolved server identity and effective access posture."""

    name: str | None = Field(default=None, max_length=MAX_SYSTEM_DEPLOYMENT_NAME_CHARS)
    name_status: SystemDiagnosticTextStatus
    api_access: SystemAccessKind
    dashboard_access: SystemAccessKind | None
    dashboard_enabled: StrictBool
    docs_enabled: StrictBool | None

    @model_validator(mode="after")
    def validate_availability(self) -> SystemDeploymentDiagnostics:
        if self.name_status == "available" and self.name is None:
            raise ValueError("Available deployment names require a value.")
        if self.name_status != "available" and self.name is not None:
            raise ValueError("Unavailable deployment names cannot include a value.")
        if self.dashboard_enabled and self.dashboard_access is None:
            raise ValueError("Enabled dashboards require an access posture.")
        if not self.dashboard_enabled and self.dashboard_access is not None:
            raise ValueError("Disabled dashboards cannot include an access posture.")
        return self


class SystemVersionDiagnostics(_DiagnosticsModel):
    cayu: str | None = Field(max_length=128)
    server_contract: str = Field(max_length=32)


class ArtifactStoreDiagnostic(_DiagnosticsModel):
    """Path-safe registration identity and the required ArtifactStore contract."""

    fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$", max_length=71)
    store_contract_operations: tuple[
        Literal["list"],
        Literal["read"],
        Literal["write"],
        Literal["delete"],
    ] = (
        "list",
        "read",
        "write",
        "delete",
    )


class ArtifactStoreDiagnostics(_DiagnosticsModel):
    registrations: tuple[ArtifactStoreDiagnostic, ...] = Field(
        max_length=MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS
    )
    total_count: StrictInt = Field(ge=0)
    truncated: StrictBool

    @model_validator(mode="after")
    def validate_count(self) -> ArtifactStoreDiagnostics:
        if self.total_count < len(self.registrations):
            raise ValueError("Artifact store total cannot be smaller than registrations.")
        if self.truncated is not (self.total_count > len(self.registrations)):
            raise ValueError("Artifact store truncation must match the returned count.")
        return self


class PricingCatalogDiagnostics(_DiagnosticsModel):
    configured: StrictBool
    metadata_status: Literal["available", "not_configured", "omitted"]
    price_book_version: str | None = Field(
        default=None,
        max_length=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )
    generated_at: str | None = Field(
        default=None,
        max_length=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )

    @model_validator(mode="after")
    def validate_metadata(self) -> PricingCatalogDiagnostics:
        metadata_present = self.price_book_version is not None and self.generated_at is not None
        if self.metadata_status == "available" and not metadata_present:
            raise ValueError("Available pricing metadata requires both identity fields.")
        if self.metadata_status != "available" and (
            self.price_book_version is not None or self.generated_at is not None
        ):
            raise ValueError("Unavailable pricing metadata cannot include identity fields.")
        if self.configured and self.metadata_status == "not_configured":
            raise ValueError("Configured pricing cannot be marked not configured.")
        if not self.configured and self.metadata_status != "not_configured":
            raise ValueError("Unconfigured pricing must be marked not configured.")
        return self


class CapabilityOperation(_DiagnosticsModel):
    """Availability of one control-plane read or mutation operation."""

    enabled: StrictBool
    unavailable_reason: CapabilityUnavailableReason | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> CapabilityOperation:
        if self.enabled and self.unavailable_reason is not None:
            raise ValueError("Enabled capability operations cannot have an unavailable reason.")
        if not self.enabled and self.unavailable_reason is None:
            raise ValueError("Disabled capability operations require an unavailable reason.")
        return self


class OptionalSurfaceCapability(_DiagnosticsModel):
    """Configuration and operation availability for one optional surface."""

    configured: StrictBool
    read: CapabilityOperation
    mutate: CapabilityOperation

    @model_validator(mode="after")
    def validate_configuration(self) -> OptionalSurfaceCapability:
        if not self.configured and (self.read.enabled or self.mutate.enabled):
            raise ValueError("Unconfigured surfaces cannot expose enabled operations.")
        return self


class EvalsOperationReadiness(_DiagnosticsModel):
    """Discovery state for one Evals product operation.

    Readiness is presentation metadata, not an authorization grant. Underlying
    routes continue to enforce authentication, mutation policy, and runtime
    preconditions.
    """

    state: EvalsReadinessState
    reason_code: EvalsReadinessReasonCode | None

    @model_validator(mode="after")
    def validate_reason(self) -> EvalsOperationReadiness:
        if self.state == "ready" and self.reason_code is not None:
            raise ValueError("Ready Evals operations cannot have a reason code.")
        if self.state != "ready" and self.reason_code is None:
            raise ValueError("Unavailable Evals operations require a reason code.")
        if self.state == "gated" and self.reason_code not in _GATED_EVALS_READINESS_REASONS:
            raise ValueError("Gated Evals operations require a gated reason code.")
        if (
            self.state == "unsupported"
            and self.reason_code not in _UNSUPPORTED_EVALS_READINESS_REASONS
        ):
            raise ValueError("Unsupported Evals operations require an unsupported reason code.")
        return self


class EvalsReadiness(_DiagnosticsModel):
    """Independent availability of the Evals product workflows."""

    captured_evaluation: EvalsOperationReadiness
    catalog_read: EvalsOperationReadiness
    catalog_write: EvalsOperationReadiness
    captured_result_persistence: EvalsOperationReadiness
    scenario_conversion: EvalsOperationReadiness
    fresh_launch: EvalsOperationReadiness
    cancellation: EvalsOperationReadiness
    comparison: EvalsOperationReadiness
    reports: EvalsOperationReadiness


class ControlPlaneSurfaceCapabilities(_DiagnosticsModel):
    dashboard: OptionalSurfaceCapability
    workflow: OptionalSurfaceCapability | None = None
    tasks: OptionalSurfaceCapability
    reviewed_knowledge: OptionalSurfaceCapability
    artifacts: OptionalSurfaceCapability
    usage: OptionalSurfaceCapability
    pricing: OptionalSurfaceCapability
    evaluation_promotion: OptionalSurfaceCapability
    evals: OptionalSurfaceCapability


class ControlPlaneMutationCapabilities(_DiagnosticsModel):
    session_execution: CapabilityOperation
    session_interruption: CapabilityOperation
    provider_operation_resolution: CapabilityOperation
    pending_action_resolution: CapabilityOperation
    session_annotations: CapabilityOperation
    task_lifecycle: CapabilityOperation
    knowledge_review: CapabilityOperation


class ServerContractActor(_DiagnosticsModel):
    """Bounded actor projection that deliberately excludes arbitrary claims."""

    subject: str = Field(max_length=512)
    tenant: str | None = Field(default=None, max_length=512)


class ControlPlaneCapabilities(_DiagnosticsModel):
    """Server-authoritative discovery metadata for the Cayu control plane.

    This projection is presentation metadata rather than an authorization
    token. Every underlying route continues to enforce its configured access
    dependency and runtime preconditions.
    """

    cayu_version: str | None = Field(max_length=128)
    configured_store_roles: tuple[ConfiguredStoreRole, ...] = Field(max_length=4)
    actor: ServerContractActor | None
    surfaces: ControlPlaneSurfaceCapabilities
    mutations: ControlPlaneMutationCapabilities
    evals_readiness: EvalsReadiness


class SystemDiagnosticsResponse(_DiagnosticsModel):
    """Protected bounded Cayu configuration and registration diagnostics."""

    observed_at: datetime
    deployment: SystemDeploymentDiagnostics
    versions: SystemVersionDiagnostics
    capabilities: ControlPlaneCapabilities
    artifact_stores: ArtifactStoreDiagnostics
    pricing_catalog: PricingCatalogDiagnostics

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        return value.astimezone(UTC)

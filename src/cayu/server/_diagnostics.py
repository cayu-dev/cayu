"""Bounded, probe-free system diagnostics for the protected control plane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.server._capabilities import ControlPlaneCapabilitySnapshot
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS,
    MAX_SYSTEM_DEPLOYMENT_NAME_CHARS,
    MAX_SYSTEM_PRICING_METADATA_CHARS,
    SERVER_CONTRACT_VERSION,
    ArtifactStoreDiagnostic,
    ArtifactStoreDiagnostics,
    PricingCatalogDiagnostics,
    SystemDeploymentDiagnostics,
    SystemDiagnosticsResponse,
    SystemVersionDiagnostics,
)


@dataclass(frozen=True, slots=True)
class SystemDiagnosticsSnapshot:
    """Fixed server wiring captured without retaining secret-bearing configuration."""

    capabilities: ControlPlaneCapabilitySnapshot
    deployment_name: str | None
    deployment_name_status: Literal["available", "not_provided", "omitted"]
    api_access: Literal["open", "authenticated"]
    dashboard_access: Literal["open", "authenticated"] | None
    dashboard_enabled: bool
    docs_enabled: bool | None
    pricing_catalog: PricingCatalogDiagnostics

    def project(
        self,
        actor: AuthContext | None,
        *,
        artifact_store_fingerprints: tuple[str, ...],
        artifact_store_total_count: int,
    ) -> SystemDiagnosticsResponse:
        """Project request identity and the bounded dynamic registration snapshot."""

        if type(artifact_store_total_count) is not int:
            raise TypeError("artifact_store_total_count must be an integer.")
        if artifact_store_total_count < 0:
            raise ValueError("artifact_store_total_count must be non-negative.")
        if len(artifact_store_fingerprints) > MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS:
            raise ValueError("Too many artifact store fingerprints for system diagnostics.")
        if len(artifact_store_fingerprints) > artifact_store_total_count:
            raise ValueError("Artifact store fingerprints cannot exceed the total count.")

        artifacts_configured = artifact_store_total_count > 0
        return SystemDiagnosticsResponse(
            observed_at=datetime.now(UTC),
            deployment=SystemDeploymentDiagnostics(
                name=self.deployment_name,
                name_status=self.deployment_name_status,
                api_access=self.api_access,
                dashboard_access=self.dashboard_access,
                dashboard_enabled=self.dashboard_enabled,
                docs_enabled=self.docs_enabled,
            ),
            versions=SystemVersionDiagnostics(
                cayu=self.capabilities.cayu_version,
                server_contract=SERVER_CONTRACT_VERSION,
            ),
            capabilities=self.capabilities.project(
                actor,
                artifacts_configured=artifacts_configured,
            ),
            artifact_stores=ArtifactStoreDiagnostics(
                registrations=tuple(
                    ArtifactStoreDiagnostic(
                        fingerprint=fingerprint,
                    )
                    for fingerprint in artifact_store_fingerprints
                ),
                total_count=artifact_store_total_count,
                truncated=len(artifact_store_fingerprints) < artifact_store_total_count,
            ),
            pricing_catalog=self.pricing_catalog,
        )


def inspect_system_diagnostics(
    *,
    capabilities: ControlPlaneCapabilitySnapshot,
    deployment_name: str | None,
    api_authenticated: bool,
    dashboard_authenticated: bool | None,
    dashboard_enabled: bool,
    docs_enabled: bool | None,
    pricing_configured: bool,
    pricing_metadata: tuple[str, str] | None,
) -> SystemDiagnosticsSnapshot:
    """Capture bounded system metadata once without probing external dependencies."""

    if not isinstance(capabilities, ControlPlaneCapabilitySnapshot):
        raise TypeError("capabilities must be a ControlPlaneCapabilitySnapshot.")
    for field_name, value in (
        ("api_authenticated", api_authenticated),
        ("dashboard_enabled", dashboard_enabled),
        ("pricing_configured", pricing_configured),
    ):
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be a bool.")
    for field_name, value in (
        ("dashboard_authenticated", dashboard_authenticated),
        ("docs_enabled", docs_enabled),
    ):
        if value is not None and type(value) is not bool:
            raise TypeError(f"{field_name} must be a bool or None.")
    if dashboard_enabled and dashboard_authenticated is None:
        raise ValueError("Enabled dashboards require an explicit access posture.")
    if not dashboard_enabled and dashboard_authenticated is not None:
        raise ValueError("Disabled dashboards cannot report an access posture.")

    bounded_deployment_name, deployment_name_status = _bounded_diagnostic_text(
        deployment_name,
        max_chars=MAX_SYSTEM_DEPLOYMENT_NAME_CHARS,
    )
    pricing_catalog = _pricing_catalog_diagnostics(
        configured=pricing_configured,
        metadata=pricing_metadata,
    )
    if pricing_catalog.configured is not capabilities.dashboard_pricing_configured:
        raise ValueError("Pricing diagnostics must match the capability configuration.")

    return SystemDiagnosticsSnapshot(
        capabilities=capabilities,
        deployment_name=bounded_deployment_name,
        deployment_name_status=deployment_name_status,
        api_access="authenticated" if api_authenticated else "open",
        dashboard_access=(
            None
            if dashboard_authenticated is None
            else "authenticated"
            if dashboard_authenticated
            else "open"
        ),
        dashboard_enabled=dashboard_enabled,
        docs_enabled=docs_enabled,
        pricing_catalog=pricing_catalog,
    )


def dashboard_pricing_metadata(
    runtime_config: Mapping[str, object],
) -> tuple[str, str] | None:
    """Extract only validated catalog identity from resolved dashboard configuration."""

    if not isinstance(runtime_config, Mapping):
        raise TypeError("Resolved dashboard runtime configuration must be a mapping.")
    price_book = runtime_config.get("priceBook")
    if price_book is None:
        return None
    if not isinstance(price_book, Mapping):
        raise RuntimeError("Resolved dashboard priceBook must be an object.")
    price_book = cast("Mapping[str, object]", price_book)
    version = price_book.get("price_book_version")
    generated_at = price_book.get("generated_at")
    if type(version) is not str or type(generated_at) is not str:
        raise RuntimeError("Resolved dashboard priceBook identity must contain strings.")
    return version, generated_at


def _pricing_catalog_diagnostics(
    *,
    configured: bool,
    metadata: tuple[str, str] | None,
) -> PricingCatalogDiagnostics:
    if not configured:
        if metadata is not None:
            raise ValueError("Unconfigured pricing diagnostics cannot include metadata.")
        return PricingCatalogDiagnostics(
            configured=False,
            metadata_status="not_configured",
            price_book_version=None,
            generated_at=None,
        )
    if metadata is None:
        return PricingCatalogDiagnostics(
            configured=True,
            metadata_status="omitted",
            price_book_version=None,
            generated_at=None,
        )
    if type(metadata) is not tuple or len(metadata) != 2:
        raise TypeError("pricing_metadata must be a two-string tuple or None.")
    version, generated_at = metadata
    bounded_version, version_status = _bounded_diagnostic_text(
        version,
        max_chars=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )
    bounded_generated_at, generated_at_status = _bounded_diagnostic_text(
        generated_at,
        max_chars=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )
    metadata_available = version_status == generated_at_status == "available"
    return PricingCatalogDiagnostics(
        configured=True,
        metadata_status="available" if metadata_available else "omitted",
        price_book_version=bounded_version if metadata_available else None,
        generated_at=bounded_generated_at if metadata_available else None,
    )


def _bounded_diagnostic_text(
    value: str | None,
    *,
    max_chars: int,
) -> tuple[str | None, Literal["available", "not_provided", "omitted"]]:
    if value is None:
        return None, "not_provided"
    if type(value) is not str:
        raise TypeError("Diagnostic text values must be strings or None.")
    try:
        value = require_clean_nonblank(value, "diagnostic_value")
        value = require_unicode_scalar_text(value, "diagnostic_value")
    except ValueError:
        return None, "omitted"
    if len(value) > max_chars:
        return None, "omitted"
    return value, "available"

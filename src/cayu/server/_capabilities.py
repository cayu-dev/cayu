"""Bounded control-plane capability discovery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    CapabilityOperation,
    ConfiguredStoreRole,
    ControlPlaneCapabilities,
    ControlPlaneMutationCapabilities,
    ControlPlaneSurfaceCapabilities,
    OptionalSurfaceCapability,
    ServerContractActor,
)


@dataclass(frozen=True, slots=True)
class ControlPlaneCapabilitySnapshot:
    """Immutable server wiring captured when control-plane routes are built."""

    cayu_version: str | None
    dashboard_configured: bool
    tasks_configured: bool
    knowledge_configured: bool
    dashboard_pricing_configured: bool
    session_usage_aggregates_supported: bool

    def project(
        self,
        actor: AuthContext | None,
        *,
        artifacts_configured: bool,
    ) -> ControlPlaneCapabilities:
        """Add request-time actor and registration state without retaining either."""

        if type(artifacts_configured) is not bool:
            raise TypeError("artifacts_configured must be a bool.")
        configured_store_roles: tuple[ConfiguredStoreRole, ...] = tuple(
            role
            for role, configured in (
                ("session", True),
                ("task", self.tasks_configured),
                ("knowledge", self.knowledge_configured),
                ("artifact", artifacts_configured),
            )
            if configured
        )

        return ControlPlaneCapabilities(
            cayu_version=self.cayu_version,
            configured_store_roles=configured_store_roles,
            actor=(
                None
                if actor is None
                else ServerContractActor(subject=actor.subject, tenant=actor.tenant)
            ),
            surfaces=ControlPlaneSurfaceCapabilities(
                dashboard=_optional_surface(
                    self.dashboard_configured,
                    mutation_supported=False,
                ),
                tasks=_optional_surface(
                    self.tasks_configured,
                    mutation_supported=True,
                ),
                reviewed_knowledge=_optional_surface(
                    self.knowledge_configured,
                    mutation_supported=True,
                ),
                artifacts=_optional_surface(
                    artifacts_configured,
                    mutation_supported=False,
                ),
                usage=_optional_surface(
                    True,
                    read_supported=self.session_usage_aggregates_supported,
                    mutation_supported=False,
                ),
                pricing=_optional_surface(
                    self.dashboard_pricing_configured,
                    read_supported=self.session_usage_aggregates_supported,
                    mutation_supported=False,
                ),
            ),
            mutations=ControlPlaneMutationCapabilities(
                session_execution=_operation(True),
                session_interruption=_operation(True),
                pending_action_resolution=_operation(True),
                session_annotations=_operation(True),
                task_lifecycle=_operation(self.tasks_configured),
                knowledge_review=_operation(self.knowledge_configured),
            ),
        )


def inspect_control_plane_capabilities(
    *,
    dashboard_configured: bool,
    tasks_configured: bool,
    knowledge_configured: bool,
    dashboard_pricing_configured: bool,
    session_usage_aggregates_supported: bool,
) -> ControlPlaneCapabilitySnapshot:
    """Capture fixed capability inputs once, without probing external services."""

    for field_name, value in (
        ("dashboard_configured", dashboard_configured),
        ("tasks_configured", tasks_configured),
        ("knowledge_configured", knowledge_configured),
        ("dashboard_pricing_configured", dashboard_pricing_configured),
        ("session_usage_aggregates_supported", session_usage_aggregates_supported),
    ):
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be a bool.")
    if dashboard_pricing_configured and not dashboard_configured:
        raise ValueError("Dashboard pricing cannot be configured when the dashboard is disabled.")

    return ControlPlaneCapabilitySnapshot(
        cayu_version=_cayu_distribution_version(),
        dashboard_configured=dashboard_configured,
        tasks_configured=tasks_configured,
        knowledge_configured=knowledge_configured,
        dashboard_pricing_configured=dashboard_pricing_configured,
        session_usage_aggregates_supported=session_usage_aggregates_supported,
    )


def _cayu_distribution_version() -> str | None:
    try:
        value = version("cayu")
    except PackageNotFoundError:
        return None
    if not value or len(value) > 128:
        return None
    return value


def _operation(
    enabled: bool,
    *,
    unavailable_reason: Literal["not_configured", "unsupported"] = "not_configured",
) -> CapabilityOperation:
    return CapabilityOperation(
        enabled=enabled,
        unavailable_reason=None if enabled else unavailable_reason,
    )


def _optional_surface(
    configured: bool,
    *,
    read_supported: bool = True,
    mutation_supported: bool,
) -> OptionalSurfaceCapability:
    read_enabled = configured and read_supported
    return OptionalSurfaceCapability(
        configured=configured,
        read=_operation(
            read_enabled,
            unavailable_reason="not_configured" if not configured else "unsupported",
        ),
        mutate=(
            _operation(configured)
            if mutation_supported
            else _operation(False, unavailable_reason="unsupported")
        ),
    )

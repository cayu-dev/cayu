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
    EvalsOperationReadiness,
    EvalsReadiness,
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
    session_topology_supported: bool
    evaluation_promotion_configured: bool
    terminal_session_evidence_supported: bool
    session_lineage_supported: bool
    evals_configured: bool
    eval_store_configured: bool
    eval_target_configured: bool
    eval_project_identity_configured: bool
    eval_captured_results_supported: bool

    @property
    def evaluation_promotion_supported(self) -> bool:
        return (
            self.evaluation_promotion_configured
            and self.terminal_session_evidence_supported
            and self.session_lineage_supported
        )

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
                workflow=_optional_surface(
                    True,
                    read_supported=self.session_topology_supported,
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
                evaluation_promotion=_evaluation_promotion_surface(
                    self.evaluation_promotion_configured,
                    supported=self.evaluation_promotion_supported,
                ),
                evals=_optional_surface(
                    self.evals_configured,
                    mutation_supported=True,
                ),
            ),
            mutations=ControlPlaneMutationCapabilities(
                session_execution=_operation(True),
                session_interruption=_operation(True),
                provider_operation_resolution=_operation(True),
                pending_action_resolution=_operation(True),
                session_annotations=_operation(True),
                task_lifecycle=_operation(self.tasks_configured),
                knowledge_review=_operation(self.knowledge_configured),
            ),
            evals_readiness=_evals_readiness(self),
        )


def inspect_control_plane_capabilities(
    *,
    dashboard_configured: bool,
    tasks_configured: bool,
    knowledge_configured: bool,
    dashboard_pricing_configured: bool,
    session_usage_aggregates_supported: bool,
    session_topology_supported: bool,
    evaluation_promotion_configured: bool = False,
    terminal_session_evidence_supported: bool = False,
    session_lineage_supported: bool = False,
    evals_configured: bool = False,
    eval_store_configured: bool | None = None,
    eval_target_configured: bool | None = None,
    eval_project_identity_configured: bool | None = None,
    eval_captured_results_supported: bool = False,
) -> ControlPlaneCapabilitySnapshot:
    """Capture fixed capability inputs once, without probing external services."""

    eval_store_configured = (
        evals_configured if eval_store_configured is None else eval_store_configured
    )
    eval_target_configured = (
        evals_configured if eval_target_configured is None else eval_target_configured
    )
    eval_project_identity_configured = (
        evals_configured
        if eval_project_identity_configured is None
        else eval_project_identity_configured
    )
    for field_name, value in (
        ("dashboard_configured", dashboard_configured),
        ("tasks_configured", tasks_configured),
        ("knowledge_configured", knowledge_configured),
        ("dashboard_pricing_configured", dashboard_pricing_configured),
        ("session_usage_aggregates_supported", session_usage_aggregates_supported),
        ("session_topology_supported", session_topology_supported),
        ("evaluation_promotion_configured", evaluation_promotion_configured),
        ("terminal_session_evidence_supported", terminal_session_evidence_supported),
        ("session_lineage_supported", session_lineage_supported),
        ("evals_configured", evals_configured),
        ("eval_store_configured", eval_store_configured),
        ("eval_target_configured", eval_target_configured),
        ("eval_project_identity_configured", eval_project_identity_configured),
        ("eval_captured_results_supported", eval_captured_results_supported),
    ):
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be a bool.")
    if dashboard_pricing_configured and not dashboard_configured:
        raise ValueError("Dashboard pricing cannot be configured when the dashboard is disabled.")
    if evals_configured and not (
        eval_store_configured and eval_target_configured and eval_project_identity_configured
    ):
        raise ValueError("Configured Evals requires store, target, and project identity evidence.")
    return ControlPlaneCapabilitySnapshot(
        cayu_version=_cayu_distribution_version(),
        dashboard_configured=dashboard_configured,
        tasks_configured=tasks_configured,
        knowledge_configured=knowledge_configured,
        dashboard_pricing_configured=dashboard_pricing_configured,
        session_usage_aggregates_supported=session_usage_aggregates_supported,
        session_topology_supported=session_topology_supported,
        evaluation_promotion_configured=evaluation_promotion_configured,
        terminal_session_evidence_supported=terminal_session_evidence_supported,
        session_lineage_supported=session_lineage_supported,
        evals_configured=evals_configured,
        eval_store_configured=eval_store_configured,
        eval_target_configured=eval_target_configured,
        eval_project_identity_configured=eval_project_identity_configured,
        eval_captured_results_supported=eval_captured_results_supported,
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


def _evaluation_promotion_surface(
    configured: bool,
    *,
    supported: bool,
) -> OptionalSurfaceCapability:
    enabled = configured and supported
    unavailable_reason: Literal["not_configured", "unsupported"] = (
        "not_configured" if not configured else "unsupported"
    )
    return OptionalSurfaceCapability(
        configured=configured,
        read=_operation(enabled, unavailable_reason=unavailable_reason),
        mutate=_operation(enabled, unavailable_reason=unavailable_reason),
    )


def _evals_readiness(snapshot: ControlPlaneCapabilitySnapshot) -> EvalsReadiness:
    # Captured evaluation routes resolve the session's agent through the
    # published eval-target registry.  The older runnable-promotion config has
    # its own capability surface and must not make this workflow look ready.
    if not snapshot.eval_target_configured:
        captured_evaluation = EvalsOperationReadiness(
            state="gated",
            reason_code="eval_target_not_configured",
        )
    elif not snapshot.terminal_session_evidence_supported:
        captured_evaluation = EvalsOperationReadiness(
            state="unsupported",
            reason_code="terminal_evidence_not_supported",
        )
    elif not snapshot.session_lineage_supported:
        captured_evaluation = EvalsOperationReadiness(
            state="unsupported",
            reason_code="session_lineage_not_supported",
        )
    else:
        captured_evaluation = EvalsOperationReadiness(state="ready", reason_code=None)

    store_ready = snapshot.eval_store_configured
    target_ready = snapshot.eval_target_configured
    identity_ready = snapshot.eval_project_identity_configured
    if store_ready and target_ready and identity_ready:
        catalog = EvalsOperationReadiness(state="ready", reason_code=None)
        stored_result_operation = EvalsOperationReadiness(state="ready", reason_code=None)
    else:
        blocking_reason = (
            "eval_store_not_configured" if not store_ready else "eval_target_not_configured"
        )
        catalog = EvalsOperationReadiness(
            state="gated",
            reason_code=blocking_reason,
        )
        stored_result_operation = EvalsOperationReadiness(
            state="gated",
            reason_code=blocking_reason,
        )

    if store_ready and target_ready and identity_ready:
        fresh_launch = EvalsOperationReadiness(state="ready", reason_code=None)
    else:
        fresh_launch = EvalsOperationReadiness(
            state="gated",
            reason_code=(
                "eval_target_not_configured"
                if not target_ready or not identity_ready
                else "eval_store_not_configured"
            ),
        )

    if not store_ready or not target_ready or not identity_ready:
        captured_result_persistence = EvalsOperationReadiness(
            state="gated",
            reason_code=(
                "eval_store_not_configured" if not store_ready else "eval_target_not_configured"
            ),
        )
    elif snapshot.eval_captured_results_supported:
        captured_result_persistence = EvalsOperationReadiness(
            state="ready",
            reason_code=None,
        )
    else:
        captured_result_persistence = EvalsOperationReadiness(
            state="unsupported",
            reason_code="captured_result_persistence_not_available",
        )

    return EvalsReadiness(
        captured_evaluation=captured_evaluation,
        catalog_read=catalog,
        catalog_write=catalog.model_copy(),
        captured_result_persistence=captured_result_persistence,
        scenario_conversion=captured_evaluation.model_copy(),
        fresh_launch=fresh_launch,
        cancellation=stored_result_operation,
        comparison=stored_result_operation.model_copy(),
        reports=stored_result_operation.model_copy(),
    )

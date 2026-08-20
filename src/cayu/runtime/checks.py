from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.agents import AgentAuthoringState
from cayu.runtime.manifest import AppManifest, FrozenJsonObject
from cayu.runtime.service_manifest import (
    PublicServiceManifest,
    RuntimeStoreDurability,
    ServiceMode,
)

CHECK_REPORT_SCHEMA_VERSION = "2"
AVAILABLE_CHECK_TAGS = frozenset({"authoring", "configuration", "deploy", "providers", "security"})
BUILTIN_DIAGNOSTIC_CODES = (
    "AGENT_GENERATED_TRACER_BULLET_UNFINISHED",
    "AGENT_PROVIDER_AMBIGUOUS",
    "AGENT_PROVIDER_NOT_FOUND",
    "AGENT_WORKFLOW_COMMAND_POLICY_NOT_REGISTERED",
    "AGENT_WORKFLOW_RUNNER_NOT_REGISTERED",
    "AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
    "AGENT_WORKFLOW_WORKSPACE_NOT_REGISTERED",
    "APP_NO_AGENTS",
    "EVALS_PROJECT_IDENTITY_NOT_CONFIGURED",
    "EVALS_PROJECT_STORE_NOT_CONFIGURED",
    "EVALS_SERVICE_FACTORY_CONTEXT_MIGRATION_REQUIRED",
    "EXTERNAL_TOOL_COVERAGE_UNKNOWN",
    "EXTERNAL_TOOL_UNGUARDED",
    "PUBLIC_SERVICE_DEVELOPMENT_MODE",
    "PUBLIC_SERVICE_IDENTITY_STORE_NOT_DURABLE",
    "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
    "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
    "PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE",
    "PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE",
    "PUBLIC_SERVICE_TASK_STORE_REQUIRED",
    "TOOL_INPUT_SCHEMA_UNCONSTRAINED",
)
_WORKSPACE_TOOL_NAMES = frozenset(
    {
        "delete_file",
        "edit_file",
        "list_files",
        "read_file",
        "search_text",
        "write_file",
    }
)
_RUNNER_TOOL_NAMES = frozenset({"exec_command", "git_changes"})


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ProjectDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: DiagnosticSeverity
    subject: str
    path: str
    message: str
    hint: str | None = None
    tags: tuple[str, ...] = ()
    parameters: FrozenJsonObject = Field(default_factory=lambda: MappingProxyType({}))
    documentation_anchor: str | None = None
    verification_command: str


class ServiceCheckEvidence(BaseModel):
    """Bounded claims for the maintained public-service deployment path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_plane_access: Literal["not_evaluated", "verified_authenticated", "verified_unsafe"]
    service_contract: Literal["not_declared", "verified_maintained"]
    application_security: Literal["not_evaluated", "generated_suite_required"]
    configuration: Literal["not_applicable", "supported", "unsupported"]
    host_owned_behavior: Literal["unverified_outside_contract"]
    security_verification_command: Literal["pytest -q tests/test_public_service_security.py"]


@dataclass(frozen=True, slots=True)
class ProjectControlPlaneCheckEvidence:
    """Safe project-bootstrap facts supplied by the CLI without runtime objects."""

    project_identity_configured: bool
    eval_store_configured: bool
    service_context: Literal["not_applicable", "attached", "migration_required"]

    def __post_init__(self) -> None:
        if type(self.project_identity_configured) is not bool:
            raise TypeError("project_identity_configured must be a bool.")
        if type(self.eval_store_configured) is not bool:
            raise TypeError("eval_store_configured must be a bool.")


class ProjectCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2"] = CHECK_REPORT_SCHEMA_VERSION
    manifest_fingerprint: str
    diagnostics: tuple[ProjectDiagnostic, ...]
    service_evidence: ServiceCheckEvidence


def check_manifest(
    manifest: AppManifest,
    *,
    service_manifest: PublicServiceManifest | None = None,
    project_control_plane: ProjectControlPlaneCheckEvidence | None = None,
    tags: frozenset[str] | None = None,
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Run deterministic, side-effect-free checks over an application manifest."""

    if not isinstance(manifest, AppManifest):
        raise TypeError("check_manifest requires an AppManifest.")
    if (
        project_control_plane is not None
        and type(project_control_plane) is not ProjectControlPlaneCheckEvidence
    ):
        raise TypeError("project_control_plane must be ProjectControlPlaneCheckEvidence or None.")
    requested_tags = frozenset() if tags is None else frozenset(tags)
    unknown = requested_tags - AVAILABLE_CHECK_TAGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown check tags: {names}.")

    diagnostics: list[ProjectDiagnostic] = []
    if service_manifest is not None:
        diagnostics.extend(_check_public_service(service_manifest))
    if not manifest.agents:
        diagnostics.append(
            ProjectDiagnostic(
                code="APP_NO_AGENTS",
                severity=DiagnosticSeverity.ERROR,
                subject="application",
                path="agents",
                message="The application does not register an agent.",
                hint="Register at least one AgentSpec with CayuApp.register_agent().",
                tags=("authoring", "configuration"),
                documentation_anchor="cayu guide diagnostics#app-no-agents",
                verification_command="cayu check --json",
            )
        )

    for agent in manifest.agents:
        if agent.authoring_state is AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET:
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_GENERATED_TRACER_BULLET_UNFINISHED",
                    severity=DiagnosticSeverity.WARNING,
                    subject=f"agent:{agent.name}",
                    path=f"agents.{agent.name}.authoring_state",
                    message=(
                        f"Agent '{agent.name}' is still marked as an unfinished generated "
                        "tracer bullet; generated placeholder behavior is not domain completion."
                    ),
                    hint=(
                        "Replace the domain system prompt, tool schema and body, runtime test "
                        "inputs and assertions, and trajectory eval behavior and assertions; "
                        "then remove the authoring_state marker and its unused import."
                    ),
                    tags=("authoring", "deploy"),
                    parameters={
                        "agent": agent.name,
                        "authoring_state": str(agent.authoring_state),
                    },
                    documentation_anchor=(
                        "cayu guide diagnostics#agent-generated-tracer-bullet-unfinished"
                    ),
                    verification_command=(
                        "cayu inspect --json && cayu check --fail-on warning --json"
                    ),
                )
            )
        if agent.provider_resolution == "missing":
            provider_name = agent.configured_provider or "<default>"
            hint = (
                f"Register provider '{provider_name}' or change {agent.name}.provider_name."
                if agent.configured_provider is not None
                else "Register a default provider or configure this agent's provider_name."
            )
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_PROVIDER_NOT_FOUND",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"agent:{agent.name}",
                    path=f"agents.{agent.name}.configured_provider",
                    message=(
                        f"Agent '{agent.name}' cannot resolve model provider '{provider_name}'."
                    ),
                    hint=hint,
                    tags=("configuration", "providers"),
                    parameters={"agent": agent.name, "provider": provider_name},
                    documentation_anchor="cayu guide diagnostics#agent-provider-not-found",
                    verification_command="cayu inspect --json && cayu check --json",
                )
            )
        elif agent.provider_resolution == "ambiguous":
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_PROVIDER_AMBIGUOUS",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"agent:{agent.name}",
                    path=f"agents.{agent.name}.model",
                    message=(
                        f"Agent '{agent.name}' model '{agent.model}' matches multiple providers: "
                        f"{', '.join(agent.provider_candidates)}."
                    ),
                    hint=f"Set {agent.name}.provider_name explicitly or make model patterns disjoint.",
                    tags=("configuration", "providers"),
                    parameters={
                        "agent": agent.name,
                        "model": agent.model,
                        "providers": list(agent.provider_candidates),
                    },
                    documentation_anchor="cayu guide diagnostics#agent-provider-ambiguous",
                    verification_command="cayu inspect --json && cayu check --json",
                )
            )

        registered_tool_names = tuple(tool.name for tool in agent.tools)
        registered_tool_set = frozenset(registered_tool_names)
        for workflow_tool_name in agent.workflow_tool_names:
            if workflow_tool_name in registered_tool_set:
                continue
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"agent:{agent.name}",
                    path=(f"agents.{agent.name}.workflow_tool_names.{workflow_tool_name}"),
                    message=(
                        f"Agent '{agent.name}' declares workflow tool '{workflow_tool_name}', "
                        "but that tool is not registered for the agent."
                    ),
                    hint=(
                        "Use the exact registered tool name in the machine-checkable workflow "
                        "contract, or register the intended tool for this agent."
                    ),
                    tags=("authoring", "configuration", "deploy"),
                    parameters={
                        "agent": agent.name,
                        "workflow_tool": workflow_tool_name,
                        "registered_tools": list(registered_tool_names),
                    },
                    documentation_anchor=(
                        "cayu guide diagnostics#agent-workflow-tool-not-registered"
                    ),
                    verification_command="cayu inspect --json && cayu check --json",
                )
            )

        registered_workflow_tools = registered_tool_set.intersection(agent.workflow_tool_names)
        workspace_workflow_tools = sorted(
            registered_workflow_tools.intersection(_WORKSPACE_TOOL_NAMES)
        )
        if workspace_workflow_tools and not _has_workspace_provider(manifest):
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_WORKFLOW_WORKSPACE_NOT_REGISTERED",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"agent:{agent.name}",
                    path=(f"agents.{agent.name}.workflow_tool_names.{workspace_workflow_tools[0]}"),
                    message=(
                        f"Agent '{agent.name}' requires workspace-backed workflow tools, "
                        "but no static workspace or environment factory is registered."
                    ),
                    hint=(
                        "Register an Environment with a Workspace, or register an "
                        "EnvironmentFactory that creates one for the session."
                    ),
                    tags=("authoring", "configuration", "deploy"),
                    parameters={
                        "agent": agent.name,
                        "workflow_tools": workspace_workflow_tools,
                    },
                    documentation_anchor=(
                        "cayu guide diagnostics#agent-workflow-workspace-not-registered"
                    ),
                    verification_command="cayu inspect --json && cayu check --json",
                )
            )

        runner_workflow_tools = sorted(registered_workflow_tools.intersection(_RUNNER_TOOL_NAMES))
        if runner_workflow_tools and not _has_runner_provider(manifest):
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_WORKFLOW_RUNNER_NOT_REGISTERED",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"agent:{agent.name}",
                    path=(f"agents.{agent.name}.workflow_tool_names.{runner_workflow_tools[0]}"),
                    message=(
                        f"Agent '{agent.name}' requires runner-backed workflow tools, "
                        "but no static runner or environment factory is registered."
                    ),
                    hint=(
                        "Register an Environment with a Runner, or register an "
                        "EnvironmentFactory that creates one for the session."
                    ),
                    tags=("authoring", "configuration", "deploy"),
                    parameters={
                        "agent": agent.name,
                        "workflow_tools": runner_workflow_tools,
                    },
                    documentation_anchor=(
                        "cayu guide diagnostics#agent-workflow-runner-not-registered"
                    ),
                    verification_command="cayu inspect --json && cayu check --json",
                )
            )

        exec_command = next(
            (
                tool
                for tool in agent.tools
                if tool.name == "exec_command" and tool.name in registered_workflow_tools
            ),
            None,
        )
        if exec_command is not None and exec_command.command_policy is None:
            diagnostics.append(
                ProjectDiagnostic(
                    code="AGENT_WORKFLOW_COMMAND_POLICY_NOT_REGISTERED",
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"tool:{agent.name}/exec_command",
                    path=f"agents.{agent.name}.tools.exec_command.command_policy",
                    message=(
                        f"Agent '{agent.name}' requires exec_command in its workflow, "
                        "but the tool has no enforcing CommandPolicy."
                    ),
                    hint=(
                        "Attach a deny-by-default CommandPolicy such as "
                        "ProcessCommandPolicy or GitCommandPolicy."
                    ),
                    tags=("authoring", "deploy", "security"),
                    parameters={"agent": agent.name, "tool": "exec_command"},
                    documentation_anchor=(
                        "cayu guide diagnostics#agent-workflow-command-policy-not-registered"
                    ),
                    verification_command=("cayu inspect --json && cayu check --deploy --json"),
                )
            )

        for tool in agent.tools:
            if not tool.input_schema:
                diagnostics.append(
                    ProjectDiagnostic(
                        code="TOOL_INPUT_SCHEMA_UNCONSTRAINED",
                        severity=DiagnosticSeverity.WARNING,
                        subject=f"tool:{agent.name}/{tool.name}",
                        path=f"agents.{agent.name}.tools.{tool.name}.input_schema",
                        message=(
                            f"Tool '{tool.name}' uses an empty object, which is valid JSON "
                            "Schema but accepts every JSON value."
                        ),
                        hint=(
                            "Declare the tool's expected object properties, required fields, "
                            "and additionalProperties behavior in ToolSpec.input_schema or "
                            "the Tool.schema property."
                        ),
                        tags=("authoring", "deploy"),
                        parameters={"agent": agent.name, "tool": tool.name},
                        documentation_anchor=(
                            "cayu guide diagnostics#tool-input-schema-unconstrained"
                        ),
                        verification_command=(
                            "cayu inspect --json && cayu check --fail-on warning --json"
                        ),
                    )
                )
            if tool.effect != "external" or tool.policy_coverage in {
                "conditional",
                "denied",
                "approval_required",
            }:
                continue
            coverage_unknown = tool.policy_coverage == "unknown"
            diagnostics.append(
                ProjectDiagnostic(
                    code=(
                        "EXTERNAL_TOOL_COVERAGE_UNKNOWN"
                        if coverage_unknown
                        else "EXTERNAL_TOOL_UNGUARDED"
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    subject=f"tool:{agent.name}/{tool.name}",
                    path=f"agents.{agent.name}.tools.{tool.name}.policy_coverage",
                    message=(
                        f"External-effect tool '{tool.name}' uses {agent.tool_policy}, whose "
                        "effective coverage cannot be verified statically."
                        if coverage_unknown
                        else f"External-effect tool '{tool.name}' has effective policy coverage "
                        f"'{tool.policy_coverage}' under {agent.tool_policy}."
                    ),
                    hint=(
                        "Register a statically describable policy with enforcing coverage for "
                        "this tool; custom policy behavior remains fail-closed."
                        if coverage_unknown
                        else "Register an enforcing ToolPolicy, such as "
                        "AlwaysRequireApprovalToolPolicy scoped to this tool."
                    ),
                    tags=("deploy", "security"),
                    parameters={
                        "agent": agent.name,
                        "tool": tool.name,
                        "policy": agent.tool_policy,
                        "coverage": tool.policy_coverage,
                    },
                    documentation_anchor=(
                        "cayu guide diagnostics#external-tool-coverage-unknown"
                        if coverage_unknown
                        else "cayu guide diagnostics#external-tool-unguarded"
                    ),
                    verification_command="cayu inspect --json && cayu check --deploy --json",
                )
            )

    if project_control_plane is not None:
        diagnostics.extend(_check_project_control_plane(project_control_plane))

    selected = [
        diagnostic
        for diagnostic in diagnostics
        if (not requested_tags or requested_tags.intersection(diagnostic.tags))
        and (not deploy_only or "deploy" in diagnostic.tags)
    ]
    unregistered_codes = {item.code for item in diagnostics} - set(BUILTIN_DIAGNOSTIC_CODES)
    if unregistered_codes:
        names = ", ".join(sorted(unregistered_codes))
        raise RuntimeError(f"Built-in checks emitted unregistered diagnostic codes: {names}.")
    selected.sort(key=lambda item: (item.code, item.path, item.subject))
    return ProjectCheckReport(
        manifest_fingerprint=manifest.fingerprint,
        diagnostics=tuple(selected),
        service_evidence=_service_evidence(service_manifest, diagnostics),
    )


def _check_project_control_plane(
    evidence: ProjectControlPlaneCheckEvidence,
) -> list[ProjectDiagnostic]:
    diagnostics: list[ProjectDiagnostic] = []
    if not evidence.project_identity_configured:
        diagnostics.append(
            ProjectDiagnostic(
                code="EVALS_PROJECT_IDENTITY_NOT_CONFIGURED",
                severity=DiagnosticSeverity.WARNING,
                subject="evals",
                path="project.name",
                message=(
                    "Automatic Evals target identity is unavailable because [project].name "
                    "is not configured."
                ),
                hint="Add a valid name under [project] in pyproject.toml.",
                tags=("configuration",),
                documentation_anchor=(
                    "cayu guide diagnostics#evals-project-identity-not-configured"
                ),
                verification_command="cayu check --json",
            )
        )
    if not evidence.eval_store_configured:
        diagnostics.append(
            ProjectDiagnostic(
                code="EVALS_PROJECT_STORE_NOT_CONFIGURED",
                severity=DiagnosticSeverity.WARNING,
                subject="evals",
                path="tool.cayu.session_store",
                message=("Production Evals persistence has no declared durable project store."),
                hint=(
                    "Configure [tool.cayu.session_store] or CAYU_DATABASE_URL; trusted local "
                    "development may use data/cayu.db automatically."
                ),
                tags=("configuration",),
                documentation_anchor=("cayu guide diagnostics#evals-project-store-not-configured"),
                verification_command="cayu check --json",
            )
        )
    if evidence.service_context == "migration_required":
        diagnostics.append(
            ProjectDiagnostic(
                code="EVALS_SERVICE_FACTORY_CONTEXT_MIGRATION_REQUIRED",
                severity=DiagnosticSeverity.WARNING,
                subject="service",
                path="service_factory.project_context",
                message=(
                    "The maintained service factory does not carry Cayu's framework-owned "
                    "project context into create_agent_service()."
                ),
                hint=(
                    "Run `cayu generate service-context`, review the generated edit, and "
                    "rerun `cayu check --json`."
                ),
                tags=("configuration",),
                documentation_anchor=(
                    "cayu guide diagnostics#evals-service-factory-context-migration-required"
                ),
                verification_command="cayu check --json",
            )
        )
    return diagnostics


def _check_public_service(manifest: PublicServiceManifest) -> list[ProjectDiagnostic]:
    if not isinstance(manifest, PublicServiceManifest):
        raise TypeError("service_manifest must be a PublicServiceManifest.")
    diagnostics: list[ProjectDiagnostic] = []
    verification = "cayu check --deploy --fail-on warning --json"
    if manifest.mode is not ServiceMode.PRODUCTION:
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_DEVELOPMENT_MODE",
                severity=DiagnosticSeverity.WARNING,
                subject="public_service",
                path="service.mode",
                message="The maintained public service is assembled in development mode.",
                hint="Build and serve the service with mode='production'.",
                tags=("deploy", "security"),
                documentation_anchor="cayu guide diagnostics#public-service-development-mode",
                verification_command=verification,
            )
        )
    if manifest.product_access != "authenticated":
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.product_access",
                message="The product API does not use configured production authentication.",
                hint="Configure AuthenticatedProductAccess with trusted tenant resolution.",
                tags=("deploy", "security"),
                parameters={"configured": manifest.product_access},
                documentation_anchor="cayu guide diagnostics#public-service-product-access-unsafe",
                verification_command=verification,
            )
        )
    if manifest.operator_access != "authenticated":
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.operator_access",
                message="The Cayu operator control plane is not authenticated.",
                hint="Configure AuthenticatedAccess for the separate operator policy.",
                tags=("deploy", "security"),
                parameters={"configured": manifest.operator_access},
                documentation_anchor="cayu guide diagnostics#public-service-operator-access-unsafe",
                verification_command=verification,
            )
        )
    if manifest.identity_store != "durable":
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_IDENTITY_STORE_NOT_DURABLE",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.identity_store",
                message="The product identity mapping uses development-only state.",
                hint="Use durable application-owned tenant-qualified product storage.",
                tags=("deploy", "security"),
                parameters={"configured": manifest.identity_store.value},
                documentation_anchor=(
                    "cayu guide diagnostics#public-service-identity-store-not-durable"
                ),
                verification_command=verification,
            )
        )
    if manifest.runtime_session_store is not RuntimeStoreDurability.DURABLE:
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.runtime_session_store",
                message="The public agent service uses a non-durable Cayu session store.",
                hint="Construct CayuApp with a durable SessionStore for public service work.",
                tags=("deploy", "security"),
                parameters={"configured": manifest.runtime_session_store.value},
                documentation_anchor=(
                    "cayu guide diagnostics#public-service-session-store-not-durable"
                ),
                verification_command=verification,
            )
        )
    if manifest.runtime_task_store == "missing":
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_TASK_STORE_REQUIRED",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.runtime_task_store",
                message="The public agent service has no Cayu task store.",
                hint="Construct CayuApp with a durable TaskStore for public service work.",
                tags=("deploy", "security"),
                documentation_anchor="cayu guide diagnostics#public-service-task-store-required",
                verification_command=verification,
            )
        )
    elif manifest.runtime_task_store is not RuntimeStoreDurability.DURABLE:
        diagnostics.append(
            ProjectDiagnostic(
                code="PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE",
                severity=DiagnosticSeverity.ERROR,
                subject="public_service",
                path="service.runtime_task_store",
                message="The public agent service uses a non-durable Cayu task store.",
                hint="Construct CayuApp with a durable TaskStore for public service work.",
                tags=("deploy", "security"),
                parameters={"configured": manifest.runtime_task_store.value},
                documentation_anchor=(
                    "cayu guide diagnostics#public-service-task-store-not-durable"
                ),
                verification_command=verification,
            )
        )
    return diagnostics


def check_public_service_deployment(
    manifest: PublicServiceManifest,
) -> tuple[ProjectDiagnostic, ...]:
    """Return stable fail-closed findings for a maintained service manifest."""

    return tuple(_check_public_service(manifest))


def _service_evidence(
    manifest: PublicServiceManifest | None,
    diagnostics: list[ProjectDiagnostic],
) -> ServiceCheckEvidence:
    if manifest is None:
        return ServiceCheckEvidence(
            control_plane_access="not_evaluated",
            service_contract="not_declared",
            application_security="not_evaluated",
            configuration="not_applicable",
            host_owned_behavior="unverified_outside_contract",
            security_verification_command="pytest -q tests/test_public_service_security.py",
        )
    unsafe_codes = {
        "PUBLIC_SERVICE_DEVELOPMENT_MODE",
        "PUBLIC_SERVICE_IDENTITY_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_TASK_STORE_REQUIRED",
    }
    configured = not any(item.code in unsafe_codes for item in diagnostics)
    return ServiceCheckEvidence(
        control_plane_access=(
            "verified_authenticated"
            if manifest.operator_access == "authenticated"
            else "verified_unsafe"
        ),
        service_contract="verified_maintained",
        application_security="generated_suite_required",
        configuration="supported" if configured else "unsupported",
        host_owned_behavior="unverified_outside_contract",
        security_verification_command="pytest -q tests/test_public_service_security.py",
    )


def _has_workspace_provider(manifest: AppManifest) -> bool:
    return any(
        environment.workspace is not None or environment.factory_backed
        for environment in manifest.environments
    )


def _has_runner_provider(manifest: AppManifest) -> bool:
    return any(
        environment.runner is not None or environment.factory_backed
        for environment in manifest.environments
    )


def severity_at_least(value: DiagnosticSeverity, threshold: DiagnosticSeverity) -> bool:
    rank = {
        DiagnosticSeverity.INFO: 0,
        DiagnosticSeverity.WARNING: 1,
        DiagnosticSeverity.ERROR: 2,
    }
    return rank[value] >= rank[threshold]

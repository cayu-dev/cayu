from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.agents import AgentAuthoringState
from cayu.runtime.manifest import AppManifest, FrozenJsonObject

CHECK_REPORT_SCHEMA_VERSION = "1"
AVAILABLE_CHECK_TAGS = frozenset({"authoring", "configuration", "deploy", "providers", "security"})
BUILTIN_DIAGNOSTIC_CODES = (
    "AGENT_GENERATED_TRACER_BULLET_UNFINISHED",
    "AGENT_PROVIDER_AMBIGUOUS",
    "AGENT_PROVIDER_NOT_FOUND",
    "AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
    "APP_NO_AGENTS",
    "EXTERNAL_TOOL_COVERAGE_UNKNOWN",
    "EXTERNAL_TOOL_UNGUARDED",
    "TOOL_INPUT_SCHEMA_UNCONSTRAINED",
)


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


class ProjectCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CHECK_REPORT_SCHEMA_VERSION
    manifest_fingerprint: str
    diagnostics: tuple[ProjectDiagnostic, ...]


def check_manifest(
    manifest: AppManifest,
    *,
    tags: frozenset[str] | None = None,
    deploy_only: bool = False,
) -> ProjectCheckReport:
    """Run deterministic, side-effect-free checks over an application manifest."""

    if not isinstance(manifest, AppManifest):
        raise TypeError("check_manifest requires an AppManifest.")
    requested_tags = frozenset() if tags is None else frozenset(tags)
    unknown = requested_tags - AVAILABLE_CHECK_TAGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown check tags: {names}.")

    diagnostics: list[ProjectDiagnostic] = []
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
    )


def severity_at_least(value: DiagnosticSeverity, threshold: DiagnosticSeverity) -> bool:
    rank = {
        DiagnosticSeverity.INFO: 0,
        DiagnosticSeverity.WARNING: 1,
        DiagnosticSeverity.ERROR: 2,
    }
    return rank[value] >= rank[threshold]

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu import (
    AgentAuthoringState,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    ParameterConstrainedToolPolicy,
    ProjectDiagnostic,
    RequiredFieldRule,
    ScriptedModelProvider,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    check_manifest,
)
from cayu.runtime import BUILTIN_DIAGNOSTIC_CODES


class _ExternalTool(Tool):
    spec = ToolSpec(
        name="send",
        effect=ToolEffect.EXTERNAL,
        input_schema={"type": "object", "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="sent")


class _ConfiguredTool(Tool):
    def __init__(self, name: str, *, effect: ToolEffect, content: str) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                effect=effect,
                input_schema={"type": "object", "additionalProperties": False},
            )
        )
        self._content = content

    async def run(self, ctx, args):
        return ToolResult(content=self._content)


class _PermissiveApprovalPolicy(AlwaysRequireApprovalToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_builtin_diagnostic_codes_are_unique_and_compatibility_pinned() -> None:
    assert BUILTIN_DIAGNOSTIC_CODES == (
        "AGENT_GENERATED_TRACER_BULLET_UNFINISHED",
        "AGENT_PROVIDER_AMBIGUOUS",
        "AGENT_PROVIDER_NOT_FOUND",
        "AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
        "APP_NO_AGENTS",
        "EXTERNAL_TOOL_COVERAGE_UNKNOWN",
        "EXTERNAL_TOOL_UNGUARDED",
    )
    assert len(BUILTIN_DIAGNOSTIC_CODES) == len(set(BUILTIN_DIAGNOSTIC_CODES))


def test_workflow_tool_references_must_match_that_agents_registered_tools() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="test-model",
            workflow_tool_names=("read_file", "write_file"),
        ),
        tools=[
            _ConfiguredTool("read_source", effect=ToolEffect.NONE, content="source"),
            _ConfiguredTool("edit_file", effect=ToolEffect.IDEMPOTENT, content="edited"),
        ],
    )
    app.register_agent(
        AgentSpec(name="other", model="test-model"),
        tools=[
            _ConfiguredTool("read_file", effect=ToolEffect.NONE, content="source"),
            _ConfiguredTool("write_file", effect=ToolEffect.IDEMPOTENT, content="edited"),
        ],
    )

    manifest = app.describe()
    report = check_manifest(manifest)
    deploy_report = check_manifest(manifest, deploy_only=True)

    reviewer = next(agent for agent in manifest.agents if agent.name == "reviewer")
    assert reviewer.workflow_tool_names == ("read_file", "write_file")
    assert [item.code for item in report.diagnostics] == [
        "AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
        "AGENT_WORKFLOW_TOOL_NOT_REGISTERED",
    ]
    assert [item.parameters["workflow_tool"] for item in report.diagnostics] == [
        "read_file",
        "write_file",
    ]
    assert report.diagnostics[0].parameters["registered_tools"] == (
        "edit_file",
        "read_source",
    )
    assert report.diagnostics[0].verification_command == (
        "cayu inspect --json && cayu check --json"
    )
    assert "exact registered tool name" in report.diagnostics[0].hint
    assert deploy_report.diagnostics == report.diagnostics


def test_generated_tracer_bullet_warning_is_explicit_deterministic_and_deploy_visible() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name="writer",
            model="test-model",
            system_prompt="A completed prompt may still say sample, echo, or tracer bullet.",
        )
    )
    for name in ("reviewer", "analyst"):
        app.register_agent(
            AgentSpec(
                name=name,
                model="test-model",
                authoring_state=(AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET),
            )
        )

    full = check_manifest(app.describe())
    authoring = check_manifest(app.describe(), tags=frozenset({"authoring"}))
    deploy = check_manifest(app.describe(), deploy_only=True)
    providers = check_manifest(app.describe(), tags=frozenset({"providers"}))

    assert [item.subject for item in full.diagnostics] == [
        "agent:analyst",
        "agent:reviewer",
    ]
    assert authoring.diagnostics == full.diagnostics
    assert deploy.diagnostics == full.diagnostics
    assert providers.diagnostics == ()
    finding = full.diagnostics[0]
    assert finding.code == "AGENT_GENERATED_TRACER_BULLET_UNFINISHED"
    assert finding.severity == "warning"
    assert finding.path == "agents.analyst.authoring_state"
    assert finding.parameters == {
        "agent": "analyst",
        "authoring_state": "unfinished_generated_tracer_bullet",
    }
    assert finding.tags == ("authoring", "deploy")
    assert "unfinished generated tracer bullet" in finding.message
    assert "remove the authoring_state marker" in finding.hint
    assert finding.documentation_anchor == (
        "cayu guide diagnostics#agent-generated-tracer-bullet-unfinished"
    )
    assert finding.verification_command == (
        "cayu inspect --json && cayu check --fail-on warning --json"
    )


def test_generated_tracer_bullet_verification_command_does_not_interpolate_agent_name() -> None:
    agent_name = "reviewer; echo unsafe"
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name=agent_name,
            model="test-model",
            authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,
        )
    )

    finding = check_manifest(app.describe()).diagnostics[0]

    assert finding.code == "AGENT_GENERATED_TRACER_BULLET_UNFINISHED"
    assert finding.parameters["agent"] == agent_name
    assert agent_name not in finding.verification_command
    assert finding.verification_command == (
        "cayu inspect --json && cayu check --fail-on warning --json"
    )


def test_check_tags_and_deploy_selection_are_orthogonal() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([], name="one"),
        model_patterns=("shared-*",),
    )
    app.register_provider(
        ScriptedModelProvider([], name="two"),
        model_patterns=("shared-*",),
    )
    app.register_agent(AgentSpec(name="agent", model="shared-model"))
    manifest = app.describe()

    full = check_manifest(manifest)
    providers = check_manifest(manifest, tags=frozenset({"providers"}))
    deploy = check_manifest(manifest, deploy_only=True)

    assert [item.code for item in full.diagnostics] == ["AGENT_PROVIDER_AMBIGUOUS"]
    assert providers.diagnostics == full.diagnostics
    assert deploy.diagnostics == ()


def test_every_builtin_diagnostic_has_a_seeded_misconfiguration() -> None:
    empty_codes = {
        item.code for item in check_manifest(CayuApp(enable_logging=False).describe()).diagnostics
    }

    missing = CayuApp(enable_logging=False)
    missing.register_agent(AgentSpec(name="missing", model="model", provider_name="absent"))
    missing_codes = {item.code for item in check_manifest(missing.describe()).diagnostics}

    ambiguous = CayuApp(enable_logging=False)
    for name in ("one", "two"):
        ambiguous.register_provider(
            ScriptedModelProvider([], name=name),
            model_patterns=("shared-*",),
        )
    ambiguous.register_agent(AgentSpec(name="ambiguous", model="shared-model"))
    ambiguous_codes = {item.code for item in check_manifest(ambiguous.describe()).diagnostics}

    unsafe = CayuApp(enable_logging=False)
    unsafe.register_agent(AgentSpec(name="unsafe", model="model"), tools=[_ExternalTool()])
    unsafe_codes = {item.code for item in check_manifest(unsafe.describe()).diagnostics}

    unknown = CayuApp(enable_logging=False)
    unknown.register_agent(
        AgentSpec(name="unknown", model="model"),
        tools=[_ExternalTool()],
        tool_policy=_PermissiveApprovalPolicy(tools=["send"]),
    )
    unknown_codes = {item.code for item in check_manifest(unknown.describe()).diagnostics}

    misaligned = CayuApp(enable_logging=False)
    misaligned.register_agent(
        AgentSpec(
            name="misaligned",
            model="model",
            workflow_tool_names=("missing_tool",),
        )
    )
    misaligned_codes = {item.code for item in check_manifest(misaligned.describe()).diagnostics}

    unfinished = CayuApp(enable_logging=False)
    unfinished.register_agent(
        AgentSpec(
            name="unfinished",
            model="model",
            authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,
        )
    )
    unfinished_codes = {item.code for item in check_manifest(unfinished.describe()).diagnostics}

    assert (
        empty_codes
        | missing_codes
        | ambiguous_codes
        | unsafe_codes
        | unknown_codes
        | misaligned_codes
        | unfinished_codes
        == set(BUILTIN_DIAGNOSTIC_CODES)
    )


def test_external_tool_requires_effective_policy_coverage_for_that_tool() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="sender", model="model"),
        tools=[_ExternalTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["other"]),
    )

    report = check_manifest(app.describe())

    assert [item.code for item in report.diagnostics] == [
        "AGENT_PROVIDER_NOT_FOUND",
        "EXTERNAL_TOOL_UNGUARDED",
    ]
    finding = report.diagnostics[1]
    assert finding.parameters["coverage"] == "allowed"
    assert finding.verification_command == "cayu inspect --json && cayu check --deploy --json"

    covered = CayuApp(enable_logging=False)
    covered.register_agent(
        AgentSpec(name="sender", model="model"),
        tools=[_ExternalTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["send"]),
    )
    assert [item.code for item in check_manifest(covered.describe()).diagnostics] == [
        "AGENT_PROVIDER_NOT_FOUND"
    ]

    conditional = CayuApp(enable_logging=False)
    conditional.register_agent(
        AgentSpec(name="sender", model="model"),
        tools=[_ExternalTool()],
        tool_policy=ParameterConstrainedToolPolicy({"send": [RequiredFieldRule("recipient")]}),
    )
    assert [item.code for item in check_manifest(conditional.describe()).diagnostics] == [
        "AGENT_PROVIDER_NOT_FOUND"
    ]


def test_custom_policy_subclasses_are_not_described_as_trusted_builtins() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="sender", model="model"),
        tools=[_ExternalTool()],
        tool_policy=_PermissiveApprovalPolicy(tools=["send"]),
    )

    manifest = app.describe()
    report = check_manifest(manifest)

    assert manifest.agents[0].tools[0].policy_coverage == "unknown"
    assert [item.code for item in report.diagnostics] == [
        "AGENT_PROVIDER_NOT_FOUND",
        "EXTERNAL_TOOL_COVERAGE_UNKNOWN",
    ]
    finding = report.diagnostics[1]
    assert finding.severity == "error"
    assert finding.parameters["coverage"] == "unknown"
    assert "statically describable policy" in finding.hint


def test_diagnostic_parameters_are_copied_read_only_json() -> None:
    parameters = {"nested": {"value": "original"}}
    diagnostic = ProjectDiagnostic(
        code="TEST",
        severity="info",
        subject="test",
        path="test",
        message="test",
        parameters=parameters,
        verification_command="cayu check --json",
    )
    parameters["nested"]["value"] = "changed"

    assert diagnostic.parameters["nested"]["value"] == "original"
    with pytest.raises(TypeError):
        diagnostic.parameters["nested"]["value"] = "changed"  # type: ignore[index]
    with pytest.raises(ValidationError, match="JSON-compatible"):
        ProjectDiagnostic(
            code="TEST",
            severity="info",
            subject="test",
            path="test",
            message="test",
            parameters={"bad": object()},
            verification_command="cayu check --json",
        )

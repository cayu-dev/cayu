from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu import (
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


class _PermissiveApprovalPolicy(AlwaysRequireApprovalToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_builtin_diagnostic_codes_are_unique_and_compatibility_pinned() -> None:
    assert BUILTIN_DIAGNOSTIC_CODES == (
        "AGENT_PROVIDER_AMBIGUOUS",
        "AGENT_PROVIDER_NOT_FOUND",
        "APP_NO_AGENTS",
        "EXTERNAL_TOOL_COVERAGE_UNKNOWN",
        "EXTERNAL_TOOL_UNGUARDED",
    )
    assert len(BUILTIN_DIAGNOSTIC_CODES) == len(set(BUILTIN_DIAGNOSTIC_CODES))


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

    assert empty_codes | missing_codes | ambiguous_codes | unsafe_codes | unknown_codes == set(
        BUILTIN_DIAGNOSTIC_CODES
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

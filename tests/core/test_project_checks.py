from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.agents import AgentAuthoringState
from cayu.environments import (
    NO_ENVIRONMENT,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.evals import ScriptedModelProvider
from cayu.runners import LocalRunner
from cayu.runtime import ProjectDiagnostic, check_manifest
from cayu.runtime.checks import BUILTIN_DIAGNOSTIC_CODES, ProjectControlPlaneCheckEvidence
from cayu.runtime.service_manifest import PublicServiceManifest
from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.tools import (
    AlwaysRequireApprovalToolPolicy,
    ExecCommandTool,
    ParameterConstrainedToolPolicy,
    ProcessCommandPolicy,
    ReadKnowledgeTool,
    RequiredFieldRule,
    SearchKnowledgeTool,
    ToolPolicyDecision,
    ToolPolicyResult,
    WriteFileTool,
)
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.workspaces import LocalWorkspace

_BOUND = 64 * 1024


class _ExternalTool(Tool):
    spec = ToolSpec(
        name="send",
        effect=ToolEffect.EXTERNAL,
        input_schema={"type": "object", "additionalProperties": False},
        max_terminal_payload_bytes=_BOUND,
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
                max_terminal_payload_bytes=_BOUND,
            )
        )
        self._content = content

    async def run(self, ctx, args):
        return ToolResult(content=self._content)


class _UnconstrainedTool(Tool):
    spec = ToolSpec(name="lookup", effect=ToolEffect.NONE, max_terminal_payload_bytes=_BOUND)

    async def run(self, ctx, args):
        return ToolResult(content="found")


class _RemoteSchemaTool(Tool):
    spec = ToolSpec(
        name="fetch",
        effect=ToolEffect.NONE,
        input_schema={"$ref": "https://schemas.example/fetch.json"},
        max_terminal_payload_bytes=_BOUND,
    )

    async def run(self, ctx, args):
        return ToolResult(content="fetched")


class _UnboundedTool(Tool):
    spec = ToolSpec(
        name="report",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="reported")


class _PermissiveApprovalPolicy(AlwaysRequireApprovalToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _UncalledEnvironmentFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.called = False

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        del request
        self.called = True
        raise AssertionError("project checks must not materialize environment factories")


def test_builtin_diagnostic_codes_are_unique_and_compatibility_pinned() -> None:
    assert BUILTIN_DIAGNOSTIC_CODES == (
        "AGENT_ENVIRONMENT_NOT_FOUND",
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
        "TOOL_INPUT_SCHEMA_RUNTIME_UNSUPPORTED",
        "TOOL_APPROVAL_VALIDITY_RULE",
        "TOOL_TERMINAL_PAYLOAD_UNBOUNDED",
        "TOOL_KNOWLEDGE_STORE_UNAVAILABLE",
    )
    assert len(BUILTIN_DIAGNOSTIC_CODES) == len(set(BUILTIN_DIAGNOSTIC_CODES))


def test_public_service_deployment_rejects_in_memory_runtime_stores() -> None:
    report = check_manifest(
        CayuApp(enable_logging=False).describe(),
        service_manifest=PublicServiceManifest(
            mode="production",
            product_access="authenticated",
            operator_access="authenticated",
            identity_store="durable",
            runtime_session_store="development",
            runtime_task_store="development",
            product_api_path="/api",
            control_plane_path="/cayu",
        ),
        deploy_only=True,
    )

    assert [item.code for item in report.diagnostics] == [
        "PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE",
    ]
    assert report.service_evidence.configuration == "unsupported"


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


def test_workflow_file_and_command_tools_require_structural_environment_support() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name="operator",
            model="test-model",
            workflow_tool_names=("write_file", "exec_command"),
        ),
        tools=[WriteFileTool(), ExecCommandTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    report = check_manifest(app.describe())

    assert [item.code for item in report.diagnostics] == [
        "AGENT_WORKFLOW_COMMAND_POLICY_NOT_REGISTERED",
        "AGENT_WORKFLOW_RUNNER_NOT_REGISTERED",
        "AGENT_WORKFLOW_WORKSPACE_NOT_REGISTERED",
    ]
    assert report.diagnostics[0].path == ("agents.operator.tools.exec_command.command_policy")
    assert report.diagnostics[1].path == "agents.operator.workflow_tool_names.exec_command"
    assert report.diagnostics[2].path == "agents.operator.workflow_tool_names.write_file"


def test_workflow_environment_checks_accept_static_or_factory_backed_support(
    tmp_path,
) -> None:
    static = CayuApp(enable_logging=False)
    static.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    static.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path),
            runner=LocalRunner(tmp_path),
        ),
        default=True,
    )
    static.register_agent(
        AgentSpec(
            name="operator",
            model="test-model",
            workflow_tool_names=("write_file", "exec_command"),
        ),
        tools=[
            WriteFileTool(),
            ExecCommandTool(
                policy=ProcessCommandPolicy(
                    allowed_executables=["python3"],
                    allowed_cwds=[str(tmp_path)],
                )
            ),
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    factory = _UncalledEnvironmentFactory()
    dynamic = CayuApp(enable_logging=False)
    dynamic.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    dynamic.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    dynamic.register_agent(
        AgentSpec(
            name="operator",
            model="test-model",
            workflow_tool_names=("write_file", "exec_command"),
        ),
        tools=[
            WriteFileTool(),
            ExecCommandTool(
                policy=ProcessCommandPolicy(
                    allowed_executables=["python3"],
                    allowed_cwds=["/workspace"],
                )
            ),
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    assert check_manifest(static.describe()).diagnostics == ()
    assert check_manifest(dynamic.describe()).diagnostics == ()
    assert factory.called is False


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


def test_unconstrained_tool_schema_is_an_actionable_authoring_warning() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="test-model"),
        tools=[_UnconstrainedTool()],
    )

    report = check_manifest(app.describe())

    assert [item.code for item in report.diagnostics] == ["TOOL_INPUT_SCHEMA_UNCONSTRAINED"]
    finding = report.diagnostics[0]
    assert finding.severity == "warning"
    assert finding.path == "agents.researcher.tools.lookup.input_schema"
    assert finding.tags == ("authoring", "deploy")
    assert finding.parameters == {"agent": "researcher", "tool": "lookup"}
    assert "valid JSON Schema" in finding.message
    assert finding.documentation_anchor == (
        "cayu guide diagnostics#tool-input-schema-unconstrained"
    )


def test_tool_without_terminal_payload_bound_is_a_deploy_visible_warning() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(name="reviewer", model="test-model"),
        tools=[
            _UnboundedTool(),
            _ConfiguredTool("bounded", effect=ToolEffect.NONE, content="ok"),
            ExecCommandTool(),
            WriteFileTool(),
            # A foreground child keeps its exclusive lease on purpose.
            SubagentTool(app, agents={"child": SubagentSpec(agent_name="child")}),
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    manifest = app.describe()
    assert {
        tool.name for tool in manifest.agents[0].tools if tool._terminal_payload_bound_undeclared
    } == {"report"}

    report = check_manifest(manifest)

    findings = [
        item for item in report.diagnostics if item.code == "TOOL_TERMINAL_PAYLOAD_UNBOUNDED"
    ]
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "warning"
    assert finding.subject == "tool:reviewer/report"
    assert finding.path == "agents.reviewer.tools.report.max_terminal_payload_bytes"
    assert finding.tags == ("authoring", "deploy")
    assert finding.parameters == {"agent": "reviewer", "tool": "report"}
    assert "every call reserves the largest terminal result" in finding.message
    assert "ToolSpec(max_terminal_payload_bytes=...)" in finding.hint
    assert finding.documentation_anchor == (
        "cayu guide diagnostics#tool-terminal-payload-unbounded"
    )
    assert finding in check_manifest(manifest, deploy_only=True).diagnostics
    # A manifest read back from JSON cannot tell, so it reports nothing.
    reloaded = type(manifest).model_validate_json(manifest.model_dump_json())
    assert "TOOL_TERMINAL_PAYLOAD_UNBOUNDED" not in {
        item.code for item in check_manifest(reloaded).diagnostics
    }


def test_every_builtin_diagnostic_has_a_seeded_misconfiguration() -> None:
    empty_codes = {
        item.code for item in check_manifest(CayuApp(enable_logging=False).describe()).diagnostics
    }

    missing = CayuApp(enable_logging=False)
    missing.register_agent(AgentSpec(name="missing", model="model", provider_name="absent"))
    missing_codes = {item.code for item in check_manifest(missing.describe()).diagnostics}

    homeless = CayuApp(enable_logging=False)
    homeless.register_agent(AgentSpec(name="homeless", model="model"), environment_name="absent")
    homeless_codes = {item.code for item in check_manifest(homeless.describe()).diagnostics}

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

    validity = CayuApp(enable_logging=False)
    validity.register_agent(
        AgentSpec(name="validity", model="model"),
        tools=[_ExternalTool()],
        tool_policy=ParameterConstrainedToolPolicy(
            {"send": [RequiredFieldRule("to")]},
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
        ),
    )
    validity_codes = {item.code for item in check_manifest(validity.describe()).diagnostics}

    remote_schema = CayuApp(enable_logging=False)
    remote_schema.register_agent(
        AgentSpec(name="remote_schema", model="model"),
        tools=[_RemoteSchemaTool()],
    )
    remote_schema_codes = {
        item.code for item in check_manifest(remote_schema.describe()).diagnostics
    }

    misaligned = CayuApp(enable_logging=False)
    misaligned.register_agent(
        AgentSpec(
            name="misaligned",
            model="model",
            workflow_tool_names=("missing_tool",),
        )
    )
    misaligned_codes = {item.code for item in check_manifest(misaligned.describe()).diagnostics}

    structurally_incomplete = CayuApp(enable_logging=False)
    structurally_incomplete.register_agent(
        AgentSpec(
            name="operator",
            model="model",
            workflow_tool_names=("write_file", "exec_command"),
        ),
        tools=[WriteFileTool(), ExecCommandTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    structurally_incomplete_codes = {
        item.code for item in check_manifest(structurally_incomplete.describe()).diagnostics
    }

    unfinished = CayuApp(enable_logging=False)
    unfinished.register_agent(
        AgentSpec(
            name="unfinished",
            model="model",
            authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,
        )
    )
    unfinished_codes = {item.code for item in check_manifest(unfinished.describe()).diagnostics}

    unconstrained = CayuApp(enable_logging=False)
    unconstrained.register_agent(
        AgentSpec(name="unconstrained", model="model"),
        tools=[_UnconstrainedTool()],
    )
    unconstrained_codes = {
        item.code for item in check_manifest(unconstrained.describe()).diagnostics
    }

    unbounded = CayuApp(enable_logging=False)
    unbounded.register_agent(AgentSpec(name="unbounded", model="model"), tools=[_UnboundedTool()])
    unbounded_codes = {item.code for item in check_manifest(unbounded.describe()).diagnostics}

    knowledgeless = CayuApp(enable_logging=False)
    knowledgeless.register_agent(
        AgentSpec(name="knowledgeless", model="model"), tools=[SearchKnowledgeTool()]
    )
    knowledgeless_codes = {
        item.code for item in check_manifest(knowledgeless.describe()).diagnostics
    }
    public_service_codes = {
        item.code
        for item in check_manifest(
            unconstrained.describe(),
            service_manifest=PublicServiceManifest(
                mode="development",
                product_access="placeholder",
                operator_access="open",
                identity_store="development",
                runtime_session_store="development",
                runtime_task_store="missing",
                product_api_path="/api",
                control_plane_path="/cayu",
            ),
        ).diagnostics
    }
    nondurable_task_service_codes = {
        item.code
        for item in check_manifest(
            unconstrained.describe(),
            service_manifest=PublicServiceManifest(
                mode="production",
                product_access="authenticated",
                operator_access="authenticated",
                identity_store="durable",
                runtime_session_store="durable",
                runtime_task_store="development",
                product_api_path="/api",
                control_plane_path="/cayu",
            ),
        ).diagnostics
    }
    project_control_plane_codes = {
        item.code
        for item in check_manifest(
            unconstrained.describe(),
            project_control_plane=ProjectControlPlaneCheckEvidence(
                project_identity_configured=False,
                eval_store_configured=False,
                service_context="migration_required",
            ),
        ).diagnostics
    }

    assert (
        empty_codes
        | validity_codes
        | remote_schema_codes
        | missing_codes
        | homeless_codes
        | ambiguous_codes
        | unsafe_codes
        | unknown_codes
        | misaligned_codes
        | structurally_incomplete_codes
        | unfinished_codes
        | unconstrained_codes
        | unbounded_codes
        | knowledgeless_codes
        | public_service_codes
        | nondurable_task_service_codes
        | project_control_plane_codes
        == set(BUILTIN_DIAGNOSTIC_CODES)
    )


def test_knowledge_tools_without_a_store_in_their_environment_are_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryKnowledgeStore()
    scope = KnowledgeAccessScope.for_namespace("project:app")
    # The app's store alone doesn't reach tool calls; the environment must carry one.
    app = CayuApp(enable_logging=False, knowledge_store=store, knowledge_access_scope=scope)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="plain")), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="stocked"), knowledge_store=store, knowledge_access_scope=scope
        )
    )
    factory = _UncalledEnvironmentFactory()
    app.register_environment_factory(EnvironmentSpec(name="custom"), factory)
    knowledge_tools = [SearchKnowledgeTool(), ReadKnowledgeTool()]
    with caplog.at_level(logging.WARNING, logger="cayu._application_agent_registry"):
        app.register_agent(AgentSpec(name="plain", model="model"), tools=knowledge_tools)
        app.register_agent(
            AgentSpec(name="toolonly", model="model"),
            tools=knowledge_tools,
            environment_name=NO_ENVIRONMENT,
        )
        app.register_agent(
            AgentSpec(name="stocked", model="model"),
            tools=knowledge_tools,
            environment_name="stocked",
        )
        # A factory that doesn't say which store it attaches isn't reported.
        app.register_agent(
            AgentSpec(name="custom", model="model"),
            tools=knowledge_tools,
            environment_name="custom",
        )
        app.register_agent(AgentSpec(name="none", model="model"))
    warned = sorted(
        record.getMessage().split()[1]
        for record in caplog.records
        if "missing_knowledge_store" in record.getMessage()
    )
    assert warned == ["plain", "toolonly"]

    manifest = app.describe()
    findings = [
        item
        for item in check_manifest(manifest).diagnostics
        if item.code == "TOOL_KNOWLEDGE_STORE_UNAVAILABLE"
    ]
    assert [item.subject for item in findings] == [
        "tool:plain/read_knowledge",
        "tool:plain/search_knowledge",
        "tool:toolonly/read_knowledge",
        "tool:toolonly/search_knowledge",
    ]
    finding = findings[0]
    assert finding.severity == "warning"
    assert finding.path == "agents.plain.tools.read_knowledge"
    assert finding.parameters == {"agent": "plain", "tool": "read_knowledge"}
    assert "missing_knowledge_store" in finding.message
    assert "knowledge_store=" in finding.hint
    assert finding.documentation_anchor == (
        "cayu guide diagnostics#tool-knowledge-store-unavailable"
    )
    assert factory.called is False
    # A manifest read back from JSON cannot tell, so it reports nothing.
    reloaded = type(manifest).model_validate_json(manifest.model_dump_json())
    assert "TOOL_KNOWLEDGE_STORE_UNAVAILABLE" not in {
        item.code for item in check_manifest(reloaded).diagnostics
    }


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


def test_default_policy_accepts_read_file_and_ask_user_but_not_implicit_writes() -> None:
    from cayu import AgentSpec, CayuApp
    from cayu.evals import ScriptedModelProvider
    from cayu.runtime.checks import check_manifest
    from cayu.tools import StaticToolPolicy
    from cayu.tools.files import ReadFileTool, WriteFileTool
    from cayu.tools.user_input import UserInputTool

    def unguarded(policy: StaticToolPolicy | None) -> set[str]:
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake"),
            tools=[UserInputTool(), ReadFileTool(), WriteFileTool()],
            **({} if policy is None else {"tool_policy": policy}),
        )
        manifest = app.describe()
        return {
            diagnostic.subject
            for diagnostic in check_manifest(manifest).diagnostics
            if diagnostic.code == "EXTERNAL_TOOL_UNGUARDED"
        }

    assert unguarded(None) == {"tool:assistant/write_file"}
    assert unguarded(StaticToolPolicy()) == {"tool:assistant/write_file"}
    assert unguarded(StaticToolPolicy(allow=["ask_user", "read_file", "write_file"])) == set()


def test_explicit_allow_is_not_published_in_the_manifest() -> None:
    from cayu import AgentSpec, CayuApp
    from cayu.evals import ScriptedModelProvider
    from cayu.runtime.manifest import AppManifest
    from cayu.tools import StaticToolPolicy
    from cayu.tools.files import WriteFileTool

    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[WriteFileTool()],
        tool_policy=StaticToolPolicy(allow=["write_file"]),
    )
    manifest = app.describe()
    serialized = manifest.model_dump_json()
    assert "allows_explicitly" not in serialized
    # A manifest loaded from JSON keeps the conservative default.
    reloaded = AppManifest.model_validate_json(serialized)
    assert reloaded.agents[0].tools[0]._policy_allows_explicitly is False


@pytest.mark.parametrize("bound", [None, 64 * 1024])
def test_process_isolated_application_tool_reports_missing_bound(
    bound, caplog, monkeypatch
) -> None:
    import logging

    from cayu.runtime import ExecutionProfileBehaviorIdentity
    from cayu.tools import (
        ProcessIsolatedTool,
        ProcessIsolatedToolFactoryRef,
        ProcessIsolatedToolLimits,
    )

    # This checks registration diagnostics without launching a child process.
    monkeypatch.setattr(
        "cayu.runtime._isolated_tool_process._complete_process_tree_supervision_available",
        lambda: True,
    )
    identity = ExecutionProfileBehaviorIdentity(
        name="isolated-report", behavior_version="1", implementation_version="1"
    )
    tool = ProcessIsolatedTool(
        ToolSpec(
            name="isolated_report",
            max_terminal_payload_bytes=bound,
            execution_profile_identity=identity,
        ),
        factory=ProcessIsolatedToolFactoryRef(
            module="cayu.testing.isolated_tools",
            qualname="build_deterministic_isolated_tool",
            identity=identity,
        ),
        limits=ProcessIsolatedToolLimits(deadline_seconds=1),
    )
    app = CayuApp(enable_logging=False)
    with caplog.at_level(logging.WARNING, logger="cayu._application_agent_registry"):
        app.register_agent(AgentSpec(name="worker", model="test-model"), tools=[tool])
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "declares no max_terminal_payload_bytes" in record.getMessage()
    ]
    findings = [
        item
        for item in check_manifest(app.describe()).diagnostics
        if item.code == "TOOL_TERMINAL_PAYLOAD_UNBOUNDED"
    ]
    assert len(warnings) == len(findings) == (1 if bound is None else 0)
    if findings:
        assert findings[0].parameters == {"agent": "worker", "tool": "isolated_report"}

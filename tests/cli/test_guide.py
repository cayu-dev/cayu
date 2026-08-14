from __future__ import annotations

import asyncio
import re

import pytest
from tests._session_provenance import fixture_session_invocation

from cayu import (
    AgentSpec,
    ResolutionActor,
    ResolutionActorSource,
    Session,
    ToolContext,
    ToolPolicyDecision,
    ToolPolicyRequest,
)
from cayu.cli import main


def test_bare_guide_lists_topics_and_help_describes_them(capsys) -> None:
    assert main(["guide"]) == 0
    listing = capsys.readouterr().out
    assert "Package-shipped Cayu guides:" in listing
    assert "structured-output" in listing
    assert "Credential-free structured-output runtime proof." in listing
    assert "durable-operations" in listing
    assert "providers" in listing

    with pytest.raises(SystemExit) as excinfo:
        main(["guide", "--help"])

    assert excinfo.value.code == 0
    help_output = capsys.readouterr().out
    assert "TOPIC[#SECTION]" in help_output
    assert "Primary integrations and compatible Chat Completions endpoints." in help_output


def test_guide_accepts_emitted_section_anchors(capsys) -> None:
    assert main(["guide", "diagnostics#app-no-agents"]) == 0
    section = capsys.readouterr().out

    assert section.startswith("## app-no-agents")
    assert "APP_NO_AGENTS" in section
    assert "agent-provider-not-found" not in section


def test_package_shipped_authoring_and_diagnostic_guides_are_discoverable(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    assert "# Building applications with Cayu" in authoring
    assert "Start with one model-only agent" in authoring
    assert "## Cayu Map" in authoring
    assert "A tool-backed slice is optional" in authoring
    assert "Clarify users, jobs, triggers" not in authoring
    assert "Model-controlled command selectors are untrusted argv input" in authoring
    assert "An executable allowlist does not authorize its argument protocol" in authoring
    assert "do not replace container or microVM isolation" in authoring
    assert "workflow_tool_names" in authoring
    assert "registered for that same agent" in authoring
    assert "cannot prove prompt comprehension" in authoring
    assert "cayu guide tool-effects" in authoring
    assert "uv run cayu serve --dev" in authoring
    assert "http://127.0.0.1:8000/cayu/" in authoring
    assert "A deployed control plane still requires configured authentication" in authoring
    assert "Never use `OpenAccess()` on a public listener" in authoring
    assert "Client-IP and forwarded-header checks are not authentication" in authoring

    assert main(["guide", "references#server"]) == 0
    server = capsys.readouterr().out
    assert "uv run cayu serve --dev" in server
    assert "http://127.0.0.1:8000/cayu/" in server
    assert 'mount_cayu(..., path="/cayu")' in server
    assert "requires `AuthenticatedAccess(...)` on any public listener" in server.replace("\n", " ")
    assert "Do not substitute client-IP or forwarded-header checks" in server.replace("\n", " ")

    assert main(["guide", "diagnostics"]) == 0
    diagnostics = capsys.readouterr().out
    assert "# Cayu project diagnostics" in diagnostics
    assert "## agent-generated-tracer-bullet-unfinished" in diagnostics
    assert "## agent-provider-not-found" in diagnostics
    assert "## agent-workflow-tool-not-registered" in diagnostics


def test_package_shipped_application_anatomy_guide_is_discoverable(capsys) -> None:
    assert main(["guide", "anatomy"]) == 0
    anatomy = capsys.readouterr().out

    assert "# Cayu application anatomy" in anatomy
    assert "## Application lifecycle boundaries" in anatomy
    assert "SQLite store constructors open their files" in anatomy
    assert "## Process roles" in anatomy
    for role in (
        "One-off script",
        "Interactive console",
        "Server integration",
        "Worker integration",
        "Test",
    ):
        assert role in anatomy


def test_package_shipped_durable_operations_guide_is_runnable(capsys) -> None:
    assert main(["guide", "durable-operations"]) == 0
    guide = capsys.readouterr().out
    normalized = guide.casefold()

    assert "# Durable operations lifecycle" in guide
    for phase in ("observe", "diagnose", "propose", "authorize", "act once", "verify", "recover"):
        assert phase in normalized
    assert "never automatically replay" in normalized
    assert "effect=ToolEffect.IDEMPOTENT" in guide
    assert "class RecordProposal(Tool)" in guide
    assert "proposal_store.records[proposal.session.id]" in guide
    assert "ResolutionActorSource.REQUEST" in guide
    assert "resolved_by=authorized_operator" in guide
    assert "self.system.observe_target(action_id)" in guide
    assert 'self.system.apply_once(action_id, args["target"])' in guide
    assert "verified = observed_target == ALLOWED_TARGET" in guide
    assert "self.action.receipts.get" not in guide
    guide.encode("cp437")
    example = re.search(r"```python\n(.*?)```", guide, re.DOTALL)
    assert example is not None
    namespace: dict[str, object] = {}
    exec(example.group(1), namespace)

    proposal_store = namespace["ProposalStore"]()
    record_proposal = namespace["RecordProposal"](proposal_store)
    assert record_proposal.spec.parallel_safe is False
    policy = namespace["ReviewedActionPolicy"](proposal_store)
    session = Session(
        id="same-round",
        agent_name="operator",
        provider_name="scripted",
        model="scripted-model",
        causal_budget_id="same-round",
        invocation=fixture_session_invocation("same-round"),
    )
    request = ToolPolicyRequest(
        session=session,
        agent=AgentSpec(name="operator", model="scripted-model"),
        tool_name="apply_change",
        tool_call_id="action-call",
        arguments={
            "action_id": "change-0001",
            "target": "demo",
            "reason": "observed_drift",
        },
    )
    missing_record = asyncio.run(policy.authorize(request))
    assert missing_record.decision is ToolPolicyDecision.DENY

    proposal_store.records[session.id] = {
        "action_id": "change-0002",
        "target": "demo",
        "reason": "observed_drift",
    }
    mismatch = asyncio.run(policy.authorize(request))
    assert mismatch.decision is ToolPolicyDecision.DENY

    unknown = request.model_copy(
        update={
            "tool_name": "future_effect",
            "tool_call_id": "future-effect-call",
            "arguments": {},
        }
    )
    unknown_result = asyncio.run(policy.authorize(unknown))
    assert unknown_result.decision is ToolPolicyDecision.DENY

    system = namespace["FakeSystem"]()
    action = namespace["ApplyChange"](namespace["ProposalStore"](), system)
    assert action.spec.parallel_safe is False
    invalid_action = asyncio.run(
        action.run(
            ToolContext(session_id="invalid-action"),
            {
                "action_id": "change-0001",
                "target": "demo",
                "reason": "secret",
            },
        )
    )
    assert invalid_action.is_error
    assert system.operations == {}

    with pytest.raises(ValueError, match="action_id"):
        namespace["validate_proposal"](
            {
                "action_id": "change-١٢٣٤",
                "target": "demo",
                "reason": "observed_drift",
            }
        )

    invalid_verification = asyncio.run(
        namespace["VerifyChange"](system).run(
            ToolContext(session_id="invalid-verification"),
            {"action_id": "unbounded secret value"},
        )
    )
    assert invalid_verification.is_error
    assert invalid_verification.structured == {"verified": False}

    malformed_actor = ResolutionActor(
        subject="operator@example.test",
        tenant="demo-tenant",
        source=ResolutionActorSource.REQUEST,
        claims={"roles": "operations-approver"},
    )
    with pytest.raises(PermissionError, match="role"):
        namespace["require_authorized_operator"](
            malformed_actor,
            {
                "action_id": "change-0001",
                "target": "demo",
                "reason": "observed_drift",
            },
        )

    assert main(["guide", "durable-operations#recovery-decision-table"]) == 0
    recovery = capsys.readouterr().out
    assert recovery.startswith("## Recovery decision table")
    assert "External action started but no trustworthy terminal receipt" in recovery
    assert "## Unsafe shortcuts" not in recovery


def test_authoring_and_reference_guides_route_operations_to_the_quickstart(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = " ".join(capsys.readouterr().out.split())
    assert "Durable operational changes" in authoring
    assert "cayu guide durable-operations" in authoring
    assert "`cayu guide durable-operations`; model stable action identity" in authoring

    assert main(["guide", "references#approvals"]) == 0
    approvals = " ".join(capsys.readouterr().out.split())
    assert "runnable proposal-to-verification recipe" in approvals
    assert "cayu guide durable-operations" in approvals


def test_package_shipped_tool_effect_guide_renders_canonical_decisions(capsys) -> None:
    assert main(["guide", "tool-effects"]) == 0
    guidance = capsys.readouterr().out
    normalized = guidance.casefold()

    assert "# Choosing a ToolEffect" in guidance
    assert "public http read" in normalized
    assert "`NONE`" in guidance
    assert "paid or logged read" in normalized
    assert "stable downstream idempotency key" in normalized
    assert "stable operation identity or equivalent idempotency contract" in normalized
    assert "durable snapshot or artifact" in normalized
    assert "outcome is unknown after a timeout" in normalized
    assert "does not authorize execution" in guidance
    assert "verify_tool_effect" in guidance
    assert "bounded temporary Cayu workspace" in guidance
    assert "`cayu check` remains structural" in guidance


def test_domain_tool_reference_documents_the_public_authoring_interface(capsys) -> None:
    assert main(["guide", "references#domain-tool"]) == 0
    guidance = capsys.readouterr().out

    assert guidance.startswith("## domain-tool")
    normalized = " ".join(guidance.split())
    for field in (
        "`name`",
        "`description`",
        "`input_schema`",
        "`parallel_safe`",
        "`effect`",
        "`session_id`",
        "`content`",
        "`structured`",
        "`artifacts`",
        "`is_error`",
    ):
        assert field in guidance
    assert '`ToolContext(session_id="test-session")`' in guidance
    assert "the only required `ToolContext` constructor field" in normalized
    assert "Cayu constructs `ToolContext` for normal runtime execution" in normalized
    assert "runtime-owned secret-capture hooks" in normalized
    assert "`mcp_servers` is always a tuple" in normalized
    assert "absence is represented by `()`" in normalized
    assert "rather than comparing it with `None`" in normalized
    assert "`name` and `description`" in normalized
    assert "pass a `ToolSpec` to the inherited constructor" in normalized
    assert "asyncio.run(tool.run(context" in guidance
    assert "does not prove schema validation, policy authorization" in guidance
    example = re.search(r"```python\n(.*?)```", guidance, re.DOTALL)
    assert example is not None
    exec(example.group(1), {})


def test_every_cayu_map_row_routes_to_a_package_shipped_local_guide(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    rows = [line for line in authoring.splitlines() if line.startswith("|")][2:]

    assert len(rows) >= 20
    for row in rows:
        commands = re.findall(r"`(cayu guide [^`]+)`", row)
        assert commands, row
        for command in commands:
            assert main(command.split()[1:]) == 0, command
            assert capsys.readouterr().out


def test_structured_output_and_provider_guides_are_credential_free_and_public(capsys) -> None:
    assert main(["guide", "structured-output"]) == 0
    structured = capsys.readouterr().out
    assert "scripted_structured_output" in structured
    assert "invalid first" in structured
    assert "outcome.structured_output.output" in structured
    assert "cayu.runtime" not in structured

    assert main(["guide", "providers#primary-integrations"]) == 0
    providers = capsys.readouterr().out
    assert "AnthropicProvider" in providers
    assert "ANTHROPIC_API_KEY" in providers


def test_package_shipped_provider_guide_is_short_and_agent_discoverable(capsys) -> None:
    assert main(["guide", "providers"]) == 0
    guide = capsys.readouterr().out

    assert "# Cayu providers" in guide
    for primary_service in (
        "OpenAI Platform",
        "Anthropic API",
        "Google AI Studio",
        "Amazon Bedrock",
        "Anthropic on Vertex AI",
    ):
        assert primary_service in guide

    compatible_services = (
        "OpenRouter",
        "Fireworks",
        "Baseten Model APIs",
        "OpenCode Go",
        "Together AI",
        "Mistral AI",
        "Google AI Studio",
        "Ollama",
        "vLLM",
    )
    for service in compatible_services:
        assert service in guide

    for configuration_value in (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        "https://inference.baseten.co/v1",
        "BASETEN_API_KEY",
        "https://opencode.ai/zen/go/v1",
        "OPENCODE_API_KEY",
    ):
        assert configuration_value in guide

    assert "`CAYU_PROVIDER` is only a scaffold convenience" in guide
    assert "ChatCompletionsProvider" in guide
    assert "AgentSpec.provider_name" in guide
    assert "`opencode-go/...`" in guide
    assert "authenticated live inference" not in guide
    assert "Route/auth only" not in guide
    assert len(guide.splitlines()) < 140

    assert main(["guide", "providers#compatible-chat-completions"]) == 0
    compatible = capsys.readouterr().out
    assert compatible.startswith("## Compatible Chat Completions")
    assert "OpenRouter" in compatible
    assert "OpenCode Go" in compatible

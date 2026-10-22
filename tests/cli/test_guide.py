from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from tests._session_provenance import session_fixture

from cayu import AgentSpec, ToolContext
from cayu.approvals import ResolutionActor, ResolutionActorSource
from cayu.cli import main
from cayu.cli._identity import CAYU_CAPABILITIES, CAYU_IDENTITY, CAYU_IS_NOT
from cayu.cli.guide import _GUIDES
from cayu.tools import ToolPolicyDecision, ToolPolicyRequest


@pytest.mark.parametrize("topic", _GUIDES)
def test_every_registered_guide_is_listed_and_readable(topic: str, capsys) -> None:
    filename, description = _GUIDES[topic]
    assert main(["guide", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    entries = [entry for entry in listing["topics"] if entry["topic"] == topic]
    assert len(entries) == 1
    assert entries[0]["summary"] == description
    assert entries[0]["package_source"] == f"cayu.guides/{filename}"

    assert main(["guide", topic]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# ")
    assert "<!-- cayu-guide-include:" not in text

    assert main(["guide", topic, "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["topic"] == topic
    assert document["package_source"] == entries[0]["package_source"]
    assert document["content"].strip() == text.strip()


def test_durable_service_guide_discovery_and_executable_injection(capsys) -> None:
    from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry

    assert main(["guide", "durable-service-tools", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["package_source"] == "cayu.guides/durable-service-tools.md"
    guide = result["content"]
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="rule", namespace="handbook", text="shipping authorized-evidence")],
        access_scope=KnowledgeAccessScope.for_namespace("handbook"),
    )
    namespace: dict[str, Any] = {"store": store}
    for snippet in re.findall(r"```python\n(.*?)```", guide, re.DOTALL):
        exec(compile(snippet, "durable-service-tools", "exec"), namespace)
    reader = namespace["HandbookReader"](store)
    output = asyncio.run(reader.run(ToolContext(session_id="guide-test"), {}))
    assert "authorized-evidence" in output.content
    environment = namespace["environment"]
    assert environment.knowledge_store is store
    assert environment.knowledge_access_scope == store.bound_access_scope()
    with pytest.raises(ValueError, match="bound knowledge store"):
        namespace["HandbookReader"](None)

    assert main(["guide", "durable-service-tools#bind-through-an-environment"]) == 0
    assert "ctx.knowledge_store.search" in capsys.readouterr().out
    for topic in ["authoring", "references#domain-tool", "durable-operations"]:
        assert main(["guide", topic]) == 0
        assert "cayu guide durable-service-tools" in capsys.readouterr().out
    root = Path(__file__).resolve().parents[2]
    for index in ["README.md", "docs/README.md", "examples/README.md"]:
        assert (
            "durable_service_tools" in (root / index).read_text()
            or "durable-service-tools" in (root / index).read_text()
        )


def test_bare_guide_lists_topics_and_help_describes_them(capsys) -> None:
    assert main(["guide"]) == 0
    listing = capsys.readouterr().out
    assert "Package-shipped Cayu guides:" in listing
    assert "structured-output" in listing
    assert "Credential-free structured-output runtime proof." in listing
    assert "durable-operations" in listing
    assert "durable-service-tools" in listing
    assert "evals-ai-quality" in listing
    assert "evals-first" in listing
    assert "evals-production" in listing
    assert "providers" in listing
    assert "thinking" in listing

    with pytest.raises(SystemExit) as excinfo:
        main(["guide", "--help"])

    assert excinfo.value.code == 0
    help_output = capsys.readouterr().out
    assert "TOPIC[#SECTION]" in help_output
    assert "Primary integrations and compatible Chat Completions endpoints." in help_output


def test_bare_guide_starts_with_the_cayu_identity_and_capability_parts(capsys) -> None:
    assert main(["guide"]) == 0
    listing = capsys.readouterr().out

    assert listing.startswith(CAYU_IDENTITY + "\n")
    for capability, parts in CAYU_CAPABILITIES:
        assert f"| {capability} | {parts} |" in listing
    assert CAYU_IS_NOT in listing
    assert listing.index(CAYU_IS_NOT) < listing.index("Package-shipped Cayu guides:")

    assert main(["guide", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["identity"] == CAYU_IDENTITY


def test_package_shipped_eval_guides_cover_the_three_user_journeys(capsys) -> None:
    assert main(["guide", "evals-first"]) == 0
    first = capsys.readouterr().out
    assert "# Run your first evaluation" in first
    assert "New evaluation" in first
    assert "Check launch" in first
    assert "Approve baseline" in first
    assert "cayu eval compare" in first

    assert main(["guide", "evals-ai-quality", "--json"]) == 0
    quality = json.loads(capsys.readouterr().out)
    assert quality["package_source"] == "cayu.guides/evals-ai-quality.md"
    assert quality["related_topics"] == ["evals-first", "evals-production", "authoring"]
    assert "[tool.cayu.evals.default_judge]" in quality["content"]
    assert "allow_same_model = true" in quality["content"]
    assert "Calibrate before gating" in quality["content"]

    assert main(["guide", "evals-production"]) == 0
    production = capsys.readouterr().out
    assert "# Evaluate production sessions, scenarios, and memory" in production
    assert "Evaluate a retained session" in production
    assert "Build a controlled multi-stage scenario" in production
    assert "Structural exposure" in production
    assert "Semantic use" in production
    assert "Causal contribution" in production


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


def test_application_guide_json_is_versioned_and_offline_safe(capsys) -> None:
    assert main(["guide", "applications#convention", "--json"]) == 0
    guide = json.loads(capsys.readouterr().out)

    assert guide["schema_version"] == 1
    assert guide["topic"] == "applications"
    assert guide["section"] == "convention"
    assert guide["package_source"] == "cayu.guides/applications.md"
    assert guide["verification_commands"] == [
        "uv run --no-sync cayu inspect --json",
        "uv run --no-sync cayu check --fail-on warning --json",
    ]


def test_package_shipped_durable_operations_guide_is_runnable(capsys) -> None:
    assert main(["guide", "durable-operations"]) == 0
    guide = capsys.readouterr().out
    normalized = guide.casefold()

    assert "# Durable operations lifecycle" in guide
    assert "## Artifact-write cancellation settlement" in guide
    assert "ArtifactWriteSettlementEvidence" in guide
    assert "register_artifact_write_operation" in guide
    assert "reconciliation-required" in guide
    for phase in ("observe", "diagnose", "propose", "authorize", "act once", "verify", "recover"):
        assert phase in normalized
    assert "never automatically replay" in normalized
    assert "cayu guide tool-effects#act-once-recovery" in guide
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
    namespace: dict[str, Any] = {}
    exec(example.group(1), namespace)

    proposal_store = namespace["ProposalStore"]()
    record_proposal = namespace["RecordProposal"](proposal_store)
    assert record_proposal.spec.parallel_safe is False
    policy = namespace["ReviewedActionPolicy"](proposal_store)
    session = session_fixture(
        id="same-round",
        agent_name="operator",
        provider_name="scripted",
        model="scripted-model",
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


def _load_act_once_example(
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, dict[str, Any]]:
    assert main(["guide", "tool-effects#act-once-recovery"]) == 0
    guidance = capsys.readouterr().out
    example = re.search(r"```python\n(.*?)```", guidance, re.DOTALL)
    assert example is not None
    namespace: dict[str, Any] = {}
    exec(example.group(1), namespace)
    return guidance, namespace


def test_act_once_recovery_example_reconciles_lost_acknowledgement_without_redispatch(
    capsys,
) -> None:
    guidance, namespace = _load_act_once_example(capsys)
    normalized = " ".join(guidance.split())

    assert guidance.startswith("## Act-once recovery")
    for state in (
        "proposed",
        "authorized",
        "intent_recorded",
        "completed",
        "outcome_unknown",
        "safe_to_retry",
        "manual_review",
    ):
        assert state in guidance
    assert "commit-then-raise" in normalized
    assert "lost success acknowledgement" in normalized
    assert "missing local receipt is not proof" in normalized
    assert "reconcile before any retry" in normalized
    assert "does not create generic exactly-once execution" in normalized
    assert "`ToolEffect` classifies replay risk; it does not authorize execution" in normalized

    operation_id = "change-0001"
    record, external_system, timeline = namespace["run_commit_then_raise_contract"]()
    assert record["state"] == "completed"
    assert record["receipt"] == {
        "operation_id": operation_id,
        "target": "demo",
        "status": "committed",
    }
    assert record["verification"] == {
        "operation_id": operation_id,
        "target": "demo",
        "verified": True,
    }
    assert external_system.dispatch_count == 1
    assert external_system.operation_ids_seen == [operation_id] * len(
        external_system.operation_ids_seen
    )
    assert [state for state, _operation_id in timeline] == [
        "proposed",
        "authorized",
        "intent_recorded",
        "outcome_unknown",
        "completed",
    ]
    assert {_operation_id for _state, _operation_id in timeline} == {operation_id}


@pytest.mark.parametrize(
    "verification",
    [
        {
            "operation_id": "change-0001",
            "target": "demo",
            "verified": False,
        },
        {
            "operation_id": "change-0001",
            "target": "demo",
        },
        {
            "operation_id": "different-operation",
            "target": "demo",
            "verified": True,
        },
        {
            "operation_id": "change-0001",
            "target": "different-target",
            "verified": True,
        },
    ],
)
def test_act_once_recovery_example_requires_bound_positive_verification(
    capsys,
    verification: dict[str, object],
) -> None:
    _guidance, namespace = _load_act_once_example(capsys)
    base_system = namespace["CommitThenRaiseSystem"]

    class ContradictoryVerificationSystem(base_system):
        def verify(self, operation_id: str) -> dict[str, object]:
            self.operation_ids_seen.append(operation_id)
            return verification

    namespace["CommitThenRaiseSystem"] = ContradictoryVerificationSystem
    record, external_system, timeline = namespace["run_commit_then_raise_contract"]()

    assert record["state"] == "manual_review"
    assert record["verification"] == verification
    assert external_system.dispatch_count == 1
    assert [state for state, _operation_id in timeline] == [
        "proposed",
        "authorized",
        "intent_recorded",
        "outcome_unknown",
        "manual_review",
    ]


@pytest.mark.parametrize(
    "receipt",
    [
        {
            "operation_id": "different-operation",
            "target": "demo",
            "status": "committed",
        },
        {
            "target": "demo",
            "status": "committed",
        },
        {
            "operation_id": "change-0001",
            "target": "different-target",
            "status": "committed",
        },
        {
            "operation_id": "change-0001",
            "status": "committed",
        },
        {
            "operation_id": "change-0001",
            "target": "demo",
            "status": "failed",
        },
        {
            "operation_id": "change-0001",
            "target": "demo",
        },
    ],
)
def test_act_once_recovery_example_requires_bound_committed_receipt(
    capsys,
    receipt: dict[str, object],
) -> None:
    _guidance, namespace = _load_act_once_example(capsys)
    base_system = namespace["CommitThenRaiseSystem"]

    class ContradictoryReceiptSystem(base_system):
        def reconcile(self, operation_id: str) -> dict[str, object]:
            self.operation_ids_seen.append(operation_id)
            return receipt

    namespace["CommitThenRaiseSystem"] = ContradictoryReceiptSystem
    record, external_system, timeline = namespace["run_commit_then_raise_contract"]()

    assert record["state"] == "manual_review"
    assert record["receipt"] == receipt
    assert record["verification"] is None
    assert external_system.dispatch_count == 1
    assert [state for state, _operation_id in timeline] == [
        "proposed",
        "authorized",
        "intent_recorded",
        "outcome_unknown",
        "manual_review",
    ]


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
    assert main(["guide", "authoring#cayu-map"]) == 0
    authoring = capsys.readouterr().out
    rows = [line for line in authoring.splitlines() if line.startswith("|")][2:]

    assert rows
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


def test_package_shipped_provider_guide_is_agent_discoverable(capsys) -> None:
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
    assert "`cayu new APP --provider openrouter`" in guide
    assert "explicit `CAYU_MODEL=vendor/model`" in guide
    assert "reasoning_details" in guide
    assert "ChatCompletionsProvider" in guide
    assert "AgentSpec.provider_name" in guide
    assert "`opencode-go/...`" in guide
    assert "authenticated live inference" not in guide
    assert "Route/auth only" not in guide

    assert main(["guide", "providers#compatible-chat-completions"]) == 0
    compatible = capsys.readouterr().out
    assert compatible.startswith("## Compatible Chat Completions")
    assert "OpenCode Go" in compatible

    assert main(["guide", "providers#openai-subscription"]) == 0
    subscription = capsys.readouterr().out
    assert subscription.startswith("## OpenAI subscription")
    assert "evaluation only" in subscription
    assert "`gpt-6-luna`" in subscription


def test_scaffold_guide_references_resolve(capsys) -> None:
    scaffold_sources = sorted((Path(__file__).parents[2] / "src/cayu/cli").glob("scaffold*.py"))
    references = {
        match.group(1)
        for source in scaffold_sources
        for match in re.finditer(r"cayu guide ([a-z0-9-]+#[a-z0-9-]+)", source.read_text())
    }

    assert "providers#openai-subscription" in references
    for reference in sorted(references):
        assert main(["guide", reference]) == 0, reference
        capsys.readouterr()


def test_human_attention_guide_discovery_and_public_contracts(capsys) -> None:
    assert main(["guide", "human-attention", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["package_source"] == "cayu.guides/human-attention.md"
    assert "get_human_attention_state" in result["content"]
    assert "HumanAttentionRequest.from_pending_action" in result["content"]
    assert "session_instance" not in result["content"] or "incarnation" in result["content"]
    assert "unavailable" in result["content"] and "next_cursor" in result["content"]
    assert main(["guide", "human-attention#one-underlying-action"]) == 0
    assert "attention_id" in capsys.readouterr().out


def test_installed_order_support_discovery(capsys) -> None:
    assert main(["guide", "order-support", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["package_source"] == "cayu.guides/order-support.md"
    assert "python -m cayu.examples.order_support" in result["content"]
    assert "receipt_to_native_request" in result["content"]
    for topic in (
        "authoring",
        "references#sessions",
        "references#approvals",
        "durable-operations",
        "durable-service-tools",
    ):
        assert main(["guide", topic]) == 0
        assert "cayu guide order-support" in capsys.readouterr().out


def test_verified_work_guide_discovery_and_reference_example(capsys) -> None:
    assert main(["guide", "verified-work", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["package_source"] == "cayu.guides/verified-work.md"
    guide = result["content"]
    assert "examples/durable_file_workflow/verified.py" in guide
    assert "cayu.verified-task-continuation.v1" in guide
    for status_reason in (
        "work_contract_attempt_limit",
        "work_contract_repeated_gap_limit",
        "work_contract_rejected",
    ):
        assert status_reason in guide
    assert set(result["related_topics"]) >= {"durable-operations", "tool-effects"}
    assert main(["guide", "verified-work#five-layers-five-owners"]) == 0
    assert "Completion proposal" in capsys.readouterr().out
    root = Path(__file__).resolve().parents[2]
    assert (root / "examples/durable_file_workflow/verified.py").is_file()
    readme = (root / "README.md").read_text()
    assert "`cayu guide verified-work`" in readme
    assert "examples/durable_file_workflow/verified.py" in readme
    assert "guides/verified-work.md" in (root / "docs/README.md").read_text()
    examples_index = (root / "examples/README.md").read_text()
    assert "durable_file_workflow/verified.py" in examples_index
    assert "`cayu guide verified-work`" in examples_index


def test_package_shipped_app_ui_guide_states_the_request_rules(capsys) -> None:
    assert main(["guide", "app-ui", "--json"]) == 0
    guide = json.loads(capsys.readouterr().out)
    assert guide["package_source"] == "cayu.guides/app-ui.md"
    assert guide["related_topics"] == ["anatomy", "human-attention", "references"]
    content = " ".join(guide["content"].split())
    for rule in (
        'import { connect } from "/cayu/client.js"',
        "Do not hand-write polling loops",
        "only while the tab is visible",
        "Poll only while something is active",
        "no more often than every 15 seconds",
        "Back off while responses do not change",
        "## Split overview from detail",
        "## Keep GET handlers read-only",
        "cayu guide human-attention",
        "GET /api/sessions/{id}/usage",
        "POST /api/sessions/{id}/cost",
        "under 1% of a 0.5 vCPU web process",
        "cayu diagnostics requests --budget-idle-cpu 0.01 --vcpu 0.5",
        "## Example: a progress panel",
    ):
        assert rule in content, rule

    root = Path(__file__).resolve().parents[2]
    assert "cayu guide app-ui" in (root / "src/cayu/guides/application-anatomy.md").read_text()
    control_plane = (root / "docs/control-plane.md").read_text()
    bring_your_own_ui = control_plane.split("## Bring your own UI", 1)[1].split("\n## ", 1)[0]
    assert "../src/cayu/guides/app-ui.md" in bring_your_own_ui


def test_cost_control_reference_states_spend_cap_behavior(capsys) -> None:
    assert main(["guide", "references#cost-control"]) == 0
    content = " ".join(capsys.readouterr().out.split())
    for rule in (
        "budget_limits=(BudgetLimit.usd(5),)",
        "priced by `default_price_book()`",
        "default_price_book().with_prices(ModelPrice.fixed(...))",
        "refused before dispatch unless `allow_unpriced=True`",
        'ends `interrupted` with `interruption_type="limit_reached"`',
        "A session cap does not reserve",
        "`budget.reservation_failed`",
        "profile_adoption=...",
        "is a deployment-wide cap",
        "is not a hard cap across processes",
    ):
        assert rule in content, rule


def test_guide_accepts_the_file_name_a_topic_is_published_under(capsys) -> None:
    assert main(["guide", "application-anatomy"]) == 0
    by_file = capsys.readouterr().out
    assert main(["guide", "anatomy"]) == 0

    assert by_file == capsys.readouterr().out


def test_unknown_guide_topic_suggests_topics_and_explains_repository_docs(capsys) -> None:
    assert main(["guide", "evals"]) == 2
    error = capsys.readouterr().err
    assert "Did you mean:" in error
    assert "evals-first" in error

    assert main(["guide", "project-server", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)["error"]
    assert report["code"] == "UNKNOWN_GUIDE_TOPIC"
    assert "docs/project-server.md are not shipped with the package" in report["message"]
    assert isinstance(report["suggestions"], list)


def test_restart_tests_guide_example_runs_across_a_restart(capsys, tmp_path: Path) -> None:
    assert main(["guide", "restart-tests"]) == 0
    guide = capsys.readouterr().out
    example = re.search(r"```python\n(.*?)```", guide, re.DOTALL)
    assert example is not None
    namespace: dict[str, Any] = {}
    exec(compile(example.group(1), "restart-tests", "exec"), namespace)

    namespace["test_refund_survives_restart_before_approval"](tmp_path)
    assert namespace["REFUNDS"] == ["A-1"]

    for reference in re.findall(r"`cayu guide ([a-z0-9#-]+)`", guide):
        assert main(["guide", reference]) == 0, reference
        capsys.readouterr()
    assert main(["guide", "authoring"]) == 0
    assert "cayu guide restart-tests" in capsys.readouterr().out


def test_long_running_guide_snippets_run_together(capsys, tmp_path: Path) -> None:
    from datetime import timedelta

    from cayu.messages import Message
    from cayu.providers import ModelStreamEvent
    from cayu.sessions import PendingActionKind
    from cayu.sessions.requests import RunRequest
    from cayu.testing import RestartHarness

    assert main(["guide", "long-running"]) == 0
    guide = capsys.readouterr().out
    assert len(guide.splitlines()) <= 230
    snippets = re.findall(r"```python\n(.*?)```", guide, re.DOTALL)
    assert len(snippets) == 6
    namespace: dict[str, Any] = {}
    for snippet in snippets:
        exec(compile(snippet, "long-running", "exec"), namespace)

    # Approval and restart: the guide's own restart test.
    namespace["test_payment_survives_restart"](tmp_path / "approval")
    assert list(namespace["PAID"].values()) == ["INV-7"]

    # Spend cap: the request carries the session limit priced by the default book.
    [limit] = namespace["request"].budget_limits
    assert limit.scope == "session"
    assert limit.max_estimated_cost == 5

    # Outside reply and timer: pause on ask_user, build the reminder, restart, reply.
    ask = ModelStreamEvent.tool_call(name="ask_user", arguments={"question": "Amount?"})
    script = [
        [ask, ModelStreamEvent.completed()],
        [ModelStreamEvent.text_delta("Done."), ModelStreamEvent.completed()],
    ]

    async def scenario() -> None:
        async with RestartHarness(
            namespace["build_app"], store_dir=tmp_path / "reply", model_script=script
        ) as harness:
            paused = await harness.run(
                RunRequest(
                    agent_name="invoices",
                    session_id="inv-8",
                    messages=[Message.text("user", "Pay INV-8.")],
                )
            )
            [question] = paused.pending_user_inputs
            assert question.kind is PendingActionKind.USER_INPUT
            created = await harness.app.create_task(namespace["reminder_task"](question))
            assert created.available_at == question.event.event.timestamp + timedelta(hours=24)
            again = await harness.app.create_task(namespace["reminder_task"](question))
            assert again.id == created.id

            await harness.restart()
            assert await harness.app.task_store.load_task(created.id) is not None
            await namespace["record_reply"](harness.app, "inv-8", "$125.00")
            session = await harness.app.session_store.load("inv-8")
            assert session is not None and session.status.value == "completed"

    asyncio.run(scenario())

    # A worker pause retains one session and hands continuation to its owner.
    async def paused_worker(tool_name, arguments):
        from cayu.evals.testing import ScriptedModelProvider
        from cayu.sessions.memory import InMemorySessionStore
        from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, run_task_worker

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name=tool_name, arguments=arguments),
                    ModelStreamEvent.completed(),
                ],
            ]
        )
        app = namespace["build_app"](
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )
        task = await app.task_store.create_task(
            TaskCreate(type="invoice-job", input={"request": "Pay INV-10."})
        )
        await asyncio.wait_for(
            run_task_worker(
                app,
                app.task_store,
                namespace["handle_job"],
                worker_id="pause-worker",
                max_tasks=1,
                execution_timeout_s=60,
                execution_grace_s=1,
            ),
            15,
        )
        settled = await app.task_store.load_task(task.id)
        assert settled.status is TaskStatus.COMPLETED
        session_id = settled.result["paused_session"]
        session = await app.session_store.load(session_id)
        assert session.status.value == "interrupted"
        assert task.id not in namespace["DELIVERED"]
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        assert len(pending.actions) == 1

    from cayu.sessions import PendingActionQuery

    asyncio.run(paused_worker("pay_invoice", {"invoice_id": "INV-10"}))
    asyncio.run(paused_worker("ask_user", {"question": "Amount?"}))

    # Worker time limit: the guide's retrying worker ends a hung job and fails it.
    async def hung_worker() -> None:
        from cayu.deadlines import ExecutionDeadline, current_handler_time_limit
        from cayu.evals.testing import ScriptedModelProvider
        from cayu.sessions.memory import InMemorySessionStore
        from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus

        class StalledProvider(ScriptedModelProvider):
            async def stream(self, request):
                # Trigger the deadline after the run starts, independent of setup load.
                limit = current_handler_time_limit()
                assert limit is not None and limit._timer is not None
                limit._deadline = ExecutionDeadline.after(0, scope="task_handler")
                limit._cancelled.set()
                limit._timer.reschedule(asyncio.get_running_loop().time())
                await asyncio.Event().wait()
                yield ModelStreamEvent.completed()

        app = namespace["build_app"](
            provider=StalledProvider([]),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )
        task = await app.task_store.create_task(
            TaskCreate(type="invoice-job", input={"request": "Pay INV-9."})
        )
        namespace["JOB_TIMEOUT_S"] = 60
        namespace["JOB_GRACE_S"] = 0.2
        stop = asyncio.Event()
        worker = asyncio.create_task(namespace["run_jobs"](app, stop))
        async with asyncio.timeout(15):
            while (await app.task_store.load_task(task.id)).status is not TaskStatus.FAILED:
                await asyncio.sleep(0.05)
        stop.set()
        await worker
        failed = await app.task_store.load_task(task.id)
        assert failed.error["error"] == "ExecutionDeadlineExceeded"
        assert failed.error["scope"] == "task_handler"
        assert failed.error["handler_cancelled"] is True
        assert namespace["DELIVERED"] == {}
        [session_id] = failed.error["session_ids"]
        events = await app.session_store.load_events(session_id)
        assert events[-1].type.value == "session.interrupted"

    asyncio.run(hung_worker())

    for reference in re.findall(r"`cayu guide ([a-z0-9#-]+)`", guide):
        assert main(["guide", reference]) == 0, reference
        capsys.readouterr()
    assert main(["guide", "authoring#cayu-map"]) == 0
    cayu_map = capsys.readouterr().out
    for phrase in (
        "Remind after N hours",
        "Cap spending",
        "Wait for an email reply",
        "Stop a hung background job",
    ):
        assert phrase in cayu_map


def test_external_agents_guide_example_records_a_scrubbed_priced_session(
    capsys, tmp_path, monkeypatch
) -> None:
    import importlib

    from cayu.budgets import default_price_book
    from cayu.external_agents import ExternalAgentEvent
    from cayu.tasks import TaskStatus

    assert main(["guide", "external-agents"]) == 0
    guide = capsys.readouterr().out
    opening, rest = guide.split("## Launchers", 1)
    local, production = opening.split("## Production setup", 1)
    launchers = rest.split("## Reference", 1)[0]
    for extra in ("cayu[server,claude-agent-sdk]", "cayu[server,codex]"):
        assert extra in production
    for name in ("local_agent_launch", "docker_agent_launch", "ExternalAgentLaunch", "Docker"):
        assert name in launchers
    example, codex = re.findall(r"```python\n(.*?)```", production, re.DOTALL)
    _, local_codex = re.findall(r"```python\n(.*?)```", local, re.DOTALL)

    # The Codex snippets are not executed, so check their imports.
    for snippet in (codex, local_codex):
        compile(snippet, "external-agents", "exec")
        for module, names in re.findall(r"^from (cayu[\w.]*) import (.+)$", snippet, re.M):
            for name in names.split(","):
                assert hasattr(importlib.import_module(module), name.strip()), name

    namespace: dict[str, Any] = {"__name__": "external_agents_guide"}
    exec(compile(example, "external-agents", "exec"), namespace)
    secret = "sk-guide-example-secret"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_ALIAS_KEY", "a" * 42 + "A")

    class ScriptedClaude:
        name = "scripted-claude"
        provider_name = "anthropic"
        model = namespace["MODEL"]

        async def stream(self, context):
            assert context.execution_deadline.expires_at is not None
            assert await context.allow_tool("Read", {"file_path": "app.py"}, "read-1")
            assert not await context.allow_tool("Bash", {"command": "rm -rf ."}, "bash-1")
            yield ExternalAgentEvent(
                "usage", {"input_tokens": 1000, "output_tokens": 200}, "step-1", self.model
            )
            yield ExternalAgentEvent("text", {"text": f"Reading with {secret}."}, "text-1")
            yield ExternalAgentEvent("result", {"output": f"Three problems. {secret}"}, "done")

    async def scenario() -> None:
        app = namespace["build_app"](secret)
        try:
            result = await namespace["review"](app, ScriptedClaude())
            assert result.output.startswith("Three problems.")
            assert secret not in result.output
            events = await app.session_store.load_events(result.session_id)
            stored = json.dumps([event.model_dump(mode="json") for event in events])
            assert secret not in stored
            task = await app.task_store.load_task(result.task_id)
            assert task is not None and task.status is TaskStatus.COMPLETED
            cost = await app.get_session_cost(result.session_id, default_price_book())
            assert cost.priced_model_steps == 1
            assert cost.total_cost > 0
        finally:
            await app.aclose()

    asyncio.run(scenario())
    printed = capsys.readouterr().out
    assert "1000 input and 200 output tokens" in printed
    assert secret not in printed

    for reference in re.findall(r"`cayu guide ([a-z0-9#-]+)`", guide):
        assert main(["guide", reference]) == 0, reference
        capsys.readouterr()


def test_external_agents_guide_local_example_runs_without_docker_or_a_gateway(
    capsys, tmp_path, monkeypatch
) -> None:
    """The quick start runs with the installed SDK, a stand-in `claude` and no gateway."""
    if os.name != "posix":
        pytest.skip("local_agent_launch is POSIX only")
    pytest.importorskip("claude_agent_sdk")
    from tests.core.test_external_local_launch import _FAKE_CLAUDE

    assert main(["guide", "external-agents"]) == 0
    guide = capsys.readouterr().out
    needs, local = guide.split("## Production setup", 1)[0].split("## Quick start", 1)
    assert "Optional:" in needs and "Without the gateway:" in needs
    assert "not a sandbox" in local
    example = re.findall(r"```python\n(.*?)```", local, re.DOTALL)[0]
    assert "Gateway" not in example and "docker" not in example.lower()

    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    claude = bin_directory / "claude"
    claude.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE}", encoding="utf-8")
    claude.chmod(0o700)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("PATH", f"{bin_directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(checkout)
    namespace: dict[str, Any] = {"__name__": "external_agents_local_guide"}
    exec(compile(example, "external-agents", "exec"), namespace)
    asyncio.run(namespace["main"]())
    printed = capsys.readouterr().out
    assert "No problems found." in printed
    assert "120 input and 30 output tokens" in printed
    assert (checkout / "sessions.db").exists()


def test_external_agents_guide_is_indexed_and_its_extras_are_listed(capsys) -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for extra in ("claude-agent-sdk", "codex"):
        assert extra in extras
        assert f"| `cayu[{extra}]` |" in readme
    assert "`cayu guide external-agents`" in readme
    index = (root / "docs" / "README.md").read_text(encoding="utf-8")
    assert "(../src/cayu/guides/external-agents.md) (`cayu guide external-agents`)" in index
    pointer = (root / "docs" / "external-agents.md").read_text(encoding="utf-8")
    assert "cayu guide external-agents" in pointer
    assert len(pointer.splitlines()) < 10

    for topic in ("authoring#cayu-map", "references#external-agents"):
        assert main(["guide", topic]) == 0
        assert "cayu guide external-agents" in capsys.readouterr().out

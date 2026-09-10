from __future__ import annotations

import asyncio

import pytest
from tests.core.test_mcp import _fake_tool_definitions, _fake_toolset
from tests.core.test_tool_effect_receipts import _receipt

from cayu import AgentSpec, CayuApp, ExecutionProfileBehaviorIdentity, Tool, ToolEffect, ToolSpec
from cayu.runtime._tool_effect_reconciliation import (
    AcceptedToolEffectReconciliation,
    project_accepted_reconciliation,
    validate_reconciliation_result,
)
from cayu.runtime.tool_effects import (
    ToolEffectReconcilerSpec,
    ToolEffectReconciliationContext,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationResult,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _Deployment(Tool):
    spec = ToolSpec(name="deploy", effect=ToolEffect.EXTERNAL)

    async def run(self, ctx, args):
        pytest.fail("Registration must not invoke a tool.")


class _Reconciler:
    async def reconcile(self, *, context, receipt):
        return ToolEffectReconciliationResult(outcome="not_found", observation="outcome_unknown")


def _spec(**changes):
    values = {
        "execution_profile_identity": ExecutionProfileBehaviorIdentity(
            name="deployment-reconciler",
            behavior_version="1",
            implementation_version="1",
        ),
        "receipt_schema": "deployment",
        "receipt_schema_version": 1,
        "required": True,
        "result_schema": {
            "type": "object",
            "properties": {"version": {"type": "integer"}},
            "additionalProperties": False,
        },
        "integrity_fields": ("signature",),
        "resource_version_fields": ("part-1", "part-2"),
    }
    return ToolEffectReconcilerSpec(**(values | changes))


def _register(registration):
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="agent", model="test"),
        tools=[_Deployment()],
        tool_effect_reconcilers={"deploy": registration},
    )
    return app.get_agent("agent").tools["deploy"].effect_reconciler


def test_registration_captures_executable_and_detached_contract():
    reconciler = _Reconciler()
    spec = _spec()
    registration = ToolEffectReconciliationRegistration(reconciler=reconciler, spec=spec)
    registered = _register(registration)
    before = registered.material()
    spec.result_schema["properties"].clear()
    registration.reconciler = None
    registration.spec = None
    assert registered.material() == before
    assert registered.operation.__self__ is reconciler
    exported = registered.material()
    exported["result_schema"]["properties"].clear()
    assert registered.material() == before
    result = asyncio.run(registered.operation(context=None, receipt=None))
    assert result.outcome == "not_found"


def test_resource_allowlist_is_canonical_and_bound_to_registration():
    def register(fields):
        return _register(
            ToolEffectReconciliationRegistration(
                reconciler=_Reconciler(),
                spec=_spec(resource_version_fields=fields),
            )
        )

    forward = register(("part-1", "part-2"))
    reverse = register(("part-2", "part-1"))
    restricted = register(("part-1",))
    assert forward.material() == reverse.material()
    assert forward.fingerprint == reverse.fingerprint
    assert forward.fingerprint != restricted.fingerprint


@pytest.mark.parametrize("reconciler", [None, object(), lambda: None])
def test_required_registration_rejects_missing_executable(reconciler):
    with pytest.raises(TypeError, match="actual async"):
        _register(ToolEffectReconciliationRegistration(reconciler=reconciler, spec=_spec()))


def test_registration_rejects_wrong_callback_signature():
    class Wrong:
        async def reconcile(self, different_argument):
            pass

    with pytest.raises(TypeError, match="incompatible signature"):
        _register(ToolEffectReconciliationRegistration(reconciler=Wrong(), spec=_spec()))


def test_registration_cannot_expand_receipt_eligibility_to_idempotent_tools():
    class Idempotent(_Deployment):
        spec = ToolSpec(name="deploy", effect=ToolEffect.IDEMPOTENT)

    app = CayuApp(enable_logging=False)
    with pytest.raises(ValueError, match="only to external"):
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            tools=[Idempotent()],
            tool_effect_reconcilers={
                "deploy": ToolEffectReconciliationRegistration(
                    reconciler=_Reconciler(), spec=_spec()
                )
            },
        )
    # Failed registration does not partially install the agent.
    app.register_agent(AgentSpec(name="agent", model="test"), tools=[Idempotent()])


@pytest.mark.parametrize(
    "changes",
    [
        {"receipt_schema_version": True},
        {"timeout_seconds": True},
        {"timeout_seconds": 301},
        {"required": 1},
        {"result_schema": {}},
        {"integrity_fields": ("signature", "signature")},
        {"resource_version_fields": ("part", "part")},
        {"resource_version_fields": tuple(f"part-{i}" for i in range(33))},
        {
            "result_schema": {
                "type": "object",
                "additionalProperties": False,
                "$ref": "https://example.test/schema",
            }
        },
    ],
)
def test_registration_spec_rejects_ambiguous_or_unbounded_contract(changes):
    with pytest.raises(ValueError):
        _spec(**changes)


def test_callback_result_must_match_exact_identity_schema_and_integrity_allowlist():
    registered = _register(
        ToolEffectReconciliationRegistration(reconciler=_Reconciler(), spec=_spec())
    )
    context = ToolEffectReconciliationContext(
        session_id="session",
        session_instance_id="instance",
        tool_round_id="round",
        tool_call_id="call-1",
        tool_name="deploy",
        idempotency_key="key-1",
        intent_digest="a" * 64,
        arguments_digest="b" * 64,
        record_revision=2,
    )
    receipt = _receipt(structured={"version": 1}, integrity={"signature": "verified"})
    result = ToolEffectReconciliationResult(
        outcome="completed", observation="sent", receipt=receipt
    )
    assert validate_reconciliation_result(result, registered=registered, context=context) == result
    for changes in (
        {"tool_call_id": "wrong"},
        {"tool_name": "wrong"},
        {"idempotency_key": "wrong"},
        {"receipt_schema": "wrong"},
        {"receipt_schema_version": 2},
        {"integrity": {"headers": "not allowed"}},
        {"resource_versions": {"raw_response": "not allowed"}},
        {"structured": {"raw_response": "not allowed"}},
    ):
        wrong = ToolEffectReconciliationResult(
            outcome="completed", observation="sent", receipt=receipt.model_copy(update=changes)
        )
        with pytest.raises(ValueError):
            validate_reconciliation_result(wrong, registered=registered, context=context)
    for outcome in ("completed", "not_found", "unsupported", "conflict"):
        wrong = ToolEffectReconciliationResult(
            outcome=outcome,
            observation="sent" if outcome == "completed" else "partial",
            receipt=receipt if outcome == "completed" else None,
            resource_versions={"raw_response": "not allowed"},
        )
        with pytest.raises(ValueError, match="registered allow-list"):
            validate_reconciliation_result(wrong, registered=registered, context=context)
    for outcome in ("completed", "not_found"):
        accepted = AcceptedToolEffectReconciliation(
            context=context,
            request_digest="c" * 64,
            result=ToolEffectReconciliationResult(
                outcome=outcome,
                observation="sent" if outcome == "completed" else "partial",
                receipt=receipt.model_copy(update={"resource_versions": {"part-2": "v2"}})
                if outcome == "completed"
                else None,
                resource_versions={"part-1": "private-resource-value"},
            ),
        )
        projected = project_accepted_reconciliation(
            accepted, registered=registered, redactor=SecretRedactor("private-resource-value")
        )
        assert projected.resource_versions == {"part-1": REDACTED_SECRET}
        if projected.receipt is not None:
            assert projected.receipt.resource_versions == {
                "part-1": REDACTED_SECRET,
                "part-2": "v2",
            }
        assert "private-resource-value" not in projected.model_dump_json()
        # Redaction is not permission to persist a key outside the registered contract.
        with pytest.raises(ValueError, match="registered allow-list"):
            project_accepted_reconciliation(
                accepted, registered=registered, redactor=SecretRedactor("part-1")
            )


def test_reconciler_contract_participates_in_public_profile_inspection():
    from cayu import Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
    from cayu.runtime.execution_profiles import ExecutionProfileComponentClass

    class VersionedTool(_Deployment):
        spec = ToolSpec(
            name="deploy",
            effect=ToolEffect.EXTERNAL,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="deployment-tool",
                behavior_version="1",
                implementation_version="1",
            ),
        )

    async def fingerprint(version):
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            tools=[VersionedTool()],
            tool_effect_reconcilers={
                "deploy": ToolEffectReconciliationRegistration(
                    reconciler=_Reconciler(),
                    spec=_spec(receipt_schema_version=version),
                )
            },
        )
        configuration = await app.inspect_effective_run_configuration(
            RunRequest(agent_name="agent", messages=[Message.text("user", "inspect")]),
        )
        return next(
            component.fingerprint
            for component in configuration.execution_profile.components
            if component.component_class == ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS
        )

    assert asyncio.run(fingerprint(1)) == asyncio.run(fingerprint(1))
    assert asyncio.run(fingerprint(1)) != asyncio.run(fingerprint(2))


def test_public_mcp_refresh_preserves_reconciler_and_rejects_silent_removal():
    async def scenario():
        toolset = _fake_toolset()
        app = CayuApp(enable_logging=False)
        name = "mcp__local-mcp__echo"
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            mcp_toolsets=(toolset,),
            tool_effect_reconcilers={
                name: ToolEffectReconciliationRegistration(reconciler=_Reconciler(), spec=_spec())
            },
        )
        before = app.get_agent("agent").tools[name].effect_reconciler
        toolset.session.definitions = _fake_tool_definitions("echo", "search")
        accepted = await app.refresh_mcp_toolset(toolset)
        assert accepted.status == "accepted"
        assert app.get_agent("agent").tools[name].effect_reconciler is before
        toolset.session.definitions = _fake_tool_definitions("search")
        with pytest.raises(ValueError, match="registered effect reconciler"):
            await app.refresh_mcp_toolset(toolset)
        assert app.get_agent("agent").tools[name].effect_reconciler is before
        await toolset.close()

    asyncio.run(scenario())

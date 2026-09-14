from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core.test_tool_execution import _ScriptedProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope
from cayu.events import Event, EventType
from cayu.failure_evidence import FailureEvidence
from cayu.messages import Message
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._event_projection import project_runtime_event
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.EXTERNAL])
def test_native_tool_deadline_diagnostic_redacts_public_and_durable_evidence(
    backend, effect, sqlite_resources
):
    secret = "tool_deadline_secret_canary"

    class DeadlineTool(Tool):
        spec = ToolSpec(name="probe", effect=effect)

        async def run(self, ctx, args):
            async with execution_deadline_scope(
                ExecutionDeadline.after(0.01, source=secret, scope=secret)
            ):
                await asyncio.Event().wait()

    async def scenario():
        async with sqlite_resources as resources:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else SQLiteSessionStore(
                    resources.path(),
                    public_authority_alias_codec=PublicAuthorityAliasCodec(
                        PublicAuthorityAliasKeyring(
                            active_key_id="test", keys={"test": SecretStr("A" * 43)}
                        )
                    ),
                )
            )
            if isinstance(store, SQLiteSessionStore):
                resources.own(store)
            app = CayuApp(
                enable_logging=False, session_store=store, secret_redactor=SecretRedactor(secret)
            )
            app.register_provider(_ScriptedProvider([("call1", "probe", {})]), default=True)
            app.register_agent(AgentSpec(name="agent", model="fake"), tools=[DeadlineTool()])
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="probe",
                        agent_name="agent",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]
            durable = await store.load_events("probe")
            for observed in (events, durable):
                assert secret not in json.dumps([event.payload for event in observed])
                diagnostic_event = next(
                    event
                    for event in observed
                    if event.type
                    == (
                        EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
                        if effect is ToolEffect.EXTERNAL
                        else EventType.TOOL_CALL_FAILED
                    )
                )
                evidence = FailureEvidence.model_validate(
                    diagnostic_event.payload["failure_evidence"]
                )
                assert evidence.classification == "deadline"
                assert evidence.deadline is not None
                assert evidence.deadline.source == evidence.deadline.scope == "redacted"
                assert evidence.exception_types == ("TimeoutError", "CancelledError")
                assert evidence.settlement == "unknown"
            assert events[-1].type == (
                EventType.SESSION_INTERRUPTED
                if effect is ToolEffect.EXTERNAL
                else EventType.SESSION_COMPLETED
            )

    asyncio.run(scenario())


def test_variable_failure_evidence_is_redacted_before_control_restoration():
    secret = "DiagnosticCanary"
    snapshot = FailureEvidence(
        classification="deadline",
        deadline=ExecutionDeadline(
            expires_at=datetime(2030, 1, 1, tzinfo=UTC), source=secret, scope=secret
        ),
        deadline_phase="in_flight",
        exception_types=(secret, "ValueError"),
        session_id=secret,
        terminal_event_id=secret,
        run_epoch=3,
    ).model_dump(mode="json")
    evidence = FailureEvidence.model_validate({**snapshot, "branch_failures": [snapshot]})
    controls = {
        "terminal_outcome": "tool_execution_error",
        "tool_effect": "external",
        "outcome_unknown": True,
        "manual_reconciliation_required": True,
        "failure_evidence": evidence.model_dump(mode="json"),
    }
    result = ToolResult(content="failed", is_error=True, structured=controls)
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="probe",
        payload={**controls, "result": result.model_dump(mode="json")},
    )
    redactor = SecretRedactor([secret, "failure", "deadline", "unknown", "classification"])
    sanitized, result = tool_results.redact_tool_result_event(
        event=event, result=result, redactor=redactor
    )
    public = project_runtime_event(sanitized, sequence=1, redactor=redactor)
    replayed = project_runtime_event(event, sequence=1, redactor=redactor)
    [outcome] = tool_results.redact_runtime_owned_tool_call_outcomes(
        [
            runtime_records.ToolCallOutcome(
                call=runtime_records.ToolCallRequest(id="call", name="probe", arguments={}),
                result=ToolResult(content="failed", is_error=True, structured=controls),
            )
        ],
        redactor,
    )
    for payload in (
        sanitized.payload,
        result.model_dump(mode="json")["structured"],
        public.payload,
        replayed.payload,
        outcome.result.model_dump(mode="json")["structured"],
    ):
        assert secret not in json.dumps(payload)
        actual = FailureEvidence.model_validate(payload["failure_evidence"])
        for branch in (actual, *actual.branch_failures):
            assert branch.classification == "deadline"
            assert branch.deadline is not None
            assert branch.deadline.source == branch.deadline.scope == "redacted"
            assert branch.exception_types == ("ValueError",)
            assert branch.session_id is branch.terminal_event_id is None
            assert branch.truncated is True
            assert branch.settlement == "unknown"


def test_secret_expiry_is_omitted_without_invalidating_failure_evidence():
    evidence = FailureEvidence(
        classification="deadline",
        deadline=ExecutionDeadline(expires_at=datetime(2030, 1, 1, tzinfo=UTC)),
        deadline_phase="in_flight",
    )
    controls = tool_results.runtime_terminal_controls(
        {
            "terminal_outcome": "tool_execution_error",
            "tool_effect": "none",
            "outcome_unknown": False,
            "manual_reconciliation_required": False,
            "failure_evidence": evidence.model_dump(mode="json"),
        },
        redactor=SecretRedactor("2030-01-01"),
    )
    actual = FailureEvidence.model_validate(controls["failure_evidence"])
    assert actual.classification == "unknown"
    assert actual.deadline is actual.deadline_phase is None
    assert actual.truncated is True
    assert "2030-01-01" not in json.dumps(controls)

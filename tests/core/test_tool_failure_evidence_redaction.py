from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core.test_tool_execution import _ScriptedProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope
from cayu.events import Event, EventType
from cayu.failure_evidence import FAILURE_EVIDENCE_MESSAGE_MAX_CHARS, FailureEvidence
from cayu.messages import Message
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._event_projection import project_runtime_event
from cayu.runtime._invocation_secrets import InvocationPublicationSnapshot
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import _terminal_controls as tool_terminal_controls
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


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
                assert evidence.exception_types == ("ExecutionDeadlineExceeded", "CancelledError")
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
    controls = tool_terminal_controls.runtime_terminal_controls(
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


@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.EXTERNAL])
def test_tool_exception_keeps_masked_message_names_call_and_logs_traceback(effect, caplog):
    secret = "tool_exception_secret_canary"

    class BuggyTool(Tool):
        spec = ToolSpec(name="probe", effect=effect)

        async def run(self, ctx, args):
            raise AttributeError(f"'Task' object has no attribute 'task_id' ({secret})")

    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(
            enable_logging=False, session_store=store, secret_redactor=SecretRedactor(secret)
        )
        app.register_provider(_ScriptedProvider([("call1", "probe", {})]), default=True)
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[BuggyTool()])
        live = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="buggy",
                    agent_name="agent",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        return live, await store.load_events("buggy")

    with caplog.at_level(logging.ERROR, logger="cayu.runtime._tool_execution"):
        live, durable = asyncio.run(scenario())

    for observed in (live, durable):
        assert secret not in json.dumps([event.payload for event in observed])
    diagnostic_type = (
        EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        if effect is ToolEffect.EXTERNAL
        else EventType.TOOL_CALL_FAILED
    )
    diagnostic = next(event for event in durable if event.type == diagnostic_type)
    evidence = FailureEvidence.model_validate(diagnostic.payload["failure_evidence"])
    assert evidence.exception_types == ("AttributeError",)
    assert evidence.exception_message == (
        f"'Task' object has no attribute 'task_id' ({REDACTED_SECRET})"
    )
    if effect is ToolEffect.EXTERNAL:
        interrupted = durable[-1]
        assert interrupted.type == EventType.SESSION_INTERRUPTED
        assert interrupted.payload["interruption_type"] == "runtime_interrupted"
        assert interrupted.payload["tool_name"] == "probe"
        assert interrupted.payload["tool_call_id"] == "call1"
    else:
        assert durable[-1].type == EventType.SESSION_COMPLETED

    [record] = [
        record for record in caplog.records if record.name == "cayu.runtime._tool_execution"
    ]
    assert record.levelno == logging.ERROR
    assert record.exc_info is None
    logged = record.getMessage()
    assert "Tool probe raised in session buggy" in logged
    assert "Traceback (most recent call last):" in logged
    assert f"AttributeError: 'Task' object has no attribute 'task_id' ({REDACTED_SECRET})" in (
        logged
    )
    assert secret not in logged


def test_failure_evidence_message_is_optional_bounded_and_masked():
    legacy = FailureEvidence.model_validate(
        {"classification": "failure", "exception_types": ["ValueError"]}
    )
    assert legacy.exception_message is None
    assert "exception_message" not in legacy.model_dump(mode="json")

    oversized = FailureEvidence(exception_message="x" * (FAILURE_EVIDENCE_MESSAGE_MAX_CHARS * 2))
    assert oversized.exception_message == "x" * FAILURE_EVIDENCE_MESSAGE_MAX_CHARS

    secret = "s3"
    snapshot = FailureEvidence(
        classification="failure",
        exception_types=("ValueError",),
        exception_message="bad " + secret * (FAILURE_EVIDENCE_MESSAGE_MAX_CHARS // 2 - 2),
    ).model_dump(mode="json")
    controls = tool_terminal_controls.runtime_terminal_controls(
        {
            "terminal_outcome": "tool_execution_error",
            "tool_effect": "external",
            "outcome_unknown": True,
            "manual_reconciliation_required": True,
            "failure_evidence": {**snapshot, "branch_failures": [snapshot]},
        },
        redactor=SecretRedactor(secret),
    )
    # Masking a short secret lengthens the text; it is bounded, not rejected.
    actual = FailureEvidence.model_validate(controls["failure_evidence"])
    for branch in (actual, *actual.branch_failures):
        assert branch.exception_message is not None
        assert branch.exception_message.startswith("bad " + REDACTED_SECRET)
        assert len(branch.exception_message.encode()) <= FAILURE_EVIDENCE_MESSAGE_MAX_CHARS
        assert secret not in branch.exception_message


def _run_raising_tool(message: str, snapshot: InvocationPublicationSnapshot):
    class RaisingTool(Tool):
        spec = ToolSpec(name="probe", effect=ToolEffect.EXTERNAL)

        async def run(self, ctx, args):
            raise ValueError(message)

    return asyncio.run(
        tool_execution.run_tool(
            tool=RaisingTool(),
            effect=ToolEffect.EXTERNAL,
            ctx=ToolContext(session_id="late-scope", idempotency_key="call-key"),
            arguments={},
            # The live scope at raise time does not know the secret yet.
            redactor=SecretRedactor,
            finalize_publication=lambda: snapshot,
        )
    )


def test_tool_exception_text_is_masked_with_the_final_secret_scope(caplog):
    secret = "late-registered-secret-canary"
    with caplog.at_level(logging.ERROR, logger="cayu.runtime._tool_execution"):
        outcome = _run_raising_tool(
            f"Invalid file contents: {secret}",
            InvocationPublicationSnapshot(redactor=SecretRedactor(secret), unsafe_output=False),
        )

    assert secret not in caplog.text
    assert f"ValueError: Invalid file contents: {REDACTED_SECRET}" in caplog.text
    controls = outcome.terminal_payload_fields()
    assert controls["failure_evidence"]["exception_message"] == (
        f"Invalid file contents: {REDACTED_SECRET}"
    )
    assert secret not in json.dumps(controls)
    # Result content is masked again at publication; the restored controls
    # in the structured result already carry the final-scope message.
    structured = outcome.result.model_dump(mode="json")["structured"]
    assert structured["failure_evidence"] == controls["failure_evidence"]


def test_tool_exception_text_is_not_logged_when_the_secret_scope_is_unsafe(caplog):
    secret = "late-workspace-secret-canary"
    prefix = secret[:16]
    with caplog.at_level(logging.ERROR, logger="cayu.runtime._tool_execution"):
        outcome = _run_raising_tool(
            f"Invalid file contents: {prefix}",
            InvocationPublicationSnapshot(redactor=SecretRedactor(secret), unsafe_output=True),
        )

    # A captured prefix cannot be matched by the redactor; nothing of the
    # exception's text or traceback may leave the process.
    assert prefix not in caplog.text
    assert "Invalid file contents" not in caplog.text
    assert "Traceback" not in caplog.text
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "Tool probe raised in session late-scope" in record.getMessage()
    assert "could not be finalized safely" in record.getMessage()
    published = json.dumps(
        [outcome.terminal_payload_fields(), outcome.result.model_dump(mode="json")]
    )
    assert prefix not in published
    assert "exception_message" not in published


def _nested_group(message: str) -> ExceptionGroup:
    leaf = ValueError(message)
    leaf.add_note(message)
    return ExceptionGroup(
        "request failed",
        [ExceptionGroup("inner failure", [leaf, KeyError("other")]), RuntimeError("sibling")],
    )


@pytest.mark.parametrize(
    "secret",
    [
        "credential-first-line\ncredential-second-line\n  credential-third-line",
        "single-line-group-credential",
    ],
    ids=["multiline", "single_line"],
)
def test_nested_group_traceback_log_keeps_no_secret_fragment(secret, caplog):
    class GroupTool(Tool):
        spec = ToolSpec(name="probe", effect=ToolEffect.EXTERNAL)

        async def run(self, ctx, args):
            raise _nested_group(f"request rejected: {secret}")

    with caplog.at_level(logging.ERROR, logger="cayu.runtime._tool_execution"):
        outcome = asyncio.run(
            tool_execution.run_tool(
                tool=GroupTool(),
                effect=ToolEffect.EXTERNAL,
                ctx=ToolContext(session_id="group-scope"),
                arguments={},
                redactor=lambda: SecretRedactor(secret),
                finalize_publication=lambda: InvocationPublicationSnapshot(
                    redactor=SecretRedactor(secret), unsafe_output=False
                ),
            )
        )

    [record] = caplog.records
    logged = record.getMessage()
    # The traceback still shows the nested group structure and its leaves.
    assert "| ValueError: request rejected: " in logged
    assert "| KeyError: 'other'" in logged
    assert REDACTED_SECRET in logged
    for fragment in (line.strip() for line in secret.splitlines()):
        assert fragment not in logged
    message = outcome.terminal_payload_fields()["failure_evidence"]["exception_message"]
    for fragment in (line.strip() for line in secret.splitlines()):
        assert fragment not in message


def test_line_forms_match_each_line_of_multiline_secrets_only():
    redactor = SecretRedactor(["alpha-line\n  beta-line\n\n", "single-value"])
    lined = redactor.with_line_forms()
    assert lined.redact_text("x alpha-line y | beta-line z") == (
        f"x {REDACTED_SECRET} y | {REDACTED_SECRET} z"
    )
    assert lined.redact_text("single-value") == REDACTED_SECRET
    assert lined.redact_text("single") == "single"
    assert (
        SecretRedactor("one line").with_line_forms().has_same_registry(SecretRedactor("one line"))
    )

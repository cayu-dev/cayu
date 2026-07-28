from __future__ import annotations

import asyncio

import pytest
from tests.core._workload_secret_support import (
    FakeProvider,
    RequireApprovalPolicy,
    SideEffectTool,
    collect_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
)

from cayu.core import AgentSpec, EventType, Message
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunRequest,
    SessionStatus,
    StructuredOutputSpec,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def test_pending_tool_round_recovery_rejects_redaction_marker_arguments() -> None:
    checkpoint, _ = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="legacy_redacted_call",
                name="side_effect",
                arguments={"value": "safe"},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
    )
    checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["value"] = REDACTED_SECRET

    with pytest.raises(ValueError, match="redaction marker"):
        tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)


def test_cayu_app_never_executes_new_tool_call_with_redaction_marker() -> None:
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_new_marker",
                    name="side_effect",
                    arguments={"value": REDACTED_SECRET},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_new_tool_marker",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == []
    assert asyncio.run(store.load_checkpoint("sess_new_tool_marker")) is None


def test_cayu_app_rejects_workload_secret_before_approval_checkpoint() -> None:
    secret = "approval-checkpoint-boundary-canary"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_secret",
                    name="side_effect",
                    arguments={"value": secret},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    tool = SideEffectTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval_redaction"))
    transcript = asyncio.run(store.load_transcript("sess_tool_approval_redaction"))
    assert events[-1].type == EventType.SESSION_FAILED
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert checkpoint is None
    assert tool.calls == []
    assert secret not in str([event.model_dump(mode="json") for event in events])
    assert secret not in str([message.model_dump(mode="json") for message in transcript])


def test_resolve_tool_approval_never_executes_legacy_redaction_marker() -> None:
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_legacy_marker",
                    name="side_effect",
                    arguments={"value": "safe"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupted = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_legacy_approval_marker",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event.payload["approval"]["approval_id"]
        for event in interrupted
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_legacy_approval_marker"))
    assert checkpoint is not None
    checkpoint["pending_tool_approval"]["arguments"]["value"] = REDACTED_SECRET
    asyncio.run(store.checkpoint("sess_legacy_approval_marker", checkpoint))

    with pytest.raises(ValueError, match="redaction marker"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_legacy_approval_marker",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )

    assert tool.calls == []


def _secret_redacting_paused_approval(
    *,
    session_id: str,
    secret: str,
) -> tuple[CayuApp, InMemorySessionStore, FakeProvider, str]:
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "safe"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )
    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]
    return app, store, provider, approval_id


def test_cayu_app_rejects_secret_structured_output_on_tool_approval() -> None:
    secret = "tool-approval-schema-secret-canary"
    session_id = "sess_secret_tool_approval"
    app, store, provider, approval_id = _secret_redacting_paused_approval(
        session_id=session_id,
        secret=secret,
    )

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "string", "const": secret},
                    ),
                ),
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_rejects_secret_structured_output_on_tool_approval_recovery() -> None:
    secret = "tool-approval-recovery-schema-secret-canary"
    session_id = "sess_secret_tool_approval_recovery"
    app, store, provider, approval_id = _secret_redacting_paused_approval(
        session_id=session_id,
        secret=secret,
    )

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id=session_id,
                    approval_id=approval_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect completed externally",
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "string", "const": secret},
                    ),
                ),
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1

"""Incomplete recovery contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.core._session_store_test_doubles import RecordingListSessionsStore
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingAfterPendingToolRoundCheckpointStore,
    FailingOrdinaryToolResultCloseStore,
)
from tests.runtime.runtime_session_support import (
    _private_events_for_public_events,
    _private_record_for_public_event,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_fork_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_round_recovery_events,
    transition_runtime_session_to_running,
)
from tests.runtime.runtime_tool_support import (
    PureRecordingTool,
    RequireApprovalPolicy,
    SideEffectTool,
)

import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.business import BusinessApprovalResolutionState, business_approval_audit
from cayu.approvals.tools import (
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRequest,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHook, ToolCallHookContext
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
)
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.tools.policy import ToolPolicyDecision
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.redaction import SecretRedactor


async def _checkpoint_allow_policy_plan(
    store: InMemorySessionStore,
    session_id: str,
) -> None:
    """Give recovery tests unrelated to policy ambiguity authoritative input."""

    checkpoint = await store.load_checkpoint(session_id)
    if checkpoint is None:
        raise AssertionError("Test session has no checkpoint.")
    payload = dict(checkpoint["pending_tool_round"])
    payload["policy_state"] = "planned"
    payload["policy_context_version"] = 1
    payload["tool_calls"] = [
        {
            **call,
            "policy_evidence": "authoritative",
            "policy_decision": ToolPolicyDecision.ALLOW.value,
            "reason": None,
            "metadata": {},
        }
        for call in payload["tool_calls"]
    ]
    updated = dict(checkpoint)
    updated["pending_tool_round"] = payload
    await store.checkpoint(session_id, updated)


def test_cayu_app_recover_incomplete_session_gates_round_without_policy_outcome():
    store = FailingAfterPendingToolRoundCheckpointStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_incomplete_tool_round",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_tool_round"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None
    asyncio.run(
        rebind_test_invocation(
            store,
            "sess_recover_incomplete_tool_round",
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, current_checkpoint: (
                execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=session.id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch + 1,
                    profile=active_profile.profile,
                    expected=active_profile,
                )
            ),
        )
    )
    provider_requests_before_recovery = len(provider.requests)

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_incomplete_tool_round")
        )
    )
    session = asyncio.run(store.load("sess_recover_incomplete_tool_round"))
    transcript = asyncio.run(store.load_transcript("sess_recover_incomplete_tool_round"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.PENDING_APPROVAL,
    )
    assert tool.calls == []
    assert len(provider.requests) == provider_requests_before_recovery
    assert [message.role for message in transcript] == ["user"]
    recovered_checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_tool_round"))
    assert recovered_checkpoint is not None
    approval = PendingToolApproval.model_validate(recovered_checkpoint["pending_tool_approval"])
    assert approval.tool_call_id == "call_1"
    assert approval.tool_calls[0].policy_evidence == "ambiguous"
    assert approval.tool_calls[0].policy_decision is None
    assert approval.metadata["policy_evaluation"] == "ambiguous"


def test_cayu_app_recover_incomplete_session_restores_required_approval_before_continuing():
    store = FailingAfterPendingToolRoundCheckpointStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "requires approval"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_required_approval_plan",
                messages=[Message.text("user", "use tool")],
                metadata={"review_context": "durable"},
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_required_approval_plan"))
    assert checkpoint is not None
    assert "pending_tool_approval" not in checkpoint
    assert checkpoint["pending_tool_round"]["policy_state"] == "unplanned"
    assert checkpoint["pending_tool_round"]["request_metadata"] == {"review_context": "durable"}
    asyncio.run(
        transition_runtime_session_to_running(
            store,
            "sess_recover_required_approval_plan",
        )
    )

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_required_approval_plan")
        )
    )
    session = asyncio.run(store.load("sess_recover_required_approval_plan"))
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_required_approval_plan"))
    events = asyncio.run(store.load_events("sess_recover_required_approval_plan"))

    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.PENDING_APPROVAL,
    )
    assert checkpoint is not None
    approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
    assert checkpoint["pending_tool_round"]["policy_state"] == "planned"
    assert checkpoint["pending_tool_round"]["tool_round_id"] == approval.tool_round_id
    assert result.pending_approval_id is not None
    assert (
        asyncio.run(
            app._resolve_public_action_linkage(
                session_id="sess_recover_required_approval_plan",
                value=result.pending_approval_id,
                field_name="approval_id",
            )
        )
        == approval.approval_id
    )
    assert any(
        event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        and event.payload["approval_id"] == approval.approval_id
        and event.payload["recovered"] is True
        for event in events
    )
    assert tool.calls == []


def test_legacy_ambiguous_policy_recovery_preserves_authoritative_denial():
    async def run() -> None:
        session_id = "sess_legacy_mixed_policy_recovery"
        store = FailingAfterPendingToolRoundCheckpointStore()
        tool = SideEffectTool()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_denied",
                        name="side_effect",
                        arguments={"value": "must not run"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_ambiguous",
                        name="side_effect",
                        arguments={"value": "human approved"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("resolved"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use both tools")],
            ),
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        legacy_round = dict(checkpoint["pending_tool_round"])
        legacy_calls = [dict(call) for call in legacy_round["tool_calls"]]
        for legacy_call in legacy_calls:
            legacy_call.pop("policy_evidence")
        legacy_calls[0].update(
            policy_decision=ToolPolicyDecision.DENY.value,
            reason="authoritative denial",
            metadata={"source": "legacy"},
        )
        legacy_calls[1].update(policy_decision=None, reason=None, metadata={})
        legacy_round["tool_calls"] = legacy_calls
        legacy_round.pop("policy_state")
        legacy_round.pop("policy_context_version")
        checkpoint = dict(checkpoint)
        checkpoint["pending_tool_round"] = legacy_round
        await store.checkpoint(session_id, checkpoint)
        await transition_runtime_session_to_running(store, session_id)

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
        assert [call.policy_decision for call in approval.tool_calls] == [
            ToolPolicyDecision.DENY.value,
            None,
        ]
        assert [call.policy_evidence for call in approval.tool_calls] == [
            "authoritative",
            "ambiguous",
        ]

        events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert tool.calls == []
        assert events[-1].type is EventType.SESSION_COMPLETED
        blocked = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
        private_blocked = await _private_events_for_public_events(store, session_id, blocked)
        assert [event.payload["tool_call_id"] for event in private_blocked] == [
            "call_denied",
            "call_ambiguous",
        ]
        assert blocked[1].payload["blocked_by"] == "policy_evaluation_ambiguous"

    asyncio.run(run())


def test_legacy_mixed_policy_recovery_prefers_authoritative_approval_gate():
    async def run() -> None:
        session_id = "sess_legacy_mixed_authoritative_approval"
        store = FailingAfterPendingToolRoundCheckpointStore()
        tool = SideEffectTool()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ambiguous",
                        name="side_effect",
                        arguments={"value": "must remain blocked"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_authoritative_approval",
                        name="side_effect",
                        arguments={"value": "explicitly approved"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("resolved"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use both tools")],
            ),
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        legacy_round = dict(checkpoint["pending_tool_round"])
        legacy_calls = [dict(call) for call in legacy_round["tool_calls"]]
        for legacy_call in legacy_calls:
            legacy_call.pop("policy_evidence")
        legacy_calls[0].update(policy_decision=None, reason=None, metadata={})
        legacy_calls[1].update(
            policy_decision=ToolPolicyDecision.REQUIRE_APPROVAL.value,
            reason="durable approval required",
            metadata={"source": "legacy"},
        )
        legacy_round["tool_calls"] = legacy_calls
        legacy_round.pop("policy_state")
        legacy_round.pop("policy_context_version")
        await store.checkpoint(
            session_id,
            {**checkpoint, "pending_tool_round": legacy_round},
        )
        await transition_runtime_session_to_running(store, session_id)

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
        assert approval.tool_call_id == "call_authoritative_approval"
        gating_call = next(
            call for call in approval.tool_calls if call.tool_call_id == approval.tool_call_id
        )
        assert gating_call.policy_evidence == "authoritative"
        assert gating_call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value

        events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        private_events = await _private_events_for_public_events(store, session_id, events)
        assert tool.calls == [{"value": "explicitly approved"}]
        assert any(
            event.type is EventType.TOOL_CALL_BLOCKED
            and event.payload.get("tool_call_id") == "call_ambiguous"
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
            for event in private_events
        )
        assert any(
            event.type is EventType.TOOL_CALL_APPROVED
            and event.payload.get("tool_call_id") == "call_authoritative_approval"
            for event in private_events
        )

    asyncio.run(run())


def test_ambiguous_policy_acknowledgement_is_retried_without_execution_or_decision_flip():
    class FailingAmbiguousApprovalCloseStore(FailingAfterPendingToolRoundCheckpointStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_close_once = False

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            if not self.failed_close_once and kwargs["request"].kind == "approval-close":
                self.failed_close_once = True
                raise RuntimeError("ambiguous approval close unavailable")
            return await super().publish_runtime_publication(session_id, **kwargs)

    async def run() -> None:
        session_id = "sess_ambiguous_policy_ack_retry"
        secret = "ambiguous-resolution-secret"
        store = FailingAmbiguousApprovalCloseStore()
        tool = SideEffectTool()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ambiguous",
                        name="side_effect",
                        arguments={"value": "must not execute"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("continued safely"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
        await transition_runtime_session_to_running(store, session_id)
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])

        request = ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval.approval_id,
            tool_round_id=approval.tool_round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
            reason="Continue safely without this side effect.",
            metadata={"credential": secret, "large": "x" * 20_000},
            resolved_by=ResolutionActor(
                subject="operator-7",
                tenant="tenant-1",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        first = await collect_tool_approval_events(app, request)
        assert tool.calls == []
        blocked = next(
            event
            for event in first
            if event.type is EventType.TOOL_CALL_BLOCKED
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
        )
        assert blocked.payload["decision"] == "ambiguous"
        assert blocked.payload["requested_decision"] == "approve"
        assert blocked.payload["resolution_reason"] == ("Continue safely without this side effect.")
        assert blocked.payload["resolved_by"] == {
            "subject": "operator-7",
            "tenant": "tenant-1",
            "source": "request",
        }
        assert blocked.payload["metadata_truncated"] is True
        serialized_metadata = json.dumps(blocked.payload["metadata"])
        assert secret not in serialized_metadata
        assert "[REDACTED_SECRET]" in serialized_metadata
        assert first[-1].type is EventType.SESSION_INTERRUPTED

        conflicting = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.DENY,
            ),
        )
        assert [event.type for event in conflicting] == [EventType.SESSION_INTERRUPTED]
        assert "cannot be retried as denied" in conflicting[0].payload["error"]
        assert tool.calls == []

        drifted = await collect_tool_approval_events(
            app,
            request.model_copy(update={"metadata": {"credential": "changed"}}),
        )
        assert [event.type for event in drifted] == [EventType.SESSION_INTERRUPTED]
        assert "different resolution request" in drifted[0].payload["error"]
        assert tool.calls == []

        retried = await collect_tool_approval_events(app, request)
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in retried)
        assert retried[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == []
        clear_event = next(
            event for event in retried if event.type is EventType.SESSION_CHECKPOINTED
        )

        replayed = await collect_tool_approval_events(app, request)
        assert [event.id for event in replayed] == [clear_event.id]
        assert tool.calls == []

        audit = business_approval_audit(await store.load_events(session_id))
        assert len(audit) == 1
        assert audit[0].resolution_state is BusinessApprovalResolutionState.BLOCKED
        assert audit[0].decision is None
        assert not audit[0].pending
        assert audit[0].reason == "Continue safely without this side effect."
        assert audit[0].resolved_at == blocked.timestamp

    asyncio.run(run())


@pytest.mark.parametrize("policy_decision", [None, "future_policy_decision"])
def test_planned_policy_round_without_authoritative_decision_fails_closed(
    policy_decision: str | None,
):
    async def run() -> None:
        session_id = "sess_invalid_planned_policy_recovery"
        store = FailingAfterPendingToolRoundCheckpointStore()
        tool = SideEffectTool()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "must remain gated"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        malformed_round = dict(checkpoint["pending_tool_round"])
        malformed_round["policy_state"] = "planned"
        malformed_round["policy_context_version"] = 1
        malformed_calls = [dict(call) for call in malformed_round["tool_calls"]]
        malformed_calls[0]["policy_decision"] = policy_decision
        malformed_round["tool_calls"] = malformed_calls
        checkpoint = dict(checkpoint)
        checkpoint["pending_tool_round"] = malformed_round
        await store.checkpoint(session_id, checkpoint)
        await transition_runtime_session_to_running(store, session_id)

        with pytest.raises(
            ValueError,
            match="checkpoint is invalid",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        retained = await store.load_checkpoint(session_id)
        assert retained is not None
        assert "pending_tool_round" in retained
        assert "pending_tool_approval" not in retained
        assert tool.calls == []

    asyncio.run(run())


def test_planned_unregistered_call_rejects_registration_drift_before_recovery():
    async def run() -> None:
        session_id = "sess_planned_unregistered_registration_drift"
        store = FailingAfterPendingToolRoundCheckpointStore()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(
            VersionedFakeProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call_late",
                        name="side_effect",
                        arguments={"value": "must not execute"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[],
        )
        await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        planned_round = dict(checkpoint["pending_tool_round"])
        planned_round["policy_state"] = "planned"
        planned_round["policy_context_version"] = 1
        planned_calls = [dict(call) for call in planned_round["tool_calls"]]
        planned_calls[0]["policy_evidence"] = "unregistered"
        planned_round["tool_calls"] = planned_calls
        await store.checkpoint(
            session_id,
            {**checkpoint, "pending_tool_round": planned_round},
        )
        await transition_runtime_session_to_running(store, session_id)

        tool = SideEffectTool()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(VersionedFakeProvider([]), default=True)
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        before = await store.load(session_id)
        assert before is not None
        with pytest.raises(ExecutionProfileMismatchError) as raised:
            await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        assert raised.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert tool.calls == []
        after = await store.load(session_id)
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch
        events = await store.load_events(session_id)
        assert events[-1].type is EventType.SESSION_EXECUTION_PROFILE_REJECTED
        assert not any(
            event.type is EventType.TOOL_CALL_STARTED
            and event.payload.get("tool_call_id") == "call_late"
            for event in events
        )

    asyncio.run(run())


def test_concurrent_incomplete_recovery_claims_tool_round_once() -> None:
    class BlockingRecoveryClaimStore(FailingAfterPendingToolRoundCheckpointStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.claim_fenced = asyncio.Event()
            self.allow_claim = asyncio.Event()

        async def reserve_stalled_run_recovery(self, *args, **kwargs):
            reserved = await super().reserve_stalled_run_recovery(*args, **kwargs)
            self.claim_fenced.set()
            await self.allow_claim.wait()
            return reserved

    class RecoveryHook(RuntimeHook):
        def __init__(self) -> None:
            self.tool_calls: list[str] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:runtime:concurrent-recovery-hook",
                behavior_version="1",
                implementation_version="1",
            )

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.tool_calls.append(context.tool_call_id)

    async def scenario() -> None:
        session_id = "sess_concurrent_incomplete_recovery"
        store = BlockingRecoveryClaimStore()
        hook = RecoveryHook()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            VersionedFakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call_1",
                            name="side_effect",
                            arguments={},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SideEffectTool()],
            runtime_hooks=[hook],
        )
        second_app = CayuApp(session_store=store, enable_logging=False)
        second_app.register_provider(VersionedFakeProvider([]), default=True)
        second_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SideEffectTool()],
            runtime_hooks=[hook],
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        round_id = checkpoint["pending_tool_round"]["tool_round_id"]
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        request = IncompleteSessionRecoveryRequest(session_id=session_id)
        first_task = asyncio.create_task(app.recover_incomplete_session(request))
        await asyncio.wait_for(store.claim_fenced.wait(), timeout=5)
        second = await asyncio.wait_for(second_app.recover_incomplete_session(request), timeout=5)
        assert second.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        with pytest.raises(
            RuntimeError,
            match="active incomplete-session recovery operation",
        ):
            await collect_tool_round_recovery_events(
                second_app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="must not race durable recovery",
                ),
            )
        with pytest.raises(
            RuntimeError,
            match="active incomplete-session recovery operation",
        ):
            await collect_resume_events(
                second_app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "must not race durable recovery")],
                ),
            )
        with pytest.raises(
            RuntimeError,
            match="active incomplete-session recovery operation",
        ):
            await collect_fork_events(
                second_app,
                ForkSessionRequest(
                    source_session_id=session_id,
                    session_id="sess_concurrent_incomplete_recovery_fork",
                    copy_checkpoint=True,
                ),
            )

        store.allow_claim.set()
        first = await asyncio.wait_for(first_task, timeout=5)
        assert first.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )

        events = await store.load_events(session_id)
        recovered_approval_events = [
            event
            for event in events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            and event.payload.get("recovered") is True
        ]
        assert len(recovered_approval_events) == 1
        transcript = await store.load_transcript(session_id)
        assert [message.role for message in transcript].count("tool") == 0
        assert hook.tool_calls == []
        recovered_checkpoint = await store.load_checkpoint(session_id)
        assert recovered_checkpoint is not None
        assert "pending_tool_approval" in recovered_checkpoint
        assert recovered_checkpoint["pending_tool_round"]["policy_state"] == "planned"

    asyncio.run(scenario())


def test_incomplete_recovery_renews_claim_while_hook_is_running(monkeypatch) -> None:
    class TransientHeartbeatFailureStore(FailingAfterPendingToolRoundCheckpointStore):
        invocation_lifecycle_command_version = 1

        def __init__(
            self,
            *,
            ownership_clock: Callable[[], datetime] | None = None,
        ) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.fail_next_transform = False

        async def transform_checkpoint_with_store_time(
            self,
            session_id,
            checkpoint_transform,
        ) -> None:
            if self.fail_next_transform:
                self.fail_next_transform = False
                raise RuntimeError("transient recovery heartbeat failure")
            await super().transform_checkpoint_with_store_time(
                session_id,
                checkpoint_transform,
            )

    class BlockingRecoveryHook(RuntimeHook):
        def __init__(self) -> None:
            self.block = False
            self.started = asyncio.Event()
            self.finish = asyncio.Event()
            self.execution_profiles: list[Any] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:runtime:blocking-recovery-hook",
                behavior_version="1",
                implementation_version="1",
            )

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            if not self.block:
                return
            self.execution_profiles.append(context.execution_profile)
            self.started.set()
            await self.finish.wait()

    async def scenario() -> None:
        session_id = "sess_incomplete_recovery_heartbeat"
        current_time = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

        def clock() -> datetime:
            return current_time["value"]

        store = TransientHeartbeatFailureStore(ownership_clock=clock)
        hook = BlockingRecoveryHook()
        app = CayuApp(session_store=store, enable_logging=False, clock=clock)
        app.register_provider(
            VersionedFakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call_1",
                            name="side_effect",
                            arguments={},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[PureRecordingTool()],
            runtime_hooks=[hook],
        )
        second_app = CayuApp(session_store=store, enable_logging=False, clock=clock)
        second_app.register_provider(VersionedFakeProvider([]), default=True)
        second_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[PureRecordingTool()],
            runtime_hooks=[hook],
        )

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
        await _checkpoint_allow_policy_plan(store, session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        hook.block = True
        recovery_task = asyncio.create_task(
            app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
        )
        await asyncio.wait_for(hook.started.wait(), timeout=10)

        store.fail_next_transform = True
        current_time["value"] += timedelta(minutes=4)
        for _attempt in range(100):
            checkpoint = await store.load_checkpoint(session_id)
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if type(marker) is dict and marker.get("renewed_at") == clock().isoformat():
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Incomplete recovery claim heartbeat did not renew the lease.")

        current_time["value"] += timedelta(minutes=2)
        competing = await second_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert competing.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)

        hook.finish.set()
        recovered = await asyncio.wait_for(recovery_task, timeout=5)
        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert_only_model_step_publication_checkpoint(checkpoint)
        active_profile = (
            execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
                checkpoint
            )
        )
        assert active_profile is not None
        assert hook.execution_profiles == [active_profile.profile]

    monkeypatch.setattr(
        recovery_ownership_module,
        "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        recovery_ownership_module,
        "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS",
        0.01,
    )
    asyncio.run(scenario())


def test_cayu_app_recover_incomplete_session_uses_recorded_tool_result():
    store = FailingOrdinaryToolResultCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_incomplete_recorded_tool_result",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_recorded_tool_result"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    asyncio.run(
        transition_runtime_session_to_running(
            store,
            "sess_recover_incomplete_recorded_tool_result",
        )
    )

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sess_recover_incomplete_recorded_tool_result"
            )
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_recover_incomplete_recorded_tool_result"))

    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert tool.calls == [{}]
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[2].content[0].tool_call_id == "call_1"
    assert transcript[2].content[0].content == "recorded"
    assert transcript[2].content[0].is_error is False
    assert_only_model_step_publication_checkpoint(
        asyncio.run(store.load_checkpoint("sess_recover_incomplete_recorded_tool_result"))
    )


def test_cayu_app_recovers_failed_session_pending_tool_round_before_publication():
    store = FailingOrdinaryToolResultCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    session_id = "sess_recover_failed_pending_tool_round"

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    session = asyncio.run(store.load(session_id))
    assert initial_events[-1].type is EventType.SESSION_FAILED
    assert session is not None
    assert session.status is SessionStatus.FAILED
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint

    result = asyncio.run(
        app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
    )

    recovered = asyncio.run(store.load(session_id))
    transcript = asyncio.run(store.load_transcript(session_id))
    assert recovered is not None
    assert recovered.status is SessionStatus.INTERRUPTED
    assert result.status is SessionStatus.INTERRUPTED
    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert tool.calls == [{}]
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[-1].content[0].content == "recorded"
    assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))


def test_cayu_app_recover_incomplete_session_preserves_pending_tool_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"value": "requires approval"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
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
                session_id="sess_recover_incomplete_pending_approval",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    asyncio.run(
        transition_runtime_session_to_running(
            store,
            "sess_recover_incomplete_pending_approval",
        )
    )

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_incomplete_pending_approval")
        )
    )
    session = asyncio.run(store.load("sess_recover_incomplete_pending_approval"))
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_pending_approval"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)
    assert result.pending_approval_id == f"{result.events[0].id}:approval_id"
    assert result.status == SessionStatus.INTERRUPTED
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert checkpoint is not None
    private_approval = _private_record_for_public_event(
        asyncio.run(
            store.query_events(EventQuery(session_id="sess_recover_incomplete_pending_approval"))
        ),
        approval_event,
    ).event
    assert (
        checkpoint["pending_tool_approval"]["approval_id"]
        == private_approval.payload["approval_id"]
    )
    assert "invocation_terminal_decision" not in checkpoint
    assert "settled_invocation_terminal_decision" not in checkpoint
    replay = asyncio.run(
        app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session.id))
    )
    assert replay.status == SessionStatus.INTERRUPTED
    replay_checkpoint = asyncio.run(store.load_checkpoint(session.id))
    assert replay_checkpoint is not None
    assert replay_checkpoint["pending_tool_approval"] == checkpoint["pending_tool_approval"]
    assert "invocation_terminal_decision" not in replay_checkpoint
    assert "settled_invocation_terminal_decision" not in replay_checkpoint
    assert "pending_session_interrupt" not in checkpoint
    assert tool.calls == []


def test_incomplete_sessions_recovery_request_requires_explicit_statuses():
    with pytest.raises(ValidationError, match="statuses"):
        IncompleteSessionsRecoveryRequest(limit=10)


def test_inactive_recovery_does_not_fence_local_active_work() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_active_recovery",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_active_recovery", SessionStatus.RUNNING)
        release = asyncio.Event()
        task = asyncio.create_task(release.wait())
        app._session_control.register_active_task(
            "sess_active_recovery",
            task,
            task_id=None,
            task_started=True,
            task_finished=False,
        )
        try:
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="sess_active_recovery",
                    inactive_for_seconds=0,
                )
            )
            session = await store.load("sess_active_recovery")
            events = await store.load_events("sess_active_recovery")
            return result, session, events
        finally:
            release.set()
            await task
            app._session_control.unregister_active_task("sess_active_recovery", task)

    result, session, events = asyncio.run(setup_and_recover())

    assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
    assert session is not None
    assert session.run_epoch == 1
    assert not any(event.type == EventType.SESSION_RUN_FENCED for event in events)


def test_cayu_app_recover_incomplete_sessions_with_explicit_interrupting_status():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str, status: SessionStatus) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, status)

    async def setup_and_recover():
        await create_session("sess_batch_interrupting", SessionStatus.INTERRUPTING)
        await create_session("sess_batch_running", SessionStatus.RUNNING)
        await create_session("sess_batch_pending", SessionStatus.PENDING)
        await create_session("sess_batch_completed", SessionStatus.COMPLETED)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.INTERRUPTING},
                limit=10,
            )
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    assert [result.session_id for result in results] == ["sess_batch_interrupting"]
    assert page.inspected_session_count == 1
    assert page.next_cursor is None
    assert results[0].status == SessionStatus.INTERRUPTED
    assert results[0].actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)


def test_cayu_app_recover_incomplete_sessions_fetches_only_requested_limit():
    store = RecordingListSessionsStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTING)

    async def setup_and_recover():
        await create_session("sess_batch_limit_1")
        await create_session("sess_batch_limit_2")
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.INTERRUPTING},
                limit=1,
            )
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    assert len(results) == 1
    assert len(store.session_queries) == 1
    assert store.session_queries[0].status == SessionStatus.INTERRUPTING
    assert store.session_queries[0].limit == 1
    assert store.session_queries[0].offset == 0


def test_cayu_app_recover_incomplete_sessions_filters_activity_before_limiting():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    worker_now = [now[0] + timedelta(days=3650)]
    store = InMemorySessionStore(ownership_clock=lambda: now[0])
    app = CayuApp(session_store=store, clock=lambda: worker_now[0])
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale_beyond_candidate_window",
                messages=[Message.text("user", "stale")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        early = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.PENDING},
                limit=1,
                inactive_for_seconds=1,
            )
        )
        assert early.results == ()
        worker_now[0] = now[0] - timedelta(days=3650)
        now[0] += timedelta(seconds=1)
        for index in range(1000):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_recent_{index:04d}",
                    messages=[Message.text("user", "recent")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.PENDING},
                limit=1,
                inactive_for_seconds=1,
            )
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    assert [result.session_id for result in results] == ["sess_stale_beyond_candidate_window"]
    assert results[0].actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)


def test_cayu_app_recover_incomplete_sessions_can_include_abandoned_running_and_pending():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str, status: SessionStatus) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, status)

    async def setup_and_recover():
        await create_session("sess_batch_running", SessionStatus.RUNNING)
        await create_session("sess_batch_pending", SessionStatus.PENDING)
        await create_session("sess_batch_completed", SessionStatus.COMPLETED)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING, SessionStatus.PENDING},
                limit=10,
            )
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    assert {result.session_id for result in results} == {
        "sess_batch_running",
        "sess_batch_pending",
    }
    assert {(result.session_id, result.status, result.actions) for result in results} == {
        (
            "sess_batch_running",
            SessionStatus.INTERRUPTED,
            (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
        ),
        (
            "sess_batch_pending",
            SessionStatus.INTERRUPTED,
            (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
        ),
    }


def test_cayu_app_recover_incomplete_sessions_skips_unregistered_agent_and_continues():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        async def create_session(session_id: str, agent_name: str, status: SessionStatus) -> None:
            await store.create(
                RunRequest(
                    agent_name=agent_name,
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            if status is not SessionStatus.PENDING:
                await store.update_status(session_id, status)

        await create_session("sess_sweep_healthy_a", "assistant", SessionStatus.RUNNING)
        await create_session("sess_sweep_healthy_b", "assistant", SessionStatus.RUNNING)
        await create_session("sess_sweep_ghost", "ghost_agent", SessionStatus.PENDING)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING, SessionStatus.PENDING},
                inactive_for_seconds=0,
            )
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    by_id = {result.session_id: result for result in results}
    assert by_id["sess_sweep_healthy_a"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert by_id["sess_sweep_healthy_b"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    ghost = by_id["sess_sweep_ghost"]
    assert ghost.actions == (IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,)
    assert "ghost_agent" in ghost.message
    ghost_session = asyncio.run(store.load("sess_sweep_ghost"))
    assert ghost_session is not None
    assert ghost_session.status == SessionStatus.PENDING
    assert ghost_session.run_epoch == 0
    ghost_events = asyncio.run(store.load_events("sess_sweep_ghost"))
    assert not any(event.type == EventType.SESSION_RUN_FENCED for event in ghost_events)


def test_cayu_app_recover_incomplete_sessions_isolates_mid_batch_failure(monkeypatch):
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret = "batch-recovery-diagnostic-canary"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        for session_id in ("sess_sweep_ok_one", "sess_sweep_broken", "sess_sweep_ok_two"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.RUNNING)

        original_load_checkpoint = store.load_checkpoint

        async def broken_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            if session_id == "sess_sweep_broken":
                raise RuntimeError(f"checkpoint store exploded with {secret}")
            return await original_load_checkpoint(session_id)

        monkeypatch.setattr(store, "load_checkpoint", broken_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    by_id = {result.session_id: result for result in results}
    assert by_id["sess_sweep_ok_one"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert by_id["sess_sweep_ok_two"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    broken = by_id["sess_sweep_broken"]
    assert broken.actions == (IncompleteSessionRecoveryAction.FAILED,)
    assert "checkpoint store exploded" in broken.message
    assert secret not in broken.message
    assert REDACTED_SECRET in broken.message


def test_cayu_app_recover_incomplete_sessions_failed_entry_reports_current_status(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sweep_mutated",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_sweep_mutated", SessionStatus.RUNNING)

        original_load_checkpoint = store.load_checkpoint
        original_transition = store.transition_status_and_checkpoint
        transitioned = {"value": False}

        async def track_transition(*args, **kwargs):
            result = await original_transition(*args, **kwargs)
            transitioned["value"] = True
            return result

        # Fail the checkpoint reload after RUNNING -> INTERRUPTING regardless
        # of how many claim-validation reads precede that durable transition.
        async def flaky_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            if transitioned["value"]:
                transitioned["value"] = False
                raise RuntimeError("checkpoint reload exploded")
            return await original_load_checkpoint(session_id)

        monkeypatch.setattr(store, "transition_status_and_checkpoint", track_transition)
        monkeypatch.setattr(store, "load_checkpoint", flaky_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    page = asyncio.run(setup_and_recover())
    results = page.results

    (failed,) = results
    assert failed.actions == (IncompleteSessionRecoveryAction.FAILED,)
    assert "checkpoint reload exploded" in failed.message
    assert failed.previous_status == SessionStatus.RUNNING
    assert failed.status == SessionStatus.INTERRUPTING
    session = asyncio.run(store.load("sess_sweep_mutated"))
    assert session is not None
    assert session.status == failed.status


def test_cayu_app_recover_incomplete_sessions_propagates_cancellation(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sweep_cancelled",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_sweep_cancelled", SessionStatus.RUNNING)

        async def cancelled_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            raise asyncio.CancelledError()

        monkeypatch.setattr(store, "load_checkpoint", cancelled_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(setup_and_recover())


def test_cayu_app_recover_incomplete_session_missing_session_still_raises():
    app = CayuApp(session_store=InMemorySessionStore())

    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="sess_never_created")
            )
        )


def test_cayu_app_recover_incomplete_session_unregistered_agent_returns_typed_skip():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="ghost_agent",
                session_id="sess_single_ghost",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_single_ghost")
        )

    result = asyncio.run(setup_and_recover())

    assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,)
    assert result.previous_status == SessionStatus.PENDING
    assert result.status == SessionStatus.PENDING
    assert "ghost_agent" in result.message
    session = asyncio.run(store.load("sess_single_ghost"))
    assert session is not None
    assert session.status == SessionStatus.PENDING

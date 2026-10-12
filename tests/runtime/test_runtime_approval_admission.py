"""Approval admission contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import (
    checkpoint_with_rebound_test_invocation_profile,
    interrupt_and_release_test_invocation,
    rebind_test_invocation,
)
from tests.runtime.runtime_budget_support import fake_budget_limit
from tests.runtime.runtime_environment_support import (
    EnvironmentProvisioningRecorder,
    assert_environment_was_not_provisioned,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailNextRunFenceReleaseStore
from tests.runtime.runtime_session_support import (
    _approval_pause_app,
    _pending_tool_approval_from_public_event,
    _private_events_for_public_events,
    _reviewer_actor,
    _tool_round_identity,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
    tool_round_identity_payload,
)
from tests.runtime.runtime_tool_support import (
    EchoTool,
    ExpiringApprovalPolicy,
    RequireApprovalPolicy,
    SideEffectApprovalPolicy,
    SideEffectTool,
)

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.budgets.run_limits import RunLimits
from cayu.environments.base import EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _approval_support as approval_support_module
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.vaults.redaction import SecretRedactor


def _assert_events_share_one_interaction(*event_batches: list[Event]) -> str:
    interaction_ids = {
        event.interaction_id
        for events in event_batches
        for event in events
        if event.type not in sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    }
    assert len(interaction_ids) == 1
    interaction_id = next(iter(interaction_ids))
    assert interaction_id is not None
    return interaction_id


def test_cayu_app_interrupts_session_when_tool_policy_requires_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
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
                session_id="sess_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == []

    private_events = asyncio.run(
        _private_events_for_public_events(store, "sess_tool_approval", events)
    )
    approval = private_events[4].payload["approval"]
    assert events[-1].payload["interruption_type"] == "tool_approval_required"
    assert private_events[-1].payload["approval"]["approval_id"] == approval["approval_id"]
    assert approval["tool_call_id"] == "call_1"
    assert approval["tool_name"] == "side_effect"
    assert "arguments" not in approval
    assert approval["arguments_state"] == "quarantined"
    assert approval["agent_name"] == "assistant"
    assert approval["reason"] == "Approval required for side_effect."
    assert approval["metadata"] == {"scope": "human"}
    assert [call["tool_call_id"] for call in approval["tool_calls"]] == ["call_1"]

    session = asyncio.run(store.load("sess_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED

    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval"))
    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["approval_id"] == approval["approval_id"]
    assert checkpoint["pending_tool_approval"]["publish_arguments"] is True
    assert checkpoint["pending_tool_approval"]["arguments"] == {"value": "secret"}

    transcript = asyncio.run(store.load_transcript("sess_tool_approval"))
    assert [message.role for message in transcript] == ["user"]
    quarantined_assistant = Message.model_validate(
        checkpoint["pending_tool_round"]["quarantined_assistant_message"]
    )
    assert quarantined_assistant.role == "assistant"
    assert quarantined_assistant.content[0].type == "tool_call"


def test_cayu_app_resolves_approved_tool_call_and_continues_session():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    later_default_provisioning = EnvironmentProvisioningRecorder("later-default")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("approved handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
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

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_allow",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    approval_id = approval["approval_id"]
    app.register_environment_factory(
        EnvironmentSpec(name="later-default"),
        later_default_provisioning.factory,
        default=True,
    )

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_allow",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                reason="approved by test",
                metadata={"reviewer": "test"},
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == [{"value": "secret"}]
    assert_environment_was_not_provisioned(later_default_provisioning)
    assert {event.environment_name for event in [*interrupt_events, *events]} == {None}
    assert provider.requests[1].messages[-2].role == "assistant"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == "recorded"

    session = asyncio.run(store.load("sess_tool_approval_allow"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert session.environment_name is None
    assert_only_model_step_publication_checkpoint(
        asyncio.run(store.load_checkpoint("sess_tool_approval_allow"))
    )


def test_stale_tool_approval_resolver_cannot_claim_repaused_session():
    from cayu.approvals.tools import PendingToolCallApproval
    from cayu.execution_units import new_model_step_identity

    class BlockingApprovalClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_next_approval_claim = False
            self.claim_started = asyncio.Event()
            self.allow_claim = asyncio.Event()

        async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
            if self.block_next_approval_claim and kwargs["to_status"] is SessionStatus.RUNNING:
                self.block_next_approval_claim = False
                self.claim_started.set()
                await self.allow_claim.wait()
            return await super().transition_status_and_checkpoint(session_id, **kwargs)

    async def run() -> None:
        session_id = "sess_stale_approval_repause"
        store = BlockingApprovalClaimStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_a = await _pending_tool_approval_from_public_event(store, approval_event)
        checkpoint_a = await store.load_checkpoint(session_id)
        assert checkpoint_a is not None
        round_a = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint_a)
        assert round_a is not None

        replacement_identity = new_model_step_identity().new_attempt().new_tool_round()
        replacement_call = PendingToolCallApproval(
            tool_call_id="call_2",
            tool_name=approval_a.tool_name,
            arguments={"value": "replacement"},
            policy_evidence="authoritative",
            policy_decision=ToolPolicyDecision.REQUIRE_APPROVAL.value,
            reason="replacement approval",
            metadata={"scope": "replacement"},
        )
        round_b = pending_rounds.PendingToolRound(
            **{
                **round_a.model_dump(mode="json"),
                **replacement_identity.payload(),
                "tool_calls": [replacement_call.model_dump(mode="json")],
            }
        )
        approval_b = PendingToolApproval(
            **{
                **approval_a.model_dump(mode="json"),
                **replacement_identity.payload(),
                "approval_id": "approval_replacement",
                "tool_call_id": replacement_call.tool_call_id,
                "arguments": replacement_call.arguments,
                "reason": replacement_call.reason,
                "metadata": replacement_call.metadata,
                "tool_calls": [replacement_call.model_dump(mode="json")],
            }
        )
        store.block_next_approval_claim = True
        stale_resolution = asyncio.create_task(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval_a.approval_id,
                    tool_round_id=approval_a.tool_round_id,
                    tool_call_id=approval_a.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )
        await asyncio.wait_for(store.claim_started.wait(), timeout=10)

        def replace_pending_approval(
            session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            rebound = checkpoint_with_rebound_test_invocation_profile(session, checkpoint)
            rebound["pending_tool_round"] = round_b.model_dump(mode="json")
            rebound["pending_tool_approval"] = approval_b.model_dump(mode="json")
            return rebound

        claimed = await rebind_test_invocation(
            store,
            session_id,
            checkpoint_transform=replace_pending_approval,
        )
        await interrupt_and_release_test_invocation(store, session_id)
        repaused = await store.load(session_id)
        assert repaused is not None
        repaused_epoch = repaused.run_epoch
        repaused_checkpoint = await store.load_checkpoint(session_id)
        store.allow_claim.set()

        with pytest.raises(
            ValueError,
            match="identity does not match the current pending approval",
        ):
            await stale_resolution
        current = await store.load(session_id)
        assert current is not None
        assert current.status is SessionStatus.INTERRUPTED
        assert repaused_epoch == claimed.run_epoch + 1
        assert current.run_epoch == repaused_epoch
        assert await store.load_checkpoint(session_id) == repaused_checkpoint
        assert tool.calls == []

    asyncio.run(run())


def test_tool_approval_resolution_rejects_partial_identity_without_mutation():
    from cayu.runtime import _approval_support as approval_support
    from cayu.vaults.redaction import SecretRedactor

    async def run() -> None:
        session_id = "sess_approval_identity_mismatch"
        store = InMemorySessionStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = await _pending_tool_approval_from_public_event(store, approval_event)
        checkpoint_before = await store.load_checkpoint(session_id)
        session_before = await store.load(session_id)
        assert session_before is not None

        for tool_round_id, tool_call_id in (
            ("tround_stale", approval.tool_call_id),
            (approval.tool_round_id, "call_stale"),
        ):
            with pytest.raises(
                ValueError,
                match="identity does not match the current pending approval",
            ):
                await collect_tool_approval_events(
                    app,
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=tool_round_id,
                        tool_call_id=tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    ),
                )

        session_after = await store.load(session_id)
        assert session_after is not None
        assert session_after.status is SessionStatus.INTERRUPTED
        assert session_after.run_epoch == session_before.run_epoch
        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert tool.calls == []

        assert checkpoint_before is not None
        replacement = PendingToolApproval(
            **{
                **approval.model_dump(mode="json"),
                "approval_id": "approval_replacement_before_clear",
            }
        )
        replacement_checkpoint = {
            **checkpoint_before,
            "pending_tool_approval": replacement.model_dump(mode="json"),
        }
        with pytest.raises(
            RuntimeError,
            match="changed before exact checkpoint clearing",
        ):
            approval_support.checkpoint_without_exact_pending_approval_round(
                replacement_checkpoint,
                approval=approval,
                redactor=SecretRedactor(),
            )
        assert replacement_checkpoint["pending_tool_approval"]["approval_id"] == (
            replacement.approval_id
        )

    asyncio.run(run())


def test_concurrent_approval_and_denial_have_exactly_one_durable_winner():
    class TwoClaimBarrierStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_claims = False
            self.claim_count = 0
            self.both_claims_started = asyncio.Event()
            self.release_claims = asyncio.Event()

        async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
            if self.block_claims and kwargs["to_status"] is SessionStatus.RUNNING:
                self.claim_count += 1
                if self.claim_count == 2:
                    self.both_claims_started.set()
                await self.release_claims.wait()
            return await super().transition_status_and_checkpoint(session_id, **kwargs)

    async def run() -> None:
        session_id = "sess_concurrent_approval_decisions"
        store = TwoClaimBarrierStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = await _pending_tool_approval_from_public_event(store, approval_event)
        store.block_claims = True

        async def resolve(decision: ToolApprovalDecision):
            return await collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=decision,
                ),
            )

        approve_task = asyncio.create_task(resolve(ToolApprovalDecision.APPROVE))
        deny_task = asyncio.create_task(resolve(ToolApprovalDecision.DENY))
        await asyncio.wait_for(store.both_claims_started.wait(), timeout=10)
        store.release_claims.set()
        outcomes = await asyncio.gather(approve_task, deny_task, return_exceptions=True)

        successes = [outcome for outcome in outcomes if type(outcome) is list]
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], SessionStatusConflict)
        events = await store.load_events(session_id)
        decision_events = [
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_APPROVAL_DENIED}
        ]
        assert len(decision_events) == 1
        assert decision_events[0].payload["approval_id"] == approval.approval_id
        if decision_events[0].type is EventType.TOOL_CALL_APPROVED:
            assert tool.calls == [{"value": "secret"}]
        else:
            assert tool.calls == []
        assert "pending_tool_approval" not in (await store.load_checkpoint(session_id) or {})

    asyncio.run(run())


def test_tool_approval_claim_acknowledgement_loss_replays_exact_command():
    class LostClaimAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_next_claim_ack = False

        async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
            result = await super().transition_status_and_checkpoint(session_id, **kwargs)
            if self.lose_next_claim_ack and kwargs["to_status"] is SessionStatus.RUNNING:
                self.lose_next_claim_ack = False
                raise OSError("approval claim acknowledgement lost")
            return result

    async def run() -> None:
        session_id = "sess_approval_claim_ack_loss"
        store = LostClaimAcknowledgementStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = await _pending_tool_approval_from_public_event(store, approval_event)
        store.lose_next_claim_ack = True

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
        completed = await store.load(session_id)
        assert completed is not None
        assert completed.status is SessionStatus.COMPLETED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_tool_approval" not in checkpoint
        assert sum(event.type is EventType.TOOL_CALL_APPROVED for event in events) == 1
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "secret"}]

    asyncio.run(run())


def test_tool_approval_publication_acknowledgement_loss_preserves_atomic_pair():
    class LostPublicationAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lost_approval_publication_ack = False

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            result = await super().publish_runtime_publication(session_id, **kwargs)
            if not self.lost_approval_publication_ack and kwargs["request"].kind == "approval-open":
                self.lost_approval_publication_ack = True
                raise OSError("approval publication acknowledgement lost")
            return result

    async def run() -> None:
        session_id = "sess_approval_publication_ack_loss"
        store = LostPublicationAcknowledgementStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        assert store.lost_approval_publication_ack is True
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
        pending_round = checkpoint["pending_tool_round"]
        assert pending_round["policy_state"] == "planned"
        assert pending_round["tool_round_id"] == approval.tool_round_id
        requested_events = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        ]
        assert len(requested_events) == 1
        assert requested_events[0].payload["approval_id"] == approval.approval_id
        assert requested_events[0].interaction_id is not None
        receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"approval-open:{approval.approval_id}",
        )
        assert receipt is not None
        assert receipt.kind == "approval-open"
        assert receipt.interaction_id == requested_events[0].interaction_id
        assert tool.calls == []

        current = await store.load(session_id)
        assert current is not None
        assert current.status is SessionStatus.INTERRUPTED
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        assert recovered.pending_approval_id is not None
        assert (
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=recovered.pending_approval_id,
                field_name="approval_id",
            )
            == approval.approval_id
        )
        repaired_interrupts = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("approval", {}).get("approval_id") == approval.approval_id
        ]
        assert len(repaired_interrupts) == 1
        assert tool.calls == []

    asyncio.run(run())


def test_tool_approval_requested_audit_metadata_is_bounded_and_redacted():
    from cayu.approvals.tools import PendingToolCallApproval
    from cayu.runtime import _approval_support as approval_support
    from cayu.sessions.pending_action_contracts import PendingActionQuery
    from cayu.sessions.records import PendingActionKind
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret = "approval-audit-secret"

    class OversizedApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="review required",
                metadata={
                    "classification": "public",
                    "large": "x" * 20_000,
                },
            )

    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"value": "safe"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
        tools=[tool],
        tool_policy=OversizedApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bounded_approval_audit",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events), [
        (event.type, event.payload) for event in events
    ]
    approval_event = next(
        event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    assert approval_event.payload["approval"]["reason"] == "review required"
    assert approval_event.payload["approval"]["metadata"]["classification"] == "public"
    assert approval_event.payload["approval_metadata_truncated"] is True
    assert approval_event.payload["tool_call_metadata_truncated"] == ["call_1"]
    assert (
        len(
            json.dumps(
                approval_event.payload["approval"]["metadata"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= 12 * 1024
    )
    assert (
        approval_event.payload["approval_id"] == approval_event.payload["approval"]["approval_id"]
    )
    assert approval_event.payload["tool_call_id"] == f"{approval_event.id}:tool_call_id"
    assert tool.calls == []
    interrupted_event = next(
        event for event in events if event.type is EventType.SESSION_INTERRUPTED
    )
    assert interrupted_event.payload["approval_metadata_truncated"] is True
    assert interrupted_event.payload["tool_call_metadata_truncated"] == ["call_1"]
    assert interrupted_event.payload["approval"]["approval_id"] == (
        f"{interrupted_event.id}:approval_id"
    )
    pending_actions = asyncio.run(
        store.query_pending_actions(PendingActionQuery(session_id="sess_bounded_approval_audit"))
    )
    assert len(pending_actions.actions) == 1
    assert pending_actions.actions[0].kind is PendingActionKind.TOOL_APPROVAL
    approval = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    assert pending_actions.actions[0].approval_id == approval.approval_id
    secret_call = PendingToolCallApproval(
        **{
            **approval.tool_calls[0].model_dump(mode="json"),
            "metadata": {"credential": secret},
        }
    )
    secret_approval = PendingToolApproval(
        **{
            **approval.model_dump(mode="json"),
            "metadata": {"credential": secret},
            "tool_calls": [secret_call.model_dump(mode="json")],
        }
    )
    redacted_payload = approval_support.bounded_pending_approval_event_payload(
        secret_approval,
        redactor=SecretRedactor(secret),
    )
    redacted_serialized = json.dumps(redacted_payload, sort_keys=True)
    assert secret not in redacted_serialized
    assert REDACTED_SECRET in redacted_serialized
    unknown_payload = approval_support.bounded_pending_approval_event_payload(
        secret_approval.model_copy(update={"secret_resolution_scope": "unknown"}),
        redactor=SecretRedactor(),
    )
    unknown_approval = unknown_payload["approval"]
    assert "reason" not in unknown_approval
    assert "metadata" not in unknown_approval
    assert all("reason" not in call for call in unknown_approval["tool_calls"])
    assert all("metadata" not in call for call in unknown_approval["tool_calls"])


class _BlockingApprovalTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Block an approved side effect until its consumer is cancelled.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Event | None = None
        self.never_complete: asyncio.Event | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if self.started is None or self.never_complete is None:
            raise AssertionError("Blocking approval tool events were not initialized.")
        self.started.set()
        await self.never_complete.wait()
        return ToolResult(content="unexpected")


def test_tool_approval_resolution_retains_round_authority_for_unknown_secret_scope():
    async def run() -> tuple[list[Event], list[dict[str, str]], dict[str, Any] | None]:
        session_id = "sess_approval_unknown_secret_scope"
        store = InMemorySessionStore()
        tool = SideEffectTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        # Unknown secret scope is supported, but the paired round is still
        # authoritative for the consumed model step and executable call plan.
        # Do not simulate an obsolete approval-only format by deleting it.
        assert checkpoint["pending_tool_round"]["model_step"] == 1
        unknown_scope_checkpoint = dict(checkpoint)
        unknown_scope_approval = dict(checkpoint["pending_tool_approval"])
        unknown_scope_approval.pop("secret_resolution_scope")
        unknown_scope_checkpoint["pending_tool_approval"] = unknown_scope_approval
        await store.checkpoint(session_id, unknown_scope_checkpoint)

        resumed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_event.payload["approval_id"],
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        return resumed, tool.calls, await store.load_checkpoint(session_id)

    events, calls, checkpoint = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert calls == [{"value": "secret"}]
    assert_only_model_step_publication_checkpoint(checkpoint)


def test_tool_approval_resolution_events_carry_resolved_by_actor():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(store=store, tool=tool, policy=RequireApprovalPolicy())

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_actor",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                reason="looks safe",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    # `claims` stay on the request and are excluded from event payloads.
    expected_actor = {
        "subject": "reviewer@example.com",
        "tenant": "acme",
        "source": "request",
    }
    resumed = next(event for event in events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"] == expected_actor
    assert resumed.payload["expired"] is False
    approved = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVED)
    assert approved.payload["resolved_by"] == expected_actor
    assert tool.calls == [{"value": "secret"}]


def test_tool_approval_resolution_closes_continuation_before_aclose_returns():
    class RecordingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.release_calls = 0

        async def release_session_invocation(self, command):
            self.release_calls += 1
            return await super().release_session_invocation(command)

    async def run() -> tuple[Event, SessionStatus, int, bool]:
        session_id = "sess_approval_resolution_stream_closed"
        store = RecordingReleaseStore()
        app, _provider = _approval_pause_app(
            store=store,
            tool=SideEffectTool(),
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        releases_before = store.release_calls
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        )
        while True:
            boundary_event = await anext(stream)
            if boundary_event.type == EventType.MODEL_STARTED:
                break
        assert app._session_control.has_active_tasks(session_id) is True
        await stream.aclose()
        release_delta = store.release_calls - releases_before
        has_active_tasks = app._session_control.has_active_tasks(session_id)
        session = await store.load(session_id)
        assert session is not None
        return boundary_event, session.status, release_delta, has_active_tasks

    boundary_event, status, release_delta, has_active_tasks = asyncio.run(run())

    assert boundary_event.type == EventType.MODEL_STARTED
    assert status == SessionStatus.INTERRUPTED
    assert release_delta == 1
    assert has_active_tasks is False


def test_tool_approval_cleanup_failure_has_one_forwarding_owner() -> None:
    async def scenario() -> None:
        session_id = "sess_approval_cleanup_failure"
        store = FailNextRunFenceReleaseStore()
        app, _provider = _approval_pause_app(
            store=store,
            tool=SideEffectTool(),
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        )
        while (await anext(stream)).type != EventType.MODEL_STARTED:
            pass

        loop = asyncio.get_running_loop()
        reported_contexts: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported_contexts.append(context))
        try:
            store.fail_next_release = True
            with pytest.raises(RuntimeError, match="run fence release failed") as raised:
                await stream.aclose()
            assert isinstance(raised.value.__cause__, OSError)
            await asyncio.sleep(0)
            assert reported_contexts == []
        finally:
            loop.set_exception_handler(previous_handler)

        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_tool_approval_resolution_task_cancellation_finalizes_and_preserves_pending_state():
    class RecordingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.release_calls = 0

        async def release_session_invocation(self, command):
            self.release_calls += 1
            return await super().release_session_invocation(command)

    async def run() -> None:
        session_id = "sess_approval_resolution_task_cancelled"
        store = RecordingReleaseStore()
        tool = _BlockingApprovalTool()
        tool.started = asyncio.Event()
        tool.never_complete = asyncio.Event()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        private_approval = await _pending_tool_approval_from_public_event(store, approval_event)

        releases_before = store.release_calls
        resolution_task = asyncio.create_task(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )
        await asyncio.wait_for(tool.started.wait(), timeout=10)
        assert resolution_task.cancelling() == 0
        resolution_task.cancel("cancel tool approval resolution")
        assert resolution_task.cancelling() == 1
        try:
            await resolution_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("cancel tool approval resolution",)
        else:
            pytest.fail("Tool approval resolution did not preserve task cancellation.")

        assert resolution_task.cancelled() is True
        assert resolution_task.cancelling() == 1
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_tool_approval"]["approval_id"] == private_approval.approval_id
        events = await store.load_events(session_id)
        assert events[-1].type == EventType.SESSION_INTERRUPTED
        assert events[-1].payload["abandoned"] is True
        assert store.release_calls - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_tool_approval_denial_events_carry_resolved_by_actor():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(store=store, tool=tool, policy=RequireApprovalPolicy())

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_denial_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_denial_actor",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    _assert_events_share_one_interaction(interrupt_events, events)
    assert denied.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert denied.payload["resolved_by"]["source"] == "request"
    assert denied.payload["expired"] is False
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert tool.calls == []


def test_resolution_actor_rejects_reserved_subject_for_request_sources():
    with pytest.raises(ValidationError, match="reserved for system actors"):
        ResolutionActor(subject="cayu:approval-expiry")
    with pytest.raises(ValidationError, match="reserved for system actors"):
        ResolutionActor(subject="cayu:anything", source=ResolutionActorSource.REQUEST)
    from cayu.approvals.tools import EXPIRY_RESOLUTION_ACTOR_SUBJECT

    system_actor = ResolutionActor(
        subject=EXPIRY_RESOLUTION_ACTOR_SUBJECT,
        source=ResolutionActorSource.SYSTEM,
    )
    assert system_actor.subject == EXPIRY_RESOLUTION_ACTOR_SUBJECT


def test_expired_tool_approval_resolves_as_deterministic_denial():
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_expiry",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    assert pending.expires_at == datetime(2026, 7, 9, 12, 1, tzinfo=UTC)

    clock["now"] = datetime(2026, 7, 9, 12, 2, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_expiry",
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approve it anyway",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    resumed = next(event for event in events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["decision"] == "deny"
    assert resumed.payload["expired"] is True
    assert resumed.payload["resolved_by"]["subject"] == "cayu:approval-expiry"
    assert resumed.payload["resolved_by"]["source"] == "system"

    expired_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
    )
    assert expired_event.payload["approval_id"] == f"{expired_event.id}:approval_id"
    private_expired_event = asyncio.run(
        _private_events_for_public_events(store, "sess_approval_expiry", [expired_event])
    )[0]
    assert private_expired_event.payload["approval_id"] == pending.approval_id
    assert expired_event.payload["expires_at"] == "2026-07-09T12:01:00+00:00"
    assert expired_event.payload["requested_decision"] == "approve"
    assert expired_event.payload["resolved_by"]["subject"] == "cayu:approval-expiry"
    assert expired_event.payload["triggered_by"]["subject"] == "reviewer@example.com"

    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    _assert_events_share_one_interaction(interrupt_events, events)
    assert denied.payload["expired"] is True
    assert "expired at" in denied.payload["reason"]
    assert denied.payload["resolved_by"]["source"] == "system"

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == []
    session = asyncio.run(store.load("sess_approval_expiry"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_unexpired_tool_approval_resolves_normally():
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_unexpired",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    clock["now"] = datetime(2026, 7, 9, 12, 0, 30, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_unexpired",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert any(event.type == EventType.TOOL_CALL_APPROVED for event in events)
    assert tool.calls == [{"value": "secret"}]


def test_multi_call_approval_round_uses_minimum_ttl():
    # The round is approved/denied as a whole, so any approval-requiring
    # call's TTL bounds the round: the shortest window wins, not the gating
    # (first) call's.
    class PerToolTtlPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                approval_expires_in_seconds=300 if request.tool_name == "side_effect" else 60,
            )

    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "one"},
                ),
                ModelStreamEvent.tool_call(id="call_2", name="echo", arguments={"text": "two"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=lambda: clock["now"])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool(), EchoTool()],
        tool_policy=PerToolTtlPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_ttl",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    # The gating call (side_effect, 300s) does not win: echo's 60s bound does.
    assert pending.expires_at == datetime(2026, 7, 9, 12, 1, tzinfo=UTC)


def test_expired_approval_retry_after_recorded_grant_is_not_coerced():
    # Expiry gates the FIRST grant only: a retry of an approval that was
    # granted in-window (crash after tool.call.approved) must honor the grant
    # even when the retry lands after the window — coercing it to a denial
    # would contradict the recorded grant and wedge the session.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expiry_retry",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    private_approval_event = asyncio.run(
        _private_events_for_public_events(store, "sess_expiry_retry", [approval_event])
    )[0]
    request = ToolApprovalRequest(
        session_id="sess_expiry_retry",
        approval_id=approval.approval_id,
        tool_round_id=approval.tool_round_id,
        tool_call_id=approval.tool_call_id,
        decision=ToolApprovalDecision.APPROVE,
    )
    resolution_request_digest = approval_support_module.approval_resolution_request_digest(request)

    # A prior in-window resolve durably claimed this exact request, then
    # crashed after recording the grant.
    checkpoint = asyncio.run(store.load_checkpoint("sess_expiry_retry"))
    asyncio.run(
        store.checkpoint(
            "sess_expiry_retry",
            approval_support_module.checkpoint_with_approval_resolution_intent(
                checkpoint,
                approval=approval,
                decision=request.decision,
                resolution_request_digest=resolution_request_digest,
                redactor=SecretRedactor(),
            ),
        )
    )
    asyncio.run(
        store.append_event(
            "sess_expiry_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVED,
                session_id="sess_expiry_retry",
                interaction_id=approval_event.interaction_id,
                agent_name="assistant",
                tool_name="side_effect",
                payload={
                    **tool_round_identity_payload(private_approval_event),
                    "approval_id": approval.approval_id,
                    "tool_call_id": "call_1",
                    "requested_decision": request.decision.value,
                    "resolution_request_digest": resolution_request_digest,
                },
            ),
        )
    )

    clock["now"] = datetime(2026, 7, 9, 12, 5, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            request,
        )
    )

    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_DENIED for event in events)
    _assert_events_share_one_interaction(interrupt_events, events)
    assert tool.calls == [{"value": "secret"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_rejects_tool_approval_resolution_from_failed_status():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]
    asyncio.run(store.update_status("sess_failed_tool_approval", SessionStatus.FAILED))

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_failed_tool_approval",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_failed_tool_approval",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="confirmed externally",
                ),
            )
        )


def test_pending_tool_approval_run_config_round_trips_json_checkpoint():
    from cayu.approvals.tools import (
        PendingToolApproval,
        PendingToolCallApproval,
        copy_pending_tool_approval,
    )

    pending = PendingToolApproval(
        approval_id="appr_1",
        **_tool_round_identity().payload(),
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
        max_steps=7,
        limits=RunLimits(max_tool_calls=3, scope="session"),
        budget_limits=(fake_budget_limit("2.50"),),
        retry_policy=RetryPolicy(max_attempts=3),
        expires_at=datetime(2026, 7, 9, 12, 30, tzinfo=UTC),
    )

    # Same round-trip as the durable checkpoint: model_dump(mode="json") in,
    # PendingToolApproval(**payload) out.
    restored = PendingToolApproval(**pending.model_dump(mode="json"))
    assert restored.max_steps == 7
    assert restored.limits == RunLimits(max_tool_calls=3, scope="session")
    assert restored.budget_limits == (fake_budget_limit("2.50"),)
    assert restored.retry_policy == RetryPolicy(max_attempts=3)
    assert restored.expires_at == pending.expires_at

    copied = copy_pending_tool_approval(pending)
    assert pending.publish_arguments is True
    assert restored.publish_arguments is True
    assert copied.publish_arguments is True
    assert copied.max_steps == 7
    assert copied.limits == pending.limits
    assert copied.budget_limits == pending.budget_limits
    assert copied.retry_policy == pending.retry_policy
    assert copied.expires_at == pending.expires_at


def test_pending_tool_approval_loads_checkpoint_without_run_config():
    from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval

    legacy = PendingToolApproval(
        approval_id="appr_legacy",
        **_tool_round_identity().payload(),
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
    )
    payload = legacy.model_dump(mode="json")
    for key in ("max_steps", "limits", "budget_limits", "retry_policy", "expires_at"):
        payload.pop(key)

    restored = PendingToolApproval(**payload)
    assert restored.publish_arguments is True
    assert restored.max_steps is None
    assert restored.limits is None
    assert restored.budget_limits is None
    assert restored.retry_policy is None
    assert restored.expires_at is None


@pytest.mark.parametrize("authority", ["missing", None, 0, 1, "true"])
def test_pending_tool_approval_rejects_checkpoint_without_explicit_publication_authority(
    authority: object,
):
    from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval

    pending = PendingToolApproval(
        approval_id="appr_missing_publication_authority",
        **_tool_round_identity().payload(),
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
    )
    payload = pending.model_dump(mode="json")
    if authority == "missing":
        payload.pop("publish_arguments")
    else:
        payload["publish_arguments"] = authority

    with pytest.raises(
        ValueError,
        match="Pending tool approval checkpoint is invalid and cannot be executed",
    ):
        pending_approval_reader.pending_approval_from_checkpoint(
            {"pending_tool_approval": payload},
            redactor=SecretRedactor(),
        )


def _run_config_approval_app() -> tuple[InMemorySessionStore, SideEffectTool, CayuApp]:
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="echo",
                    arguments={"text": "after approval"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, EchoTool()],
        tool_policy=SideEffectApprovalPolicy(),
    )
    return store, tool, app


@pytest.mark.parametrize("restate_configuration", [False, True])
@pytest.mark.parametrize("max_steps", [7, 10000])
def test_cayu_app_resolving_tool_approval_restores_original_run_limits(
    restate_configuration: bool,
    max_steps: int,
):
    store, tool, app = _run_config_approval_app()
    session_id = "sess_approval_restores_run_config"

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
                max_steps=max_steps,
                limits=RunLimits(max_tool_calls=1, scope="session"),
                retry_policy=RetryPolicy(max_attempts=3),
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    pending = checkpoint["pending_tool_approval"]
    assert pending["max_steps"] == max_steps
    assert pending["limits"]["max_tool_calls"] == 1
    assert pending["limits"]["scope"] == "session"
    assert pending["retry_policy"]["max_attempts"] == 3
    assert pending["budget_limits"] == []

    # Approving without overrides resumes under the ORIGINAL run's limits: the
    # approved tool call consumes the session's single allowed tool call, so
    # the follow-up echo round trips the limit instead of executing.
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending["approval_id"],
                tool_round_id=pending["tool_round_id"],
                tool_call_id=pending["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                max_steps=max_steps if restate_configuration else None,
                limits=(
                    RunLimits(max_tool_calls=1, scope="session") if restate_configuration else None
                ),
                retry_policy=(RetryPolicy(max_attempts=3) if restate_configuration else None),
            ),
        )
    )
    assert tool.calls == [{"value": "secret"}]
    limit_events = [event for event in events if event.type == EventType.SESSION_LIMIT_REACHED]
    assert len(limit_events) == 1
    assert limit_events[0].payload["limit"] == "tool_calls"
    session = asyncio.run(store.load(session_id))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


@pytest.mark.parametrize("change", [{"limits": RunLimits()}, {"max_steps": 10001}])
def test_cayu_app_tool_approval_rejects_explicit_limits_drift_before_dispatch(change):
    store, tool, app = _run_config_approval_app()
    session_id = "sess_approval_override_run_config"

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
                max_steps=10000,
                limits=RunLimits(max_tool_calls=1, scope="session"),
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    pending = checkpoint["pending_tool_approval"]

    # A changed limit is a changed finalization profile and must be rejected
    # before the approved tool or a follow-up provider request can run.
    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    **change,
                ),
            )
        )
    assert caught.value.changed_component_classes == (ExecutionProfileComponentClass.FINALIZATION,)
    assert tool.calls == []

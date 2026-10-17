"""Shared session-store approvals contract cases."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    interrupt_and_release_test_invocation,
)
from tests.core.session_store_conformance.support import (
    _approval_checkpoint,
    _approval_recovery_environment_spec,
    _ApprovalLimitProvider,
    _ApprovalMetadataTool,
    _ApprovalMixedRecoveryProvider,
    _ApprovalRecoveryProvider,
    _ApprovalRecoveryTool,
    _ApprovalResolutionBinding,
    _ChangingApprovalPolicy,
    _close_store,
    _identity,
    _open_store,
    _pending_approval_for_public_event,
    _private_event_for_public_event,
    _reopen_store,
    _SimulatedProcessLoss,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolCallApproval,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.budgets.run_limits import RunLimits
from cayu.context.base import (
    MessageWindowContextPolicy,
)
from cayu.environments.base import Environment
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.observability.events import EventSink
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._approval_support import _pending_approval_for_atomic_claim
from cayu.runtime._recovery_coordinator import (
    _checkpoint_with_legacy_approval_round,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_mutations import (
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import (
    ResumeRequest,
    RunRequest,
)
from cayu.tools.base import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.tools.policy import (
    AllowAllToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

pytestmark = pytest.mark.postgres


def test_session_store_conformance_stale_approval_cannot_claim_repaused_session(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"approval-repause-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            await store.update_status(session.id, SessionStatus.INTERRUPTED)
            checkpoint_a, approval_a = _approval_checkpoint("a")
            checkpoint_b, approval_b = _approval_checkpoint("b")
            await store.checkpoint(session.id, checkpoint_a)

            observed = _pending_approval_for_atomic_claim(
                await store.load_checkpoint(session.id),
                approval_id=approval_a.approval_id,
                tool_round_id=approval_a.tool_round_id,
                gating_tool_call_id=approval_a.tool_call_id,
                redactor=SecretRedactor(),
            )
            assert observed == approval_a

            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
            )
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
                checkpoint_transform=lambda _session, _checkpoint: checkpoint_b,
            )
            repaused = await store.load(session.id)
            assert repaused is not None
            repaused_epoch = repaused.run_epoch

            def stale_claim(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                _pending_approval_for_atomic_claim(
                    checkpoint,
                    approval_id=approval_a.approval_id,
                    tool_round_id=approval_a.tool_round_id,
                    gating_tool_call_id=approval_a.tool_call_id,
                    redactor=SecretRedactor(),
                )
                return checkpoint

            with pytest.raises(
                ValueError,
                match="identity does not match the current pending approval",
            ):
                await store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.INTERRUPTED},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=stale_claim,
                )

            after = await store.load(session.id)
            assert after is not None
            assert after.status is SessionStatus.INTERRUPTED
            assert after.run_epoch == repaused_epoch
            assert await store.load_checkpoint(session.id) == checkpoint_b
            assert (
                _pending_approval_for_atomic_claim(
                    await store.load_checkpoint(session.id),
                    approval_id=approval_b.approval_id,
                    tool_round_id=approval_b.tool_round_id,
                    gating_tool_call_id=approval_b.tool_call_id,
                    redactor=SecretRedactor(),
                )
                == approval_b
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_publication_is_atomic_across_restart(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"approval-publication-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            paired_checkpoint, approval = _approval_checkpoint("published")
            raw_round = pending_rounds.PendingToolRound(
                tool_round_id=approval.tool_round_id,
                model_step_id=approval.model_step_id,
                model_attempt_id=approval.model_attempt_id,
                agent_name="assistant",
                tool_calls=[
                    PendingToolCallApproval(
                        tool_call_id=approval.tool_call_id,
                        tool_name=approval.tool_name,
                        arguments=approval.arguments,
                    )
                ],
                policy_context_version=1,
            )
            raw_checkpoint = {
                pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY: (
                    raw_round.model_dump(mode="json")
                )
            }
            await store.checkpoint(session.id, raw_checkpoint)
            checkpoint_event = Event(
                id="approval-checkpointed",
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                payload={
                    "checkpoint": pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                    "tool_round_id": approval.tool_round_id,
                    "model_step_id": approval.model_step_id,
                    "model_attempt_id": approval.model_attempt_id,
                },
            )
            approval_event = Event(
                id="approval-requested",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=session.id,
                tool_name=approval.tool_name,
                payload={
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                    "tool_round_id": approval.tool_round_id,
                    "model_step_id": approval.model_step_id,
                    "model_attempt_id": approval.model_attempt_id,
                    "approval": approval.model_dump(mode="json"),
                },
            )
            await store.append_event(
                session.id,
                Event(
                    id=approval_event.id,
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                ),
            )

            with pytest.raises(ValueError, match="Event already exists for session"):
                await store.publish_checkpoint_and_events(
                    session.id,
                    checkpoint_transform=lambda _session, _checkpoint: paired_checkpoint,
                    events=[checkpoint_event, approval_event],
                )
            assert await store.load_checkpoint(session.id) == raw_checkpoint
            assert [event.id for event in await store.load_events(session.id)] == [
                approval_event.id
            ]

            replacement_approval_event = approval_event.model_copy(
                update={"id": "approval-requested-committed"},
                deep=True,
            )
            await store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=lambda _session, _checkpoint: paired_checkpoint,
                events=[checkpoint_event, replacement_approval_event],
            )
            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session.id) == paired_checkpoint
            assert [event.id for event in await store.load_events(session.id)] == [
                approval_event.id,
                checkpoint_event.id,
                replacement_approval_event.id,
            ]
            assert (
                _pending_approval_for_atomic_claim(
                    await store.load_checkpoint(session.id),
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    gating_tool_call_id=approval.tool_call_id,
                    redactor=SecretRedactor(),
                )
                == approval
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_legacy_approval_round_migrates_in_atomic_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    session_id=f"legacy-approval-claim-{session_store_case[0]}",
                    agent_name="assistant",
                    messages=[Message.text("user", "approve")],
                ),
                identity=_identity(),
            )
            await store.update_status(session.id, SessionStatus.INTERRUPTED)
            paired_checkpoint, approval = _approval_checkpoint("legacy")
            legacy_checkpoint = {
                pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: (
                    paired_checkpoint[pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY]
                )
            }
            await store.checkpoint(session.id, legacy_checkpoint)
            claimed_approval: PendingToolApproval | None = None

            def claim_legacy(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                nonlocal claimed_approval
                claimed_approval = _pending_approval_for_atomic_claim(
                    checkpoint,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    gating_tool_call_id=approval.tool_call_id,
                    redactor=SecretRedactor(),
                )
                return _checkpoint_with_legacy_approval_round(
                    checkpoint,
                    approval=claimed_approval,
                    redactor=SecretRedactor(),
                )

            claimed = await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_legacy,
            )
            assert claimed.status is SessionStatus.RUNNING
            assert claimed_approval == approval

            store = await _reopen_store(session_store_case, store)
            migrated = await store.load_checkpoint(session.id)
            assert pending_approval_reader.pending_approval_from_checkpoint(migrated) == approval
            migrated_round = pending_round_reader.pending_tool_round_from_checkpoint(migrated)
            assert migrated_round is not None
            assert migrated_round.policy_state == "planned"
            assert migrated_round.policy_context_version == 1
            assert migrated_round.tool_round_id == approval.tool_round_id
            assert migrated_round.tool_calls == approval.tool_calls
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "first_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["approve-then-deny", "deny-then-approve"],
)
def test_session_store_conformance_pre_digest_approval_claim_fails_closed(
    session_store_case,
    first_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        binding = _ApprovalResolutionBinding()
        tool_calls: list[dict[str, Any]] = []
        session_id = f"approval-decision-intent-{first_decision.value}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            first_events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            requested = next(
                event
                for event in first_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert binding.bind_calls == 1

            binding.fail_next = True
            first_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=first_decision,
            )
            first_request_digest = approval_support.approval_resolution_request_digest(
                first_request
            )
            failed_resolution = [
                event async for event in first_app.resolve_tool_approval(first_request)
            ]
            assert failed_resolution[0].type is EventType.INTERACTION_RESUMED
            assert any(event.type is EventType.SESSION_RESUMED for event in failed_resolution)
            assert failed_resolution[-1].type is EventType.SESSION_INTERRUPTED
            assert binding.bind_calls == 2
            assert tool_calls == []

            checkpoint = await store.load_checkpoint(session_id)
            intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=SecretRedactor(),
            )
            assert intent == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=first_request_digest,
                pause_resolved_at=intent.pause_resolved_at,
            )
            legacy_checkpoint = await store.load_checkpoint(session_id)
            assert legacy_checkpoint is not None
            legacy_intent = dict(
                legacy_checkpoint[pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY]
            )
            legacy_intent.pop("resolution_request_digest")
            await store.checkpoint(
                session_id,
                {
                    **legacy_checkpoint,
                    pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY: legacy_intent,
                },
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(session_store=store, enable_logging=False)
            retry_app.register_provider(
                _ApprovalRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            opposite = (
                ToolApprovalDecision.DENY
                if first_decision is ToolApprovalDecision.APPROVE
                else ToolApprovalDecision.APPROVE
            )
            opposite_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=opposite,
            )
            conflicting = [
                event async for event in retry_app.resolve_tool_approval(opposite_request)
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "already claimed with a different resolution decision"
                in conflicting[-1].payload["error"]
            )
            assert binding.bind_calls == 2
            assert tool_calls == []
            interrupted = await store.load(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTED
            assert pending_approval_reader.approval_resolution_intent_from_checkpoint(
                await store.load_checkpoint(session_id),
                redactor=SecretRedactor(),
            ) == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=None,
                pause_resolved_at=intent.pause_resolved_at,
            )

            legacy_retry = [event async for event in retry_app.resolve_tool_approval(first_request)]
            assert [event.type for event in legacy_retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert "predates exact resolution request identity" in legacy_retry[-1].payload["error"]
            assert binding.bind_calls == 2
            assert tool_calls == []
            final_checkpoint = await store.load_checkpoint(session_id)
            assert pending_approval_reader.approval_resolution_intent_from_checkpoint(
                final_checkpoint,
                redactor=SecretRedactor(),
            ) == approval_support.approval_resolution_intent_for(
                approval,
                decision=first_decision,
                resolution_request_digest=None,
                pause_resolved_at=intent.pause_resolved_at,
            )
            assert (
                pending_approval_reader.pending_approval_from_checkpoint(final_checkpoint)
                == approval
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_open_and_close_lost_ack_replay_exact_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-close-lost-ack-{session_store_case[0]}"
        try:
            original_publish = store.publish_runtime_publication
            lost_ack_kinds: set[str] = set()

            async def publish_then_lose_ack(session_id: str, **kwargs):
                result = await original_publish(session_id, **kwargs)
                kind = kwargs["request"].kind
                if kind in {"approval-open", "approval-close"} and kind not in lost_ack_kinds:
                    lost_ack_kinds.add(kind)
                    raise OSError(f"{kind} acknowledgement lost")
                return result

            store.publish_runtime_publication = publish_then_lose_ack  # type: ignore[method-assign]
            provider = _ApprovalRecoveryProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            assert "approval-open" in lost_ack_kinds
            open_receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-open:{approval.approval_id}",
            )
            assert open_receipt is not None
            assert open_receipt.kind == "approval-open"
            provider._complete_without_tools = True
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.DENY,
                reason="conformance denial",
            )
            resolved = [event async for event in app.resolve_tool_approval(request)]
            assert resolved[-1].type is EventType.SESSION_COMPLETED
            assert lost_ack_kinds == {"approval-open", "approval-close"}
            assert calls == []
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_tool_approval" not in checkpoint
            assert "pending_tool_round" not in checkpoint
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-close:{approval.approval_id}",
            )
            assert receipt is not None
            assert receipt.kind == "approval-close"

            replayed = [event async for event in app.resolve_tool_approval(request)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == receipt.appended_event_ids
            assert calls == []
            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        request.model_copy(update={"decision": ToolApprovalDecision.APPROVE})
                    )
                ]

            later_message = Message.text("user", "later recovered interaction input")
            later_interaction_id = "interaction-after-old-approval-receipt"
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_source_messages=[later_message],
                continued_interaction_id=later_interaction_id,
                defer_interaction_source=True,
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.release_run_fence(session_id)

            stale_replay = [event async for event in app.resolve_tool_approval(request)]
            private_stale_replay = [
                await _private_event_for_public_event(store, event) for event in stale_replay
            ]
            assert tuple(event.id for event in private_stale_replay) == receipt.appended_event_ids
            deferred = await store.load_deferred_interaction_input(session_id)
            assert deferred is not None
            assert deferred.interaction_id == later_interaction_id
            assert deferred.source_messages == [later_message]
            assert later_message not in await store.load_transcript(session_id)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_with_frozen_limit_replays_exact_request(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-limit-close-replay-{session_store_case[0]}"
        limits = RunLimits(max_tool_calls=1, scope="session")
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_ApprovalLimitProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                        limits=limits,
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approved within the configured limit",
                metadata={"ticket": "OPS-526"},
                resolved_by=ResolutionActor(subject="operator-1"),
                limits=limits,
            )

            closed = [event async for event in app.resolve_tool_approval(request)]
            assert EventType.SESSION_LIMIT_REACHED in [event.type for event in closed]
            assert calls == [{"value": "must remain gated"}]
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"approval-close:{approval.approval_id}",
            )
            assert receipt is not None
            assert receipt.intent["decision"] == "approve"
            assert receipt.intent["requested_decision"] == request.decision.value
            assert receipt.intent["resolution_request_digest"] == (
                runtime_publication_checkpoint_value_digest(
                    request.model_dump(
                        mode="json",
                        include={
                            "decision",
                            "reason",
                            "metadata",
                            "resolved_by",
                        },
                    )
                )
            )

            replayed = [event async for event in app.resolve_tool_approval(request)]
            private_replayed = [
                await _private_event_for_public_event(store, event) for event in replayed
            ]
            assert tuple(event.id for event in private_replayed) == receipt.appended_event_ids
            assert calls == [{"value": "must remain gated"}]

            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        request.model_copy(update={"decision": ToolApprovalDecision.DENY})
                    )
                ]
            assert calls == [{"value": "must remain gated"}]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_approval_event_ack_loss_rejects_request_drift(
    session_store_case,
) -> None:
    class FailFirstApprovedEventSink(EventSink):
        def __init__(self) -> None:
            self.failed = False

        async def emit(self, event: Event) -> None:
            if event.type is EventType.TOOL_CALL_APPROVED and not self.failed:
                self.failed = True
                raise _SimulatedProcessLoss()

    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-event-ack-loss-{session_store_case[0]}"
        redactor = SecretRedactor(["resolution", "digest"])
        try:
            original_append = store.append_event
            lost_append_ack = False

            async def append_then_lose_approved_ack(session_id: str, event: Event) -> None:
                nonlocal lost_append_ack
                await original_append(session_id, event)
                if event.type is EventType.TOOL_CALL_APPROVED and not lost_append_ack:
                    lost_append_ack = True
                    raise OSError("approval event append acknowledgement lost")

            store.append_event = append_then_lose_approved_ack  # type: ignore[method-assign]
            sink = FailFirstApprovedEventSink()
            provider = _ApprovalRecoveryProvider()
            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                secret_redactor=redactor,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approved under the original condition",
                metadata={
                    "oversized_audit_value": "x" * 20_000,
                    "condition": "limit-500",
                },
                resolved_by=ResolutionActor(
                    subject="operator-1",
                    claims={"approval_scope": "sensitive-scope"},
                ),
            )
            reordered_retry = request.model_copy(
                update={
                    "metadata": {
                        "condition": "limit-500",
                        "oversized_audit_value": "x" * 20_000,
                    }
                }
            )
            assert approval_support.approval_resolution_request_digest(
                reordered_retry
            ) == approval_support.approval_resolution_request_digest(request)

            with pytest.raises(_SimulatedProcessLoss):
                _ = [event async for event in app.resolve_tool_approval(request)]
            assert lost_append_ack is True
            assert sink.failed is True
            assert calls == []

            store = await _reopen_store(session_store_case, store)
            retry_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            retry_app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            retry_app.register_provider(retry_provider, default=True)
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            recovered = await retry_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert recovered.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)
            checkpoint = await store.load_checkpoint(session_id)
            intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                checkpoint,
                redactor=redactor,
            )
            assert intent is not None
            assert intent.resolution_request_digest == (
                approval_support.approval_resolution_request_digest(request)
            )
            approved_events = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVED
            ]
            assert len(approved_events) == 1
            approved_event_id = approved_events[0].id
            assert "resolution_request_digest" not in approved_events[0].payload
            assert approved_events[0].payload["resolved_by"] == {
                "subject": "operator-1",
                "tenant": None,
                "source": None,
            }
            assert "claims" not in approved_events[0].payload["resolved_by"]

            conflicting = [
                event
                async for event in retry_app.resolve_tool_approval(
                    reordered_retry.model_copy(
                        update={
                            "metadata": {
                                "condition": "limit-5000",
                                "oversized_audit_value": "x" * 20_000,
                            },
                            "resolved_by": ResolutionActor(subject="operator-2"),
                        }
                    )
                )
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert "different" in conflicting[-1].payload["error"]
            assert "request" in conflicting[-1].payload["error"]
            assert calls == []
            still_interrupted = await store.load(session_id)
            assert still_interrupted is not None
            assert still_interrupted.status is SessionStatus.INTERRUPTED

            completed = [event async for event in retry_app.resolve_tool_approval(reordered_retry)]
            assert completed[-1].type is EventType.SESSION_COMPLETED
            assert calls == [
                {
                    "arguments": {"value": "must remain gated"},
                    "condition": "limit-500",
                }
            ]
            approved_events = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVED
            ]
            assert len(approved_events) == 1
            assert approved_events[0].id == approved_event_id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "historical_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["historical-approve", "historical-deny"],
)
def test_session_store_conformance_legacy_history_cannot_be_poisoned_by_retry(
    session_store_case,
    historical_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        binding = _ApprovalResolutionBinding()
        tool_calls: list[dict[str, Any]] = []
        session_id = f"approval-legacy-history-{historical_decision.value}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            first_events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the protected tool")],
                    )
                )
            ]
            requested = next(
                event
                for event in first_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            historical_payload: dict[str, Any] = {
                "model_step_id": approval.model_step_id,
                "model_attempt_id": approval.model_attempt_id,
                "tool_round_id": approval.tool_round_id,
                "approval_id": approval.approval_id,
                "tool_call_id": approval.tool_call_id,
            }
            historical_type = (
                EventType.TOOL_CALL_APPROVED
                if historical_decision is ToolApprovalDecision.APPROVE
                else EventType.TOOL_CALL_APPROVAL_DENIED
            )
            if historical_decision is ToolApprovalDecision.APPROVE:
                historical_payload.update(
                    {
                        "reason": "legacy approval reason",
                        "metadata": {"condition": "legacy-approved"},
                        "resolved_by": {
                            "subject": "legacy-operator",
                            "tenant": None,
                            "source": None,
                        },
                    }
                )
            else:
                historical_payload.update(
                    {
                        "approval_required": True,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session_id,
                            tool_round_id=approval.tool_round_id,
                            tool_call_id=approval.tool_call_id,
                            approval_id=approval.approval_id,
                        ),
                        "result": ToolResult(
                            content="historically denied",
                            is_error=True,
                        ).model_dump(mode="json"),
                    }
                )
            paused_session = await store.load(session_id)
            assert paused_session is not None
            await store.append_event(
                session_id,
                approval_support.resumed_event(
                    session=paused_session,
                    agent_name="assistant",
                    environment_name="approval-environment",
                    approval=approval,
                    decision=historical_decision,
                    release_identity=None,
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    type=historical_type,
                    session_id=session_id,
                    agent_name="assistant",
                    environment_name="approval-environment",
                    tool_name=approval.tool_name,
                    payload=historical_payload,
                ),
            )
            assert (
                pending_approval_reader.approval_resolution_intent_from_checkpoint(
                    await store.load_checkpoint(session_id),
                    redactor=SecretRedactor(),
                )
                is None
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(session_store=store, enable_logging=False)
            retry_app.register_provider(
                _ApprovalRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    binding=binding,
                ),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy([]),
            )
            matching_request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=historical_decision,
                reason=(
                    "legacy approval reason"
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else None
                ),
                metadata=(
                    {"condition": "legacy-approved"}
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else {}
                ),
                resolved_by=(
                    ResolutionActor(subject="legacy-operator")
                    if historical_decision is ToolApprovalDecision.APPROVE
                    else None
                ),
            )
            if historical_decision is ToolApprovalDecision.APPROVE:
                drifted = [
                    event
                    async for event in retry_app.resolve_tool_approval(
                        matching_request.model_copy(
                            update={
                                "metadata": {"condition": "changed"},
                                "resolved_by": ResolutionActor(subject="different-operator"),
                            }
                        )
                    )
                ]
                assert drifted[-1].type is EventType.SESSION_INTERRUPTED
                assert (
                    "prior durable resolution activity has no exact resolution request identity"
                    in drifted[-1].payload["error"]
                )
                assert tool_calls == []
                assert (
                    pending_approval_reader.approval_resolution_intent_from_checkpoint(
                        await store.load_checkpoint(session_id),
                        redactor=SecretRedactor(),
                    )
                    is None
                )

            conflicting_decision = (
                ToolApprovalDecision.DENY
                if historical_decision is ToolApprovalDecision.APPROVE
                else ToolApprovalDecision.APPROVE
            )
            conflicting = [
                event
                async for event in retry_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=conflicting_decision,
                    )
                )
            ]
            assert [event.type for event in conflicting] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert binding.bind_calls == 1
            assert tool_calls == []
            assert (
                pending_approval_reader.approval_resolution_intent_from_checkpoint(
                    await store.load_checkpoint(session_id),
                    redactor=SecretRedactor(),
                )
                is None
            )

            legacy_retry = [
                event async for event in retry_app.resolve_tool_approval(matching_request)
            ]
            assert [event.type for event in legacy_retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "prior durable resolution activity has no exact resolution request identity"
                in legacy_retry[-1].payload["error"]
            )
            assert binding.bind_calls == 1
            assert tool_calls == []
            final_checkpoint = await store.load_checkpoint(session_id)
            assert (
                pending_approval_reader.approval_resolution_intent_from_checkpoint(
                    final_checkpoint,
                    redactor=SecretRedactor(),
                )
                is None
            )
            assert (
                pending_approval_reader.pending_approval_from_checkpoint(final_checkpoint)
                == approval
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lossy_legacy_grant_cannot_authorize_pending_sibling(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        calls: list[dict[str, Any]] = []
        session_id = f"approval-lossy-legacy-mixed-{session_store_case[0]}"
        original_secret = "legacy-resolution-secret-original"
        changed_secret = "legacy-resolution-secret-changed"
        redactor = SecretRedactor([original_secret, changed_secret])
        try:
            provider = _ApprovalMixedRecoveryProvider()
            policy_calls: list[ToolPolicyDecision] = []
            app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run both protected tools")],
                    )
                )
            ]
            approval = await _pending_approval_for_public_event(
                store,
                next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                ),
            )
            assert [call.policy_decision for call in approval.tool_calls] == [
                ToolPolicyDecision.REQUIRE_APPROVAL.value,
                ToolPolicyDecision.ALLOW.value,
            ]
            assert calls == []

            paused_session = await store.load(session_id)
            assert paused_session is not None
            await store.append_event(
                session_id,
                approval_support.resumed_event(
                    session=paused_session,
                    agent_name="assistant",
                    environment_name=None,
                    approval=approval,
                    decision=ToolApprovalDecision.APPROVE,
                    release_identity=None,
                    resolved_by=ResolutionActor(subject="legacy-operator"),
                ),
            )
            bounded_metadata = approval_support.bounded_resolution_metadata_payload(
                {
                    "credential": original_secret,
                    "large": "x" * 20_000,
                },
                redactor=redactor,
            )
            assert bounded_metadata["metadata"]["credential"] == REDACTED_SECRET
            assert bounded_metadata["metadata_truncated"] is True
            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_APPROVED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name=approval.tool_calls[0].tool_name,
                    payload={
                        "model_step_id": approval.model_step_id,
                        "model_attempt_id": approval.model_attempt_id,
                        "tool_round_id": approval.tool_round_id,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_calls[0].tool_call_id,
                        "reason": "legacy approval",
                        **bounded_metadata,
                        "resolved_by": {
                            "subject": "legacy-operator",
                            "tenant": None,
                            "source": None,
                        },
                    },
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name=approval.tool_calls[0].tool_name,
                    payload={
                        "model_step_id": approval.model_step_id,
                        "model_attempt_id": approval.model_attempt_id,
                        "tool_round_id": approval.tool_round_id,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_calls[0].tool_call_id,
                        "result": ToolResult(content="historically completed").model_dump(
                            mode="json"
                        ),
                    },
                ),
            )

            store = await _reopen_store(session_store_case, store)
            retry_app = CayuApp(
                session_store=store,
                secret_redactor=redactor,
                enable_logging=False,
            )
            retry_app.register_provider(
                _ApprovalMixedRecoveryProvider(complete_without_tools=True),
                default=True,
            )
            retry_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalMetadataTool(calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            retry = [
                event
                async for event in retry_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                        reason="legacy approval",
                        metadata={
                            "credential": changed_secret,
                            "large": "x" * 20_000,
                        },
                        resolved_by=ResolutionActor(subject="legacy-operator"),
                    )
                )
            ]
            assert [event.type for event in retry] == [
                EventType.INTERACTION_RESUMED,
                EventType.SESSION_INTERRUPTED,
            ]
            assert (
                "prior durable resolution activity has no exact resolution request identity"
                in retry[-1].payload["error"]
            )
            assert calls == []
            checkpoint = await store.load_checkpoint(session_id)
            assert (
                pending_approval_reader.pending_approval_from_checkpoint(
                    checkpoint,
                    redactor=redactor,
                )
                == approval
            )
            assert (
                pending_approval_reader.approval_resolution_intent_from_checkpoint(
                    checkpoint,
                    redactor=redactor,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "legacy_unversioned",
    [False, True],
    ids=["versioned-raw-round", "legacy-unversioned-raw-round"],
)
@pytest.mark.parametrize(
    "resolution_decision",
    [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY],
    ids=["approve", "deny"],
)
def test_session_store_conformance_ambiguous_policy_recovery_remains_gated(
    session_store_case,
    legacy_unversioned: bool,
    resolution_decision: ToolApprovalDecision,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        policy_calls: list[ToolPolicyDecision] = []
        tool_calls: list[dict[str, Any]] = []
        session_id = (
            f"ambiguous-policy-{legacy_unversioned}-"
            f"{resolution_decision.value}-{session_store_case[0]}"
        )
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )

            async def lose_approval_publication(**_kwargs) -> None:
                raise _SimulatedProcessLoss(
                    "process stopped after policy evaluation and before publication"
                )

            first_app._tool_round_executor.invocation.admission.pause_for_approval = (
                lose_approval_publication
            )
            with pytest.raises(
                _SimulatedProcessLoss,
                match="after policy evaluation",
            ):
                async for _event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "use the protected tool")],
                    )
                ):
                    pass

            assert policy_calls == [ToolPolicyDecision.REQUIRE_APPROVAL]
            raw_checkpoint = await store.load_checkpoint(session_id)
            assert raw_checkpoint is not None
            assert "pending_tool_approval" not in raw_checkpoint
            raw_round = dict(raw_checkpoint["pending_tool_round"])
            assert raw_round["policy_state"] == "unplanned"
            if legacy_unversioned:
                raw_round.pop("policy_state")
                raw_round.pop("policy_context_version")
                raw_checkpoint = dict(raw_checkpoint)
                raw_checkpoint["pending_tool_round"] = raw_round
                await store.checkpoint(session_id, raw_checkpoint)
            await interrupt_and_release_test_invocation(store, session_id)

            store = await _reopen_store(session_store_case, store)
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            deferred_message = Message.text(
                "user",
                "continue after the recovered approval",
            )
            resume_events = [
                event
                async for event in recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[deferred_message],
                    )
                )
            ]

            assert policy_calls == [ToolPolicyDecision.REQUIRE_APPROVAL]
            assert tool_calls == []
            assert recovery_provider.requests == []
            assert any(
                event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in resume_events
            )
            recovered_session = await store.load(session_id)
            assert recovered_session is not None
            assert recovered_session.status is SessionStatus.INTERRUPTED
            recovered_checkpoint = await store.load_checkpoint(session_id)
            assert recovered_checkpoint is not None
            approval = PendingToolApproval.model_validate(
                recovered_checkpoint["pending_tool_approval"]
            )
            planned_round = pending_rounds.PendingToolRound.model_validate(
                recovered_checkpoint["pending_tool_round"]
            )
            assert planned_round.policy_state == "planned"
            assert planned_round.policy_context_version == 1
            assert planned_round.deferred_messages == [deferred_message]
            assert approval.tool_round_id == planned_round.tool_round_id
            assert approval.tool_call_id == "call_policy_recovery"
            assert approval.tool_calls[0].policy_evidence == "ambiguous"
            assert approval.tool_calls[0].policy_decision is None
            assert approval.metadata == {
                "recovered": True,
                "policy_evaluation": "ambiguous",
            }
            requested = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            ]
            assert len(requested) == 1
            assert requested[0].payload["approval_id"] == approval.approval_id
            assert requested[0].payload["recovered"] is True

            resolution_events = [
                event
                async for event in recovery_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=resolution_decision,
                    )
                )
            ]
            assert tool_calls == []
            if resolution_decision is ToolApprovalDecision.APPROVE:
                ambiguous_blocks = [
                    event
                    for event in resolution_events
                    if event.type is EventType.TOOL_CALL_BLOCKED
                    and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
                ]
                assert len(ambiguous_blocks) == 1
                assert (
                    ambiguous_blocks[0].payload["requested_decision"]
                    == ToolApprovalDecision.APPROVE.value
                )
            assert resolution_events[-1].type is EventType.SESSION_COMPLETED
            assert len(recovery_provider.requests) == 1
            assert [message.role.value for message in recovery_provider.requests[0].messages] == [
                "user",
                "assistant",
                "tool",
                "user",
            ]
            assert recovery_provider.requests[0].messages[-1] == deferred_message
            resolved_session = await store.load(session_id)
            assert resolved_session is not None
            assert resolved_session.status is SessionStatus.COMPLETED
            resolved_transcript = await store.load_transcript(session_id)
            assert [message.role.value for message in resolved_transcript] == [
                "user",
                "assistant",
                "tool",
                "user",
                "assistant",
            ]
            assert resolved_transcript.count(deferred_message) == 1
            assert await store.load_deferred_interaction_input(session_id) is None
            resolved_checkpoint = await store.load_checkpoint(session_id)
            assert "pending_tool_approval" not in (resolved_checkpoint or {})
            assert "pending_tool_round" not in (resolved_checkpoint or {})
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "lost_outcome",
    ["deny", "exception"],
    ids=["lost-deny", "lost-policy-exception"],
)
def test_session_store_conformance_lost_policy_authority_never_becomes_executable(
    session_store_case,
    lost_outcome: str,
) -> None:
    """A missing durable outcome is not authorization, even after approval."""

    class LostOutcomePolicy(ToolPolicy):
        def __init__(self, calls: list[str]) -> None:
            self._calls = calls

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name=f"tests:lost-outcome-policy:{lost_outcome}",
                behavior_version="1",
                implementation_version="1",
            )

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            self._calls.append(lost_outcome)
            if lost_outcome == "exception":
                raise RuntimeError("policy backend failed before durable publication")
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="hard denial that must never be downgraded",
            )

    async def run() -> None:
        store = await _open_store(session_store_case)
        policy_calls: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        session_id = f"lost-{lost_outcome}-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(_ApprovalRecoveryProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=LostOutcomePolicy(policy_calls),
            )

            if lost_outcome == "deny":

                async def lose_policy_plan_publication(**_kwargs) -> None:
                    raise _SimulatedProcessLoss(
                        "process stopped after denial and before publication"
                    )

                first_app._tool_invocation.admission.checkpoint_tool_round_policy_plan = (
                    lose_policy_plan_publication
                )
                with pytest.raises(_SimulatedProcessLoss, match="after denial"):
                    async for _event in first_app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "use the protected tool")],
                        )
                    ):
                        pass
                await interrupt_and_release_test_invocation(store, session_id)
            else:
                events = [
                    event
                    async for event in first_app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="assistant",
                            messages=[Message.text("user", "use the protected tool")],
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_FAILED

            raw_checkpoint = await store.load_checkpoint(session_id)
            assert raw_checkpoint is not None
            assert "pending_tool_approval" not in raw_checkpoint
            assert raw_checkpoint["pending_tool_round"]["policy_state"] == "unplanned"
            assert policy_calls == [lost_outcome]
            assert tool_calls == []

            store = await _reopen_store(session_store_case, store)
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=LostOutcomePolicy(policy_calls),
            )

            resume_events = [
                event
                async for event in recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "recover safely")],
                    )
                )
            ]
            requested = next(
                event
                for event in resume_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert approval.tool_calls[0].policy_evidence == "ambiguous"
            assert approval.tool_calls[0].policy_decision is None

            resolution_events = [
                event
                async for event in recovery_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert policy_calls == [lost_outcome]
            assert tool_calls == []
            ambiguous_block = next(
                event for event in resolution_events if event.type is EventType.TOOL_CALL_BLOCKED
            )
            assert ambiguous_block.payload["blocked_by"] == "policy_evaluation_ambiguous"
            assert not any(
                event.type in {EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_STARTED}
                for event in resolution_events
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_registration_drift_rejects_paused_call(
    session_store_case,
) -> None:
    class MixedProvider(ModelProvider):
        name = "fake"

        def __init__(self, *, complete_without_tools: bool = False) -> None:
            self._complete_without_tools = complete_without_tools
            self.requests: list[ModelRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:session-store-conformance:registration-drift-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if self._complete_without_tools:
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            yield ModelStreamEvent.tool_call(
                id="call_late",
                name="late_effect",
                arguments={"value": "must not execute"},
            )
            yield ModelStreamEvent.tool_call(
                id="call_approval",
                name="stateful_effect",
                arguments={"value": "approved effect"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    class LateEffectTool(Tool):
        spec = ToolSpec(
            name="late_effect",
            description="A tool registered only after the round paused.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self, calls: list[dict[str, Any]]) -> None:
            super().__init__()
            self._calls = calls

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self._calls.append(dict(args))
            return ToolResult(content="unexpected")

    async def run() -> None:
        store = await _open_store(session_store_case)
        protected_calls: list[dict[str, Any]] = []
        late_calls: list[dict[str, Any]] = []
        policy_calls: list[ToolPolicyDecision] = []
        session_id = f"registration-drift-{session_store_case[0]}"
        try:
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(MixedProvider(), default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(protected_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
            )
            events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "run the mixed round")],
                    )
                )
            ]
            requested = next(
                event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            assert [call.policy_evidence for call in approval.tool_calls] == [
                "unregistered",
                "authoritative",
            ]

            store = await _reopen_store(session_store_case, store)
            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_provider = MixedProvider(complete_without_tools=True)
            resumed_app.register_provider(resumed_provider, default=True)
            resumed_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    LateEffectTool(late_calls),
                    _ApprovalRecoveryTool(protected_calls),
                ],
                tool_policy=AllowAllToolPolicy(),
            )
            with pytest.raises(ExecutionProfileMismatchError) as raised:
                _ = [
                    event
                    async for event in resumed_app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=approval.approval_id,
                            tool_round_id=approval.tool_round_id,
                            tool_call_id=approval.tool_call_id,
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]

                assert raised.value.changed_component_classes == (
                    ExecutionProfileComponentClass.DIRECT_TOOLS,
                    ExecutionProfileComponentClass.EFFECT_AUTHORITY,
                    ExecutionProfileComponentClass.EXECUTION_POLICIES,
                    ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
                )
                current = await store.load(session_id)
                assert current is not None
                assert current.tool_capability_ceiling == ToolCapabilityCeiling(
                    tool_names=("stateful_effect",)
                )
            assert late_calls == []
            assert protected_calls == []
            assert resumed_provider.requests == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_context_profile_drift_after_reopen(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = f"context-profile-reopen-{session_store_case[0]}"
        try:
            first_provider = ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="fake",
            )
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(first_provider, default=True)
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=MessageWindowContextPolicy(max_messages=4),
            )
            _ = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "first")],
                    )
                )
            ]

            store = await _reopen_store(session_store_case, store)
            replacement_provider = ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="fake",
            )
            replacement_app = CayuApp(session_store=store, enable_logging=False)
            replacement_app.register_provider(replacement_provider, default=True)
            replacement_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=MessageWindowContextPolicy(max_messages=5),
            )

            with pytest.raises(ExecutionProfileMismatchError) as raised:
                _ = [
                    event
                    async for event in replacement_app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "second")],
                        )
                    )
                ]

            assert raised.value.changed_component_classes == (
                ExecutionProfileComponentClass.CONTEXT_SELECTION,
            )
            assert replacement_provider.requests == []
        finally:
            await _close_store(store)

    asyncio.run(run())

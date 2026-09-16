from __future__ import annotations

import asyncio
import hashlib

import pytest

from cayu import AgentSpec, Message
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ContextRequest,
    _estimate_model_facing_context_pressure,
    copy_context_compaction_telemetry,
)
from cayu.events import EventType
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity


class GrowingSummary(ContextCompactor):
    def __init__(self, size=800):
        self.size = size
        self.requests = []

    def provider_budget_identity(self, _session):
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        return CompactionResult(
            summary="s" * self.size,
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode()).hexdigest()
                if request.existing_summary is not None
                else None
            ),
        )


@pytest.mark.parametrize("previous_summary", [None, "Short existing summary."])
def test_summary_growth_headroom_keeps_full_atomic_suffix_within_actual_target(previous_summary):
    async def exercise():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        messages = [
            Message.text("system", "Follow the policy."),
            Message.text("user", "Original task."),
            Message.tool_call(tool_call_id="old", tool_name="read", arguments={}),
            Message.tool_result(tool_call_id="old", tool_name="read", content="x" * 10000),
            Message.tool_call(tool_call_id="middle", tool_name="read", arguments={}),
            Message.tool_result(tool_call_id="middle", tool_name="read", content="y" * 1000),
            Message.tool_call(tool_call_id="recent", tool_name="read", arguments={}),
            Message.tool_result(tool_call_id="recent", tool_name="read", content="z" * 1000),
            Message.text("user", "Continue the original task."),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake"),
            messages=messages,
            step=4,
        )
        prefix = CheckpointCompactionContextPolicy().summary_prefix
        candidate = [
            *messages[:2],
            Message.text("user", f"{prefix}\n{previous_summary or 'Compacted prior context.'}"),
            *messages[4:],
        ]
        target = (
            _estimate_model_facing_context_pressure(
                request=request, messages=candidate
            ).estimated_context_input_tokens
            + 1
        )
        checkpoint = (
            None
            if previous_summary is None
            else {
                "context_compaction": {
                    "version": 2,
                    "summary": previous_summary,
                    "compacted_transcript_cursor": 2,
                }
            }
        )
        config = {
            "compact_after_estimated_context_tokens": target + 500,
            "max_recent_context_tokens": target,
        }
        original = CheckpointCompactionContextPolicy(compactor=GrowingSummary(), **config)
        with pytest.raises(ContextBuildError, match="within the configured size bounds") as strict:
            await original.build_with_checkpoint(request, checkpoint=checkpoint)
        best_effort_compactor = GrowingSummary()
        best_effort = CheckpointCompactionContextPolicy(
            compactor=best_effort_compactor, enforce_recent_context_target=False, **config
        )
        accepted = await best_effort.build_with_checkpoint(request, checkpoint=checkpoint)
        assert len(best_effort_compactor.requests) == 1
        assert accepted.checkpoint == strict.value.checkpoint
        pressure = _estimate_model_facing_context_pressure(
            request=request, messages=accepted.messages
        )
        assert target < pressure.estimated_context_input_tokens < target + 500
        completed = [
            item.payload
            for item in accepted.compaction_telemetry
            if item.event_type is EventType.CONTEXT_COMPACTION_COMPLETED
        ]
        assert len(completed) == 1
        assert completed[0]["retained_target_enforced"] is False
        assert completed[0]["retained_target_met"] is False
        assert completed[0]["estimated_window_within_trigger"] is True
        for event in accepted.compaction_telemetry:
            if event.event_type is EventType.CONTEXT_COMPACTION_COMPLETED:
                assert copy_context_compaction_telemetry(event).payload == event.payload
        assert "y" * 1000 in str(accepted.messages)
        assert "z" * 1000 in str(accepted.messages)
        restarted_compactor = GrowingSummary()
        restarted = await CheckpointCompactionContextPolicy(
            compactor=restarted_compactor, enforce_recent_context_target=False, **config
        ).build_with_checkpoint(request, checkpoint=accepted.checkpoint)
        assert restarted.messages == accepted.messages
        assert restarted_compactor.requests == []
        compactor = GrowingSummary()
        policy = CheckpointCompactionContextPolicy(
            compactor=compactor, reserved_summary_tokens=250, **config
        )
        result = await policy.build_with_checkpoint(request, checkpoint=checkpoint)
        assert len(compactor.requests) == 1  # No extra model retry is needed.
        assert compactor.requests[0].existing_summary == previous_summary
        assert (
            _estimate_model_facing_context_pressure(
                request=request, messages=result.messages
            ).estimated_context_input_tokens
            <= target
        )
        serialized = str([m.model_dump(mode="json") for m in result.messages])
        assert "z" * 1000 in serialized
        assert "y" * 1000 not in serialized
        assert "Original task." in serialized
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 6
        # Reservation is not permission to weaken final bounds or to truncate
        # a generated summary. Even a best-effort suffix must retain the original
        # target check when fixed overhead itself can fit that target.
        too_large = CheckpointCompactionContextPolicy(
            compactor=GrowingSummary(size=(target + 100) * 4),
            reserved_summary_tokens=target - 1,
            **config,
        )
        with pytest.raises(ContextBuildError, match="within the configured size bounds") as failed:
            await too_large.build_with_checkpoint(request, checkpoint=checkpoint)
        assert failed.value.checkpoint["context_compaction"]["summary"]

    asyncio.run(exercise())


@pytest.mark.parametrize("reserve", [-1, True, 1.5, "100", 700])
def test_invalid_summary_headroom_is_rejected(reserve):
    with pytest.raises((TypeError, ValueError)):
        CheckpointCompactionContextPolicy(
            compact_after_estimated_context_tokens=1000,
            max_recent_context_tokens=700,
            reserved_summary_tokens=reserve,
        )


def test_summary_headroom_requires_size_based_selection():
    with pytest.raises(ValueError, match="size-based"):
        CheckpointCompactionContextPolicy(reserved_summary_tokens=10)


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_best_effort_target_requires_an_explicit_boolean(value):
    with pytest.raises(TypeError, match="must be a boolean"):
        CheckpointCompactionContextPolicy(enforce_recent_context_target=value)


def test_best_effort_target_requires_size_based_selection():
    with pytest.raises(ValueError, match="size-based"):
        CheckpointCompactionContextPolicy(enforce_recent_context_target=False)

"""Checkpoint compaction contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.runtime.runtime_compaction_support import RecordingCompactor
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    _private_events_for_public_events,
    _test_session,
    checkpoint_without_active_invocation_profile,
    checkpoint_without_model_step_publication,
    collect_events,
    collect_resume_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ContextRequest,
    ModelCompactor,
)
from cayu.events import EventType
from cayu.messages import Message, MessageRole
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest


class FailingCompactor(ContextCompactor):
    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        raise RuntimeError("compaction unavailable")


def test_checkpoint_policy_rejects_unmarked_zero_coverage_result():
    class ZeroCoverageCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            del request
            return CompactionResult(
                summary="summary without represented source",
                covered_message_count=0,
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=ZeroCoverageCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )

    with pytest.raises(
        ContextBuildError,
        match="zero coverage must report progress_exhausted=true",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)
    assert exc_info.value.checkpoint is None
    assert [item.event_type for item in exc_info.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]


def test_checkpoint_policy_rejects_zero_coverage_summary_replacement():
    class ReplacingExhaustedCompactor(ContextCompactor):
        def _progress_key(self) -> str:
            return "replace-exhausted:v1"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            assert request.existing_summary == "OLD_HISTORY_FACT"
            return CompactionResult(
                summary="replacement that omits the old history",
                covered_message_count=0,
                source_chunk_count=0,
                progress_exhausted=True,
                progress_key="replace-exhausted:v1",
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=ReplacingExhaustedCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )
    checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": "OLD_HISTORY_FACT",
            "compacted_transcript_cursor": 2,
            "metadata": {"source": "seeded"},
        }
    }

    with pytest.raises(
        ContextBuildError,
        match="zero-coverage compaction must preserve the existing summary",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "already covered request"),
                        Message.text("assistant", "already covered answer"),
                        Message.text("user", "new compactable request"),
                        Message.text("assistant", "new compactable answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=checkpoint,
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)
    assert checkpoint["context_compaction"] == {
        "version": 2,
        "summary": "OLD_HISTORY_FACT",
        "compacted_transcript_cursor": 2,
        "metadata": {"source": "seeded"},
    }


@pytest.mark.parametrize(
    "represented_existing_summary_sha256",
    [None, "0" * 64],
    ids=["missing", "wrong"],
)
def test_checkpoint_policy_rejects_unbound_existing_summary_replacement(
    represented_existing_summary_sha256,
):
    class UnboundReplacementCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            assert request.existing_summary == "OLD_HISTORY_FACT"
            return CompactionResult(
                summary="new context without a prior-summary binding",
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(represented_existing_summary_sha256),
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=UnboundReplacementCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )

    with pytest.raises(
        ContextBuildError,
        match="must bind represented_existing_summary_sha256",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "already covered request"),
                        Message.text("assistant", "already covered answer"),
                        Message.text("user", "new compactable request"),
                        Message.text("assistant", "new compactable answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint={
                    "context_compaction": {
                        "version": 2,
                        "summary": "OLD_HISTORY_FACT",
                        "compacted_transcript_cursor": 2,
                        "metadata": {"source": "seeded"},
                    }
                },
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)


def test_checkpoint_policy_rejects_compactor_coverage_that_splits_tool_round():
    class SplitToolRoundCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(summary="invalid split", covered_message_count=2)

    policy = CheckpointCompactionContextPolicy(
        compactor=SplitToolRoundCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_split",
            tool_name="inspect",
            arguments={"path": "report.txt"},
        ),
        Message.tool_result(
            tool_call_id="call_split",
            tool_name="inspect",
            content="done",
        ),
        Message.text("user", "current"),
    ]
    with pytest.raises(ContextBuildError, match="splits an assistant/tool round") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=messages,
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )
    assert isinstance(exc_info.value.cause, ValueError)
    assert exc_info.value.compaction_telemetry[-1].event_type == (
        EventType.CONTEXT_COMPACTION_FAILED
    )
    assert exc_info.value.compaction_telemetry[-1].payload["represented_source_end"] == 0


def test_checkpoint_policy_discards_legacy_cursor_that_splits_tool_round():
    class CaptureCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.request: CompactionRequest | None = None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.request = request
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
            )

    compactor = CaptureCompactor()
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_legacy_split",
            tool_name="inspect",
            arguments={"path": "report.txt"},
        ),
        Message.tool_result(
            tool_call_id="call_legacy_split",
            tool_name="inspect",
            content="done",
        ),
        Message.text("user", "current"),
    ]
    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint={
                "context_compaction": {
                    "version": 2,
                    "summary": "legacy invalid summary",
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )
    assert compactor.request is not None
    assert compactor.request.existing_summary is None
    assert compactor.request.messages == messages[:3]
    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 3


def test_checkpoint_policy_invalidates_version_one_checkpoint_with_valid_boundary():
    class CaptureCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.request: CompactionRequest | None = None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.request = request
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
            )

    compactor = CaptureCompactor()
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "legacy source may have been clipped"),
        Message.text("assistant", "legacy answer"),
        Message.text("user", "current"),
    ]
    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint={
                "context_compaction": {
                    "version": 1,
                    "summary": "legacy potentially incomplete summary",
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )

    assert compactor.request is not None
    assert compactor.request.existing_summary is None
    assert compactor.request.messages == messages[:2]
    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["version"] == 2
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2


def test_checkpoint_compaction_passes_context_messages_to_compactor():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        ),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_ctx_messages",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert any(event.type == EventType.SESSION_COMPLETED for event in first_events)

    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_ctx_messages",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    assert any(event.type == EventType.SESSION_COMPLETED for event in second_events)

    assert len(compactor.requests) >= 1
    last_request = compactor.requests[-1]
    assert len(last_request.context_messages) > 0
    context_roles = [m.role for m in last_request.context_messages]
    assert MessageRole.SYSTEM in context_roles
    assert MessageRole.USER in context_roles


def test_cayu_app_checkpoint_compacts_model_context_without_rewriting_transcript():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        ),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction",
                messages=[
                    Message.text("user", "old one"),
                    Message.text("assistant", "old answer one"),
                    Message.text("user", "old two"),
                    Message.text("assistant", "old answer two"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "requested_source_start": 1,
        "requested_source_end": 5,
        "represented_source_start": 1,
        "represented_source_end": 1,
        "represented_message_count": 0,
        "coverage_mode": "pending",
        "chunk_count": 0,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "summary_chars": len("old one|old answer one|old two|old answer two"),
        "requested_source_start": 1,
        "requested_source_end": 5,
        "represented_source_start": 1,
        "represented_source_end": 5,
        "represented_message_count": 4,
        "coverage_mode": "full",
        "chunk_count": 1,
        "chunk_mode": "single_request",
        "bounded_input": False,
        "compaction_failed": False,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert "summary" not in events[2].payload
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old one",
        "old answer one",
        "old two",
        "old answer two",
    ]

    provider_context = provider.requests[0].messages
    assert [message.role for message in provider_context] == [
        "system",
        "user",
        "user",
    ]
    assert provider_context[1].content[0].text == (
        "Previous session context summary:\nold one|old answer one|old two|old answer two"
    )
    assert provider_context[2].content[0].text == "current"

    transcript = asyncio.run(store.load_transcript("sess_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "You are careful.",
        "old one",
        "old answer one",
        "old two",
        "old answer two",
        "current",
        "final answer",
    ]
    checkpoint = asyncio.run(store.load_checkpoint("sess_compaction"))
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "old one|old answer one|old two|old answer two",
            "compacted_transcript_cursor": 5,
            "metadata": {"request_count": 1},
        },
    }


def test_cayu_app_checkpoint_compaction_can_use_model_compactor():
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                options={"anthropic": {"max_tokens": 512}},
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor_provider.requests) == 1
    assert compactor_provider.requests[0].model == "summary-model"
    assert compactor_provider.requests[0].tools == []
    assert compactor_provider.requests[0].options == {"anthropic": {"max_tokens": 512}}
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 0,
        "represented_message_count": 0,
        "coverage_mode": "pending",
        "chunk_count": 0,
        "bounded_input": True,
        "model_step_id": events[6].payload["model_step_id"],
    }
    parent_started = asyncio.run(
        _private_events_for_public_events(store, "sess_model_compaction", [events[6]])
    )[0]
    assert events[3].payload == {
        "parent_model_step_id": parent_started.payload["model_step_id"],
        "finish_reason": "stop",
        "model": "summary-model",
        "provider_name": "fake",
        "requested_model": "summary-model",
        "purpose": "context_compaction",
        "compactor": "ModelCompactor",
        "model_step_id": events[6].payload["model_step_id"],
        "model_attempt_id": events[3].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
    }
    assert events[4].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "summary_chars": len("model summary"),
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 2,
        "represented_message_count": 2,
        "coverage_mode": "full",
        "chunk_count": 1,
        "chunk_mode": "single_request",
        "bounded_input": False,
        "compaction_failed": False,
        "model_step_id": events[6].payload["model_step_id"],
    }
    assert "model summary" not in str(events[4].payload)

    provider_context = runtime_provider.requests[0].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nmodel summary"
    )
    assert provider_context[1].content[0].text == "current"

    checkpoint = asyncio.run(store.load_checkpoint("sess_model_compaction"))
    private_compaction_completion = asyncio.run(
        _private_events_for_public_events(store, "sess_model_compaction", [events[3]])
    )[0]
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "model summary",
            "compacted_transcript_cursor": 2,
            "metadata": {
                "compactor": "ModelCompactor",
                "provider": "fake",
                "model": "summary-model",
                "input_truncated": False,
                "max_input_chars": 120000,
                "completed": {
                    "finish_reason": "stop",
                    "model_step_id": private_compaction_completion.payload["model_step_id"],
                    "model_attempt_id": private_compaction_completion.payload["model_attempt_id"],
                },
            },
        },
    }


def test_cayu_app_counts_compaction_model_spend_in_session_usage():
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ]
    )
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_usage",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    compaction_completed = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(compaction_completed) == 1
    assert compaction_completed[0].payload["usage_metrics"]["input_tokens"] == 40
    assert compaction_completed[0].payload["usage_metrics"]["model"] == "summary-model"
    assert compaction_completed[0].payload["usage_metrics"]["requested_model"] == "summary-model"

    summary = asyncio.run(app.get_session_usage("sess_compaction_usage"))
    assert summary.model_steps == 2
    assert summary.provider_names == ["fake"]
    assert summary.models == ["summary-model", "fake-model"]
    assert summary.usage.input_tokens == 50
    assert summary.usage.output_tokens == 10
    assert summary.usage.total_tokens == 60


def test_cayu_app_emits_compaction_failed_event_before_session_failure():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=FailingCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_failed",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[1].payload["execution_profile_fingerprint"],
        "compactor": "FailingCompactor",
        "compacted_transcript_cursor": 0,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 0,
        "recent_message_count": 1,
        "error_type": "RuntimeError",
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 0,
        "represented_message_count": 0,
        "coverage_mode": "failed",
        "chunk_count": 0,
        "chunk_mode": "failed",
        "compaction_failed": True,
        "model_step_id": events[1].payload["model_step_id"],
    }
    assert events[4].payload == {
        "failure_evidence": _expected_failure_evidence(
            "sess_compaction_failed", "RuntimeError", "ContextBuildError"
        ),
        "error": "compaction unavailable",
        "error_type": "RuntimeError",
    }
    assert provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_compaction_failed"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_cayu_app_emits_compaction_events_before_checkpoint_failure():
    class BrokenCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            del session_id, kwargs
            raise RuntimeError("checkpoint unavailable")

    store = BrokenCheckpointStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_checkpoint_failure_after_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
    assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
    expected_disposition = {
        "phase": "checkpoint_installation",
        "reason": "checkpoint_failed",
        "retryable": False,
        "provider_dispatch_disposition": "dispatched",
        "recovery_action": "fail_closed",
    }
    failure = events[4]
    assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
    assert type(failure.payload["elapsed_ms"]) is int
    assert failure.payload["compaction_failed"] is True
    assert events[6].payload["error"] == "checkpoint unavailable"
    assert events[6].payload["error_type"] == "RuntimeError"
    assert {
        key: events[6].payload["compaction_failure"][key] for key in expected_disposition
    } == expected_disposition
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_cayu_app_checkpoint_compaction_ignores_cursor_without_valid_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_bad_checkpoint_pair",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 2,
                    "summary": 123,
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }


def test_cayu_app_checkpoint_compaction_ignores_summary_without_valid_cursor():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_bad_checkpoint_cursor",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 2,
                    "summary": "stale summary",
                    "compacted_transcript_cursor": "bad",
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }


def test_cayu_app_resume_uses_checkpointed_compaction_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_compaction",
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in resume_events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor.requests) == 2
    assert [message.content[0].text for message in compactor.requests[1].messages] == [
        "current",
        "first answer",
    ]
    assert compactor.requests[1].existing_summary == "old|old answer"

    provider_context = provider.requests[1].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nold|old answer|current|first answer"
    )
    assert provider_context[1].content[0].text == "follow up"

    transcript = asyncio.run(store.load_transcript("sess_resume_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "old",
        "old answer",
        "current",
        "first answer",
        "follow up",
        "second answer",
    ]

"""Digest compaction contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _test_session, collect_events

import cayu.context.base as runtime_context_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import file_attachment
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ContextRequest,
    TranscriptDigestCompactor,
)
from cayu.events import EventType
from cayu.messages import FilePart, Message, MessageRole, ProviderStatePart
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest


def test_transcript_digest_compactor_preserves_previous_summary_when_clipping():
    # A large batch of newly compacted messages must not clip the accumulated
    # previous summary out of the front of the combined text.
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = "decision: ship plan A"

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", f"filler {index} " + "x" * 40) for index in range(30)
                ],
                existing_summary=previous_summary,
            )
        )
    )

    assert len(result.summary) <= 400
    assert f"Previous summary:\n{previous_summary}" in result.summary
    assert result.covered_message_count is not None
    assert 0 < result.covered_message_count < 30
    assert result.summary.endswith(f"filler {result.covered_message_count - 1} " + "x" * 40)


def test_transcript_digest_compactor_budgets_both_oversized_sections():
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = ("earlier decision " * 40).strip()

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "latest work " * 50)],
                existing_summary=previous_summary,
            )
        )
    )

    assert result.summary == previous_summary
    assert result.covered_message_count == 0
    assert result.progress_exhausted is True
    assert result.progress_key == compactor._progress_key()
    assert result.metadata["progress_reason"] == "existing_summary_exceeds_limit"


def test_transcript_digest_compactor_keeps_small_previous_summary_unclipped():
    compactor = TranscriptDigestCompactor()
    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                ],
                existing_summary="previous context",
            )
        )
    )
    assert result.summary == (
        "Previous summary:\nprevious context\n\n"
        "Newly compacted transcript:\nuser: old request\nassistant: old answer"
    )
    assert compactor.max_summary_chars == 8000


@pytest.mark.parametrize(
    "messages",
    [
        [Message.text("user", "café 😀"), Message.text("assistant", "résumé")],
        [
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="inspect",
                arguments={"path": "report.txt"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="inspect",
                content="result",
            ),
        ],
    ],
    ids=["unicode", "tool_round"],
)
def test_digest_compactor_preserves_message_boundaries(messages):
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=800).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    assert result.covered_message_count == len(messages)
    assert result.summary


def test_digest_compactor_covers_provider_state_and_attachment_messages():
    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content=(ProviderStatePart(provider="fake", state={"opaque": True}),),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                FilePart(
                    attachment=file_attachment(
                        artifact_id="artifact-coverage",
                        kind="image",
                        filename="coverage.png",
                        content_type="image/png",
                        size_bytes=5,
                    )
                ),
            ),
        ),
    ]
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=800).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    assert result.covered_message_count == 2
    assert "provider_state" in result.summary
    assert "coverage.png" in result.summary
    assert "artifact-coverage" in result.summary


def test_digest_compactor_represents_tool_result_structured_data_and_artifacts():
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=1200).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.tool_call(
                        tool_call_id="call_rich_result",
                        tool_name="inspect",
                        arguments={"path": "report.txt"},
                    ),
                    Message.tool_result(
                        tool_call_id="call_rich_result",
                        tool_name="inspect",
                        content="visible result",
                        structured={"decision": "STRUCTURED_MARKER"},
                        artifacts=[{"artifact_id": "TOOL_ARTIFACT_MARKER"}],
                    ),
                ],
            )
        )
    )
    assert result.covered_message_count == 2
    assert "STRUCTURED_MARKER" in result.summary
    assert "TOOL_ARTIFACT_MARKER" in result.summary


def test_digest_compactor_keeps_oversized_uncovered_source_in_every_projection():
    policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "DIGEST_OLDEST_SURVIVES " + "x" * 1000),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
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
    assert first.checkpoint is not None
    assert first.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    assert "summary" not in first.checkpoint["context_compaction"]
    assert first.checkpoint["context_compaction"]["no_progress"] is True
    assert first.checkpoint["context_compaction"]["progress"] == {
        "exhausted": True,
        "key": policy.compactor._progress_key(),
    }
    first_projection = json.dumps([message.model_dump(mode="json") for message in first.messages])
    assert first_projection.count("DIGEST_OLDEST_SURVIVES") == 1

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint=first.checkpoint,
        )
    )
    assert second.checkpoint is not None
    assert second.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    second_projection = json.dumps([message.model_dump(mode="json") for message in second.messages])
    assert second_projection.count("DIGEST_OLDEST_SURVIVES") == 1

    automatic = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=second.checkpoint,
        )
    )
    assert automatic.checkpoint is None
    assert automatic.compaction_telemetry == []

    changed_policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=300),
        max_user_turns=1,
        compact_after_messages=1,
    )
    changed = asyncio.run(
        changed_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=3,
            ),
            checkpoint=second.checkpoint,
        )
    )
    assert changed.checkpoint is not None
    assert (
        changed.checkpoint["context_compaction"]["progress"]["key"]
        != second.checkpoint["context_compaction"]["progress"]["key"]
    )
    assert any(
        item.event_type == EventType.CONTEXT_COMPACTION_STARTED
        for item in changed.compaction_telemetry
    )


def test_digest_compactor_preserves_oversized_seeded_summary_and_stops_retriggering() -> None:
    policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    previous_summary = "PRIOR_SUMMARY_FACT " * 20
    checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": previous_summary,
            "compacted_transcript_cursor": 2,
            "metadata": {"source": "larger-compactor"},
        }
    }
    messages = [
        Message.text("user", "already covered"),
        Message.text("assistant", "already covered answer"),
        Message.text("user", "NEW_UNCOVERED_FACT"),
        Message.text("assistant", "new answer"),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint=checkpoint,
        )
    )

    assert first.checkpoint is not None
    compacted = first.checkpoint["context_compaction"]
    assert compacted["summary"] == previous_summary
    assert compacted["compacted_transcript_cursor"] == 2
    assert compacted["progress"] == {
        "exhausted": True,
        "key": policy.compactor._progress_key(),
    }
    projection = json.dumps([message.model_dump(mode="json") for message in first.messages])
    assert projection.count("PRIOR_SUMMARY_FACT") == 20
    assert projection.count("NEW_UNCOVERED_FACT") == 1

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
    )
    assert second.checkpoint is None
    assert second.compaction_telemetry == []
    second_projection = json.dumps([message.model_dump(mode="json") for message in second.messages])
    assert second_projection.count("PRIOR_SUMMARY_FACT") == 20
    assert second_projection.count("NEW_UNCOVERED_FACT") == 1


def test_digest_exhaustion_does_not_suppress_replacement_subclass() -> None:
    class ProgressCapableDigestCompactor(TranscriptDigestCompactor):
        def __init__(self) -> None:
            super().__init__(max_summary_chars=200)
            self.requests: list[CompactionRequest] = []

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.requests.append(request)
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(
                    hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                    if request.existing_summary is not None
                    else None
                ),
                source_chunk_mode="custom",
                bounded_input=True,
            )

    messages = [
        Message.text("user", "REPLACEMENT_SOURCE " + "x" * 1000),
        Message.text("user", "current"),
    ]
    exhausted_policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    exhausted = asyncio.run(
        exhausted_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )
    assert exhausted.checkpoint is not None
    exhausted_context = exhausted.checkpoint["context_compaction"]
    assert exhausted_context["progress"]["exhausted"] is True

    replacement = ProgressCapableDigestCompactor()
    replacement_policy = CheckpointCompactionContextPolicy(
        compactor=replacement,
        max_user_turns=1,
        compact_after_messages=1,
    )
    progressed = asyncio.run(
        replacement_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=exhausted.checkpoint,
        )
    )

    assert len(replacement.requests) == 1
    assert progressed.checkpoint is not None
    compacted = progressed.checkpoint["context_compaction"]
    assert compacted["summary"] == "replacement summary"
    assert compacted["compacted_transcript_cursor"] == 1
    assert "progress" not in compacted


def test_digest_no_progress_modes_survive_public_event_sanitization() -> None:
    async def run() -> None:
        session_id = "sess_digest_no_progress_public_telemetry"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=200),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 1000),
                    Message.text("user", "current"),
                ],
            ),
        )

        completed = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert completed.payload["coverage_mode"] == "no_progress"
        assert completed.payload["chunk_mode"] == "digest_capacity_exhausted"
        assert completed.payload["represented_message_count"] == 0
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0

    asyncio.run(run())


def test_digest_no_progress_marker_is_discarded_with_invalid_checkpoint() -> None:
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "SOURCE_MUST_BE_REBUILT"),
        Message.text("user", "current"),
    ]
    invalid_checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": "stale summary",
            "compacted_transcript_cursor": 99,
            "metadata": {},
            "progress": {
                "exhausted": True,
                "key": compactor._progress_key(),
            },
        }
    }

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
            ),
            checkpoint=invalid_checkpoint,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
    assert "SOURCE_MUST_BE_REBUILT" in result.checkpoint["context_compaction"]["summary"]
    assert any(
        item.event_type == EventType.CONTEXT_COMPACTION_STARTED
        for item in result.compaction_telemetry
    )


def test_runtime_progress_state_does_not_overwrite_custom_compactor_metadata() -> None:
    class MetadataCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(
                summary="custom summary",
                covered_message_count=len(request.messages),
                metadata={
                    "progress_exhausted": "custom value",
                    "progress_key": {"owner": "custom compactor"},
                },
            )

    result = asyncio.run(
        CheckpointCompactionContextPolicy(
            compactor=MetadataCompactor(),
            max_user_turns=1,
            compact_after_messages=1,
        ).build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old"),
                    Message.text("user", "current"),
                ],
                step=1,
                force_compaction=True,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    compacted = result.checkpoint["context_compaction"]
    assert compacted["metadata"] == {
        "progress_exhausted": "custom value",
        "progress_key": {"owner": "custom compactor"},
    }
    assert "progress" not in compacted


def test_digest_compactor_does_not_acknowledge_half_of_tool_round():
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=240).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.tool_call(
                        tool_call_id="call_digest_atomic",
                        tool_name="inspect",
                        arguments={"query": "x" * 160},
                    ),
                    Message.tool_result(
                        tool_call_id="call_digest_atomic",
                        tool_name="inspect",
                        content="result",
                    ),
                ],
            )
        )
    )
    assert result.covered_message_count == 0


def test_digest_compactor_builds_each_source_message_once(monkeypatch):
    messages = [
        Message.text(
            "user" if index % 2 == 0 else "assistant",
            f"message-{index:04d}",
        )
        for index in range(4096)
    ]
    original = runtime_context_module._message_digest
    digest_calls = 0

    def counted_message_digest(message):
        nonlocal digest_calls
        digest_calls += 1
        return original(message)

    monkeypatch.setattr(runtime_context_module, "_message_digest", counted_message_digest)
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=1_000_000).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )

    assert result.covered_message_count == len(messages)
    assert digest_calls == len(messages)

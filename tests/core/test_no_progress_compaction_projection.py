"""A compaction that represents no source must not change model-facing context."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tests._session_provenance import fixture_session_invocation

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildResult,
    ContextRequest,
    TranscriptDigestCompactor,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, ResumeRequest, RunRequest
from cayu.sessions.records import Session

OVERSIZED_SOURCE = "OVERSIZED_SOURCE " + "x" * 1_000


class CountingDigestCompactor(TranscriptDigestCompactor):
    def __init__(self, *, max_summary_chars: int) -> None:
        super().__init__(max_summary_chars=max_summary_chars)
        self.requests: list[CompactionRequest] = []

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        return await super().compact(request)


def _session() -> Session:
    return Session(
        id="sess_no_progress",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        invocation=fixture_session_invocation("sess_no_progress"),
    )


def _policy(
    compactor: TranscriptDigestCompactor,
    *,
    compact_after_messages: int = 1,
) -> CheckpointCompactionContextPolicy:
    return CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=compact_after_messages,
    )


def _build(
    policy: CheckpointCompactionContextPolicy,
    messages: list[Message],
    *,
    checkpoint: dict[str, Any] | None,
    step: int = 1,
    force_compaction: bool = False,
) -> ContextBuildResult:
    return asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=step,
                force_compaction=force_compaction,
            ),
            checkpoint=checkpoint,
        )
    )


def _serialized(messages: list[Message]) -> str:
    return json.dumps([message.model_dump(mode="json") for message in messages], sort_keys=True)


def _transcript(*later_turns: str) -> list[Message]:
    messages = [
        Message.text("system", "SYSTEM"),
        Message.text("user", OVERSIZED_SOURCE),
        Message.text("assistant", "acknowledged"),
    ]
    for turn in later_turns:
        messages.extend([Message.text("user", turn), Message.text("assistant", f"{turn} done")])
    messages.append(Message.text("user", "current"))
    return messages


def _uncompacted(messages: list[Message]) -> str:
    never = _policy(TranscriptDigestCompactor(max_summary_chars=200), compact_after_messages=1_000)
    result = _build(never, messages, checkpoint=None)
    assert result.checkpoint is None
    return _serialized(result.messages)


def test_no_progress_compaction_records_typed_state_and_leaves_context_unchanged() -> None:
    compactor = CountingDigestCompactor(max_summary_chars=200)
    policy = _policy(compactor)
    messages = _transcript()

    first = _build(policy, messages, checkpoint=None)

    assert len(compactor.requests) == 1
    assert first.checkpoint is not None
    state = first.checkpoint["context_compaction"]
    assert "summary" not in state
    assert state["no_progress"] is True
    assert state["compacted_transcript_cursor"] == 1
    assert state["progress"] == {"exhausted": True, "key": compactor._progress_key()}
    assert state["metadata"]["progress_reason"] == "no_atomic_prefix_fits"
    completed = [
        item.payload
        for item in first.compaction_telemetry
        if item.event_type is EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    assert len(completed) == 1
    assert completed[0]["coverage_mode"] == "no_progress"
    assert completed[0]["represented_message_count"] == 0
    assert completed[0]["summary_chars"] == 0
    assert first.checkpoint_event_payload is not None
    assert first.checkpoint_event_payload["newly_compacted_message_count"] == 0

    # The turn that attempted compaction and every later turn send exactly
    # what an uncompacted policy would send: no summary message at all.
    assert _serialized(first.messages) == _uncompacted(messages)
    later = _transcript("next")
    second = _build(policy, later, checkpoint=first.checkpoint, step=2)
    assert _serialized(second.messages) == _uncompacted(later)
    assert "Previous session context summary" not in _serialized(second.messages)


def test_no_progress_compaction_does_not_retrigger_on_later_turns() -> None:
    compactor = CountingDigestCompactor(max_summary_chars=200)
    policy = _policy(compactor)
    first = _build(policy, _transcript(), checkpoint=None)
    checkpoint = first.checkpoint
    assert len(compactor.requests) == 1

    for step, turns in enumerate((("a",), ("a", "b"), ("a", "b", "c")), start=2):
        later = _build(policy, _transcript(*turns), checkpoint=checkpoint, step=step)
        assert later.checkpoint is None
        assert later.compaction_telemetry == []

    assert len(compactor.requests) == 1

    # An explicit application request still runs, and records the same state.
    forced = _build(
        policy,
        _transcript("a"),
        checkpoint=checkpoint,
        step=5,
        force_compaction=True,
    )
    assert len(compactor.requests) == 2
    assert forced.checkpoint is not None
    assert forced.checkpoint["context_compaction"]["no_progress"] is True
    assert _serialized(forced.messages) == _uncompacted(_transcript("a"))


def test_no_progress_state_yields_to_a_compactor_that_can_progress() -> None:
    first = _build(
        _policy(TranscriptDigestCompactor(max_summary_chars=200)), _transcript(), checkpoint=None
    )
    larger = CountingDigestCompactor(max_summary_chars=4_000)

    progressed = _build(_policy(larger), _transcript(), checkpoint=first.checkpoint, step=2)

    assert len(larger.requests) == 1
    # The no-progress record is not offered to the compactor as a summary.
    assert larger.requests[0].existing_summary is None
    assert progressed.checkpoint is not None
    state = progressed.checkpoint["context_compaction"]
    assert "no_progress" not in state
    assert "progress" not in state
    assert "OVERSIZED_SOURCE" in state["summary"]
    assert state["compacted_transcript_cursor"] == 3


def test_seeded_summary_at_the_first_cursor_is_still_projected() -> None:
    compactor = CountingDigestCompactor(max_summary_chars=200)
    checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": "SEEDED_PRIOR_SESSION_FACT",
            "compacted_transcript_cursor": 1,
            "metadata": {"source": "imported"},
        }
    }

    result = _build(
        _policy(compactor, compact_after_messages=1_000), _transcript(), checkpoint=checkpoint
    )

    assert compactor.requests == []
    assert "SEEDED_PRIOR_SESSION_FACT" in _serialized(result.messages)


def test_runtime_request_after_no_progress_compaction_is_byte_identical() -> None:
    def provider() -> ScriptedModelProvider:
        def respond(request: ModelRequest) -> list[ModelStreamEvent]:
            del request
            return [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    }
                ),
            ]

        return ScriptedModelProvider(response_factory=respond)

    async def requests_for(policy: CheckpointCompactionContextPolicy) -> list[str]:
        scripted = provider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(scripted, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model", system_prompt="SYSTEM"),
            context_policy=policy,
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_byte_identical",
                    messages=[Message.text("user", OVERSIZED_SOURCE)],
                )
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        for turn in ("second", "third"):
            async for _event in app.resume(
                ResumeRequest(
                    session_id="sess_byte_identical",
                    messages=[Message.text("user", turn)],
                )
            ):
                pass
        return [
            json.dumps(
                {
                    "messages": [message.model_dump(mode="json") for message in request.messages],
                    "tools": request.tools,
                },
                sort_keys=True,
            )
            for request in scripted.requests
        ]

    compactor = CountingDigestCompactor(max_summary_chars=200)
    with_compaction = asyncio.run(requests_for(_policy(compactor)))
    without_compaction = asyncio.run(
        requests_for(
            _policy(
                TranscriptDigestCompactor(max_summary_chars=200),
                compact_after_messages=1_000,
            )
        )
    )

    # The no-progress compaction ran once, on the second turn, and no later
    # request differs by a single byte from the uncompacted session.
    assert len(compactor.requests) == 1
    assert len(with_compaction) == len(without_compaction) == 3
    assert with_compaction == without_compaction

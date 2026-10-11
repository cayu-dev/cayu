"""Size-based compaction policies for long histories and tool rounds."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from tests.core._explicit_compaction_fixtures import RecordingCompactor

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import FileAttachmentKind, file_attachment
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextRequest,
)
from cayu.events import EventType
from cayu.messages import Message, ToolCallPart, ToolResultPart
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


def test_size_based_compaction_handles_one_long_user_turn_and_stays_quiet_afterward() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-single-turn",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        policy = CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=1_600,
            max_recent_context_tokens=1_000,
            reserved_output_tokens=100,
        )
        messages = [
            Message.text("system", "Solve the task precisely."),
            Message.text("user", "Original task " + "q" * 2_000),
            Message.tool_call(
                tool_call_id="old-call",
                tool_name="search",
                arguments={"query": "old evidence"},
            ),
            Message.tool_result(
                tool_call_id="old-call",
                tool_name="search",
                content="old evidence " + "x" * 4_000,
            ),
            Message.tool_call(
                tool_call_id="recent-call",
                tool_name="search",
                arguments={"query": "recent evidence"},
            ),
            Message.tool_result(
                tool_call_id="recent-call",
                tool_name="search",
                content="recent decisive evidence",
            ),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=3,
        )

        first = await policy.build_with_checkpoint(request, checkpoint=None)

        assert len(compactor.requests) == 1
        compacted = compactor.requests[0].messages
        assert compacted[0].role.value == "assistant"
        assert compacted[-1].role.value == "tool"
        assert "old evidence" in str(compacted[-1].model_dump(mode="json"))
        assert "Original task" not in json.dumps(
            [message.model_dump(mode="json") for message in compacted]
        )
        projected = json.dumps([message.model_dump(mode="json") for message in first.messages])
        assert "durable compact summary" in projected
        assert "Original task " + "q" * 2_000 in projected
        assert "recent decisive evidence" in projected
        assert "old evidence " + "x" * 100 not in projected
        assert first.checkpoint is not None

        second = await policy.build_with_checkpoint(
            request,
            checkpoint=first.checkpoint,
        )

        assert len(compactor.requests) == 1
        assert second.checkpoint is None
        assert "recent decisive evidence" in json.dumps(
            [message.model_dump(mode="json") for message in second.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_requires_a_smaller_recent_context_target() -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        CheckpointCompactionContextPolicy(
            compact_after_estimated_context_tokens=1_000,
        )
    with pytest.raises(ValueError, match="must be less"):
        CheckpointCompactionContextPolicy(
            compact_after_estimated_context_tokens=1_000,
            max_recent_context_tokens=800,
            reserved_output_tokens=200,
        )


def test_size_based_compaction_uses_configured_attachment_retention() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-attachment-retention",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "Inspect retained attachments."),
            Message.text("user", "Inspect every image."),
        ]
        for index in range(3):
            attachment = file_attachment(
                artifact_id=f"art-{index}",
                kind=FileAttachmentKind.IMAGE,
                filename=f"image-{index}.png",
                content_type="image/png",
                size_bytes=1_000_000,
            )
            messages.extend(
                [
                    Message.tool_call(
                        tool_call_id=f"call-{index}",
                        tool_name="read_image",
                        arguments={"artifact_id": f"art-{index}"},
                    ),
                    Message.tool_result(
                        tool_call_id=f"call-{index}",
                        tool_name="read_image",
                        content="attached",
                        artifacts=[attachment],
                    ),
                ]
            )
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=4,
        )

        retained_compactor = RecordingCompactor()
        retained = await CheckpointCompactionContextPolicy(
            compactor=retained_compactor,
            compact_after_estimated_context_tokens=125_070,
            max_recent_context_tokens=100_000,
            max_attachment_results=3,
        ).build_with_checkpoint(request, checkpoint=None)

        assert len(retained_compactor.requests) == 1
        assert retained.checkpoint is not None

        stripped_compactor = RecordingCompactor()
        stripped = await CheckpointCompactionContextPolicy(
            compactor=stripped_compactor,
            compact_after_estimated_context_tokens=1_000,
            max_recent_context_tokens=500,
            max_attachment_results=0,
        ).build_with_checkpoint(request, checkpoint=None)

        assert stripped_compactor.requests == []
        assert stripped.checkpoint is None
        assert "cayu.file_attachment.v1" not in json.dumps(
            [message.model_dump(mode="json") for message in stripped.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_reduces_context_when_fixed_overhead_exceeds_target() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-fixed-overhead",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "s" * 2_000),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "OLD_REMOVABLE_CONTEXT " + "x" * 4_000),
            Message.text("user", "recent request"),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=2,
        )

        compactor = RecordingCompactor()
        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=1_800,
            max_recent_context_tokens=250,
            reserved_output_tokens=100,
        ).build_with_checkpoint(request, checkpoint=None)

        projected = json.dumps([message.model_dump(mode="json") for message in result.messages])
        assert len(compactor.requests) == 1
        assert result.checkpoint is not None
        assert "ORIGINAL_TASK" in projected
        assert "OLD_REMOVABLE_CONTEXT" not in projected

        unsafe_compactor = RecordingCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds") as error:
            await CheckpointCompactionContextPolicy(
                compactor=unsafe_compactor,
                compact_after_estimated_context_tokens=600,
                max_recent_context_tokens=250,
                reserved_output_tokens=100,
            ).build_with_checkpoint(request, checkpoint=None)
        assert len(unsafe_compactor.requests) == 1
        assert error.value.checkpoint is not None

    asyncio.run(run())


@pytest.mark.parametrize("enforce_target", [True, False])
def test_size_based_compaction_fails_closed_and_checkpoints_oversized_summary(
    enforce_target,
) -> None:
    class OversizedSummaryCompactor(RecordingCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.requests.append(request)
            return CompactionResult(
                summary="z" * 5_000,
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(
                    hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                    if request.existing_summary is not None
                    else None
                ),
            )

    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-oversized-summary",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "system"),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "old context " + "x" * 5_000),
            Message.text("assistant", "latest observation"),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=2,
        )
        first_compactor = OversizedSummaryCompactor()

        with pytest.raises(ContextBuildError, match="configured size bounds") as first_error:
            await CheckpointCompactionContextPolicy(
                compactor=first_compactor,
                enforce_recent_context_target=enforce_target,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(request, checkpoint=None)

        checkpoint = first_error.value.checkpoint
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "z" * 5_000
        assert len(first_compactor.requests) == 1

        restarted_compactor = OversizedSummaryCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds") as restarted_error:
            await CheckpointCompactionContextPolicy(
                compactor=restarted_compactor,
                enforce_recent_context_target=enforce_target,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                request.model_copy(update={"force_bounded_compaction": True}),
                checkpoint=checkpoint,
            )
        assert len(restarted_compactor.requests) == 1
        assert restarted_compactor.requests[0].existing_summary == "z" * 5_000
        exhausted_checkpoint = restarted_error.value.checkpoint
        assert exhausted_checkpoint is not None
        assert exhausted_checkpoint["context_compaction"]["compacted_transcript_cursor"] == len(
            messages
        )

        exhausted_compactor = OversizedSummaryCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds"):
            await CheckpointCompactionContextPolicy(
                compactor=exhausted_compactor,
                enforce_recent_context_target=enforce_target,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                request.model_copy(update={"force_bounded_compaction": True}),
                checkpoint=exhausted_checkpoint,
            )
        assert exhausted_compactor.requests == []

    asyncio.run(run())


def test_size_based_compaction_honors_bounded_overflow_force_below_trigger() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-bounded-overflow",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=8_000,
            reserved_output_tokens=1_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("system", "system"),
                    Message.text("user", "ORIGINAL_TASK"),
                    Message.text("assistant", "compact despite the local estimate"),
                    Message.text("assistant", "latest observation"),
                ],
                step=2,
                force_bounded_compaction=True,
            ),
            checkpoint=None,
        )

        assert len(compactor.requests) == 1
        assert compactor.requests[0].force_bounded_compaction is True
        assert result.checkpoint is not None
        assert "ORIGINAL_TASK" in json.dumps(
            [message.model_dump(mode="json") for message in result.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_keeps_the_only_recent_tool_round_visible() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-only-recent-tool-round",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        messages = [
            Message.text("user", "ORIGINAL_TASK"),
            Message.tool_call(
                tool_call_id="recent-call",
                tool_name="search",
                arguments={"query": "latest"},
            ),
            Message.tool_result(
                tool_call_id="recent-call",
                tool_name="search",
                content="latest result",
            ),
        ]

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=5_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
                force_compaction=True,
            ),
            checkpoint=None,
        )

        assert compactor.requests == []
        assert result.checkpoint is None
        assert [message.model_dump(mode="json") for message in result.messages] == [
            message.model_dump(mode="json") for message in messages
        ]

    asyncio.run(run())


def test_size_based_compaction_compacts_an_oversized_latest_tool_round_atomically() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-oversized-latest-tool-round",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        calls = [
            ToolCallPart(
                tool_call_id=f"inspect-{index}",
                tool_name="product_inspect",
                arguments={"candidate_id": f"candidate-{index}"},
            )
            for index in range(12)
        ]
        results = [
            ToolResultPart(
                tool_call_id=f"inspect-{index}",
                tool_name="product_inspect",
                content=f"candidate-{index}-evidence:" + "x" * 12_000,
            )
            for index in range(12)
        ]
        messages = [
            Message.text("user", "ORIGINAL_TASK"),
            Message.tool_call(calls=calls),
            Message.tool_result(results=results),
        ]

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=8_000,
            reserved_output_tokens=1_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=None,
        )

        assert len(compactor.requests) == 1
        compacted = compactor.requests[0].messages
        assert [message.model_dump(mode="json") for message in compacted] == [
            message.model_dump(mode="json") for message in messages[1:]
        ]
        assert result.checkpoint is not None
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == len(
            messages
        )
        projected = json.dumps([message.model_dump(mode="json") for message in result.messages])
        assert "ORIGINAL_TASK" in projected
        assert "durable compact summary" in projected
        assert "candidate-0-evidence" not in projected

    asyncio.run(run())


def test_size_based_compaction_checkpoint_survives_sqlite_restart(tmp_path) -> None:
    async def run() -> None:
        database = tmp_path / "size-based-compaction-restart.sqlite"
        store = SQLiteSessionStore(database)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-sqlite-restart",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "Solve precisely."),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "old answer " + "x" * 4_000),
            Message.text("assistant", "recent observation"),
        ]
        await store.append_transcript_messages(session.id, messages)
        first_compactor = RecordingCompactor()
        first = await CheckpointCompactionContextPolicy(
            compactor=first_compactor,
            compact_after_estimated_context_tokens=600,
            max_recent_context_tokens=250,
            reserved_output_tokens=100,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await store.checkpoint(session.id, first.checkpoint)
        await store.close()

        reopened = SQLiteSessionStore(database)
        restarted_compactor = RecordingCompactor()
        try:
            await reopened.append_transcript_messages(
                session.id,
                [Message.text("user", "follow-up after restart")],
            )
            reloaded_session = await reopened.load(session.id)
            assert reloaded_session is not None
            reloaded_messages = await reopened.load_transcript(session.id)
            reloaded_checkpoint = await reopened.load_checkpoint(session.id)
            assert reloaded_checkpoint is not None

            restarted = await CheckpointCompactionContextPolicy(
                compactor=restarted_compactor,
                compact_after_estimated_context_tokens=600,
                max_recent_context_tokens=250,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                ContextRequest(
                    session=reloaded_session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=reloaded_messages,
                    step=3,
                ),
                checkpoint=reloaded_checkpoint,
            )

            projected = json.dumps(
                [message.model_dump(mode="json") for message in restarted.messages]
            )
            assert restarted_compactor.requests == []
            assert restarted.checkpoint is None
            assert "durable compact summary" in projected
            assert "ORIGINAL_TASK" in projected
            assert "recent observation" in projected
            assert "follow-up after restart" in projected
            assert "old answer " + "x" * 100 not in projected
        finally:
            await reopened.close()

    asyncio.run(run())


class _LongLoopEvidenceTool(Tool):
    spec = ToolSpec(
        name="collect_evidence",
        description="Collect one large deterministic evidence record.",
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        index = int(args["index"])
        return ToolResult(
            content=f"evidence-{index}:" + "x" * 4_000,
            structured={"index": index, "payload": "x" * 4_000},
        )


class _LongToolLoopProvider(ModelProvider):
    name = "long-tool-loop"
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests.long-tool-loop-provider",
        behavior_version="1",
        implementation_version="1",
    )

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < self.rounds:
            yield ModelStreamEvent.tool_call(
                id=f"evidence-call-{index}",
                name="collect_evidence",
                arguments={"index": index},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("finished after compacted tool loop")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _WideToolRoundProvider(ModelProvider):
    name = "wide-tool-round"
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests.wide-tool-round-provider",
        behavior_version="1",
        implementation_version="1",
    )

    def __init__(self, tool_calls: int) -> None:
        self.tool_calls = tool_calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            for index in range(self.tool_calls):
                yield ModelStreamEvent.tool_call(
                    id=f"evidence-call-{index}",
                    name="collect_evidence",
                    arguments={"index": index},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("continued after compacting the wide tool round")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_size_based_compaction_runs_through_a_real_long_tool_loop() -> None:
    async def run() -> None:
        original_task = "ORIGINAL-LONG-TOOL-TASK:" + " preserve-me" * 80
        provider = _LongToolLoopProvider(rounds=20)
        compactor = RecordingCompactor()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                workflow_tool_names=("collect_evidence",),
            ),
            tools=[_LongLoopEvidenceTool()],
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                compact_after_estimated_context_tokens=4_000,
                max_recent_context_tokens=2_500,
                reserved_output_tokens=500,
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="real-long-tool-loop-compaction",
                    messages=[Message.text("user", original_task)],
                    max_steps=32,
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert len(provider.requests) == 21
        assert len(compactor.requests) >= 1
        for request in provider.requests:
            projected = json.dumps(
                [message.model_dump(mode="json") for message in request.messages]
            )
            assert original_task in projected

    asyncio.run(run())


def test_size_based_compaction_continues_after_a_wide_oversized_tool_round() -> None:
    async def run() -> None:
        original_task = "ORIGINAL-WIDE-TOOL-TASK"
        provider = _WideToolRoundProvider(tool_calls=12)
        compactor = RecordingCompactor()
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=12)),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                workflow_tool_names=("collect_evidence",),
            ),
            tools=[_LongLoopEvidenceTool()],
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                compact_after_estimated_context_tokens=5_000,
                max_recent_context_tokens=3_000,
                reserved_output_tokens=500,
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="wide-tool-round-compaction",
                    messages=[Message.text("user", original_task)],
                    max_steps=4,
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert len(compactor.requests) == 1
        assert len(compactor.requests[0].messages) == 2
        compacted_round = json.dumps(
            [message.model_dump(mode="json") for message in compactor.requests[0].messages]
        )
        assert compacted_round.count('"type": "tool_call"') == 12
        assert compacted_round.count('"type": "tool_result"') == 12
        continued_projection = json.dumps(
            [message.model_dump(mode="json") for message in provider.requests[1].messages]
        )
        assert original_task in continued_projection
        assert "durable compact summary" in continued_projection
        assert "evidence-0:" not in continued_projection

    asyncio.run(run())

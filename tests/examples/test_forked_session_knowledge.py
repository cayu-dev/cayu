from __future__ import annotations

import asyncio
import json

from examples.forked_session_knowledge import (
    CHILD_SESSION_ID,
    KNOWLEDGE_TOOL_ID,
    PARENT_SESSION_ID,
    build_demo,
    run_demo,
)

from cayu import (
    AlwaysRequireApprovalToolPolicy,
    EventType,
    KnowledgeActorType,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.providers import ModelRequest
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.openai import build_openai_payload
from cayu.runtime.tool_grants import TARGETED_TOOL_TRANSCRIPT_REFERENCE

_TARGETED_CONTEXT_SCHEMA = "cayu.targeted-tool-context.v1"


def test_one_memory_child_runs_with_parent_and_preserves_the_provider_prefix() -> None:
    findings = (
        "Retries must preserve the outer call identity.",
        "Forks require fresh branch-local targeted grants.",
    )
    demo = build_demo(findings=findings)
    result = asyncio.run(run_demo(demo, max_calls=3))

    assert result.parent_initial_events[-1].type is EventType.SESSION_COMPLETED
    assert [event.type for event in result.fork_events] == [EventType.SESSION_FORKED]
    assert result.parent_continuation_events[-1].type is EventType.SESSION_COMPLETED
    assert result.memory_events[-1].type is EventType.SESSION_COMPLETED
    assert demo.provider.max_concurrent_requests >= 2
    assert {entry.text for entry in result.pending_entries} == set(findings)
    assert all(entry.status is KnowledgeStatus.PENDING for entry in result.pending_entries)
    assert all(
        entry.created_by_type is KnowledgeActorType.MODEL for entry in result.pending_entries
    )
    assert all(entry.source_id == CHILD_SESSION_ID for entry in result.pending_entries)

    assert all(
        [tool["name"] for tool in request.tools] == ["call_tool"]
        for request in demo.provider.requests
    )
    assert all(
        "remember_knowledge" not in [tool["name"] for tool in request.tools]
        for request in demo.provider.requests
    )

    parent_continuation = _request_with_text(demo.provider.requests, "Continue with the next task.")
    memory_request = _request_with_targeted_context(demo.provider.requests)
    parent_prefix = [_message_json(message) for message in parent_continuation.messages[:3]]
    memory_prefix = [_message_json(message) for message in memory_request.messages[:3]]
    assert parent_prefix == memory_prefix
    assert parent_continuation.messages[0].role == "system"
    assert memory_request.messages[-1].role == "user"
    assert _TARGETED_CONTEXT_SCHEMA in memory_request.messages[-1].content[0].text
    assert all(
        _TARGETED_CONTEXT_SCHEMA not in part.text
        for message in memory_request.messages
        if message.role == "system"
        for part in message.content
        if part.type == "text"
    )

    openai_parent = build_openai_payload(parent_continuation)
    openai_memory = build_openai_payload(memory_request)
    assert openai_parent["store"] is False
    assert openai_parent["instructions"] == openai_memory["instructions"]
    assert openai_parent["tools"] == openai_memory["tools"]
    assert openai_parent["input"][:2] == openai_memory["input"][:2]

    anthropic_parent = build_anthropic_payload(parent_continuation)
    anthropic_memory = build_anthropic_payload(memory_request)
    assert anthropic_parent["system"] == anthropic_memory["system"]
    assert anthropic_parent["tools"] == anthropic_memory["tools"]
    assert anthropic_parent["messages"][:2] == anthropic_memory["messages"][:2]

    assert asyncio.run(demo.session_store.list_targeted_tool_grants(PARENT_SESSION_ID)) == ()
    [grant] = asyncio.run(demo.session_store.list_targeted_tool_grants(CHILD_SESSION_ID))
    assert grant.tool_id == KNOWLEDGE_TOOL_ID
    assert grant.used_calls == 2
    assert grant.remaining_calls == 1

    parent = asyncio.run(demo.session_store.load(PARENT_SESSION_ID))
    child = asyncio.run(demo.session_store.load(CHILD_SESSION_ID))
    assert parent is not None and parent.parent_session_id is None
    assert child is not None and child.parent_session_id == PARENT_SESSION_ID
    assert parent.id != child.id

    transcript = asyncio.run(demo.session_store.load_transcript(CHILD_SESSION_ID))
    serialized_transcript = json.dumps(
        [message.model_dump(mode="json") for message in transcript],
        sort_keys=True,
    )
    assert grant.tool_ref not in serialized_transcript
    assert _TARGETED_CONTEXT_SCHEMA not in serialized_transcript
    assert TARGETED_TOOL_TRANSCRIPT_REFERENCE in serialized_transcript


def test_memory_child_can_complete_without_proposing_knowledge() -> None:
    demo = build_demo(findings=())
    result = asyncio.run(run_demo(demo, max_calls=1))

    assert result.parent_continuation_events[-1].type is EventType.SESSION_COMPLETED
    assert result.memory_events[-1].type is EventType.SESSION_COMPLETED
    assert demo.provider.max_concurrent_requests >= 2
    assert result.pending_entries == ()
    [grant] = result.child_grants
    assert grant.used_calls == 0
    assert grant.remaining_calls == 1
    assert all(
        [tool["name"] for tool in request.tools] == ["call_tool"]
        for request in demo.provider.requests
    )
    assert not any(event.type is EventType.TOOL_CALL_STARTED for event in result.memory_events)


def test_memory_child_rejects_calls_beyond_the_grant_budget() -> None:
    findings = tuple(f"Bounded finding {index}." for index in range(4))
    demo = build_demo(findings=findings)
    result = asyncio.run(run_demo(demo, max_calls=3))

    assert result.memory_events[-1].type is EventType.SESSION_COMPLETED
    assert {entry.text for entry in result.pending_entries} == set(findings[:3])
    [grant] = result.child_grants
    assert grant.used_calls == 3
    assert grant.remaining_calls == 0
    [rejected] = [
        event
        for event in result.memory_events
        if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
    ]
    assert rejected.payload["rejection_reason"] == "exhausted"
    assert sum(event.type is EventType.TOOL_CALL_STARTED for event in result.memory_events) == 3
    assert all(
        [tool["name"] for tool in request.tools] == ["call_tool"]
        for request in demo.provider.requests
    )


def test_memory_child_approval_continuation_rejoins_and_writes_once() -> None:
    async def run() -> None:
        demo = build_demo(
            findings=("Approval recovery must not duplicate knowledge writes.",),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        initial = await run_demo(demo, max_calls=1)
        approval = next(
            event
            for event in initial.memory_events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        assert initial.parent_continuation_events[-1].type is EventType.SESSION_COMPLETED
        assert not initial.pending_entries

        resumed_events = tuple(
            [
                event
                async for event in demo.app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=CHILD_SESSION_ID,
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
        )

        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert (
            sum(
                event.type is EventType.TARGETED_TOOL_REFERENCE_REJOINED for event in resumed_events
            )
            == 1
        )
        reviewer = KnowledgeReviewWorkflow(demo.knowledge_store, namespace="default")
        pending = await reviewer.list_pending(
            source_type="tool",
            source_id=CHILD_SESSION_ID,
            limit=5,
        )
        assert [item.entry.text for item in pending.entries] == [
            "Approval recovery must not duplicate knowledge writes."
        ]
        [grant] = await demo.session_store.list_targeted_tool_grants(CHILD_SESSION_ID)
        assert grant.used_calls == 1
        assert grant.remaining_calls == 0

    asyncio.run(run())


def _request_with_text(requests: list[ModelRequest], text: str) -> ModelRequest:
    return next(
        request
        for request in requests
        if any(
            part.type == "text" and part.text == text
            for message in request.messages
            for part in message.content
        )
    )


def _request_with_targeted_context(requests: list[ModelRequest]) -> ModelRequest:
    return next(
        request
        for request in requests
        if any(
            part.type == "text" and _TARGETED_CONTEXT_SCHEMA in part.text
            for message in request.messages
            for part in message.content
        )
    )


def _message_json(message) -> dict:
    return message.model_dump(mode="json")

"""Frozen core models: copies happen once, at construction trust boundaries.

Core message/tool/retry models are immutable value objects. Construction deep
copies every caller-supplied JSON payload; after that, sharing an instance is
safe, so hot-path "copies" (transcript reads, per-attempt retry-policy copies,
session deep copies) are no-ops instead of per-field rebuilds.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    copy_message,
    copy_message_part,
)
from cayu.core.tools import ToolContext, ToolResult
from cayu.runtime.retry_policy import RetryDecision, RetryPolicy, copy_retry_policy


def test_message_and_parts_reject_attribute_assignment() -> None:
    message = Message.text("user", "hello")
    part = ToolCallPart(tool_call_id="call_1", tool_name="echo", arguments={"a": 1})
    result = ToolResultPart(tool_call_id="call_1", tool_name="echo", content="ok")
    state = ProviderStatePart(provider="openai", state={"id": "rs_1"})
    thinking = ThinkingPart(text="reason")

    with pytest.raises(ValidationError):
        message.role = MessageRole.ASSISTANT  # type: ignore[misc]
    with pytest.raises(ValidationError):
        message.content[0].text = "mutated"  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        part.tool_name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        result.is_error = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        state.provider = "anthropic"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        thinking.text = "changed"  # type: ignore[misc]


def test_copy_message_is_a_no_op_for_validated_messages() -> None:
    message = Message.tool_call(
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"nested": {"value": "original"}},
    )

    assert copy_message(message) is message
    assert deepcopy(message) is message
    assert message.model_copy(deep=True) is message


def test_copy_message_still_revalidates_construct_bypass() -> None:
    bypassed = Message.model_construct(
        role="user",
        content=[TextPart.model_construct(text=" ")],
    )

    with pytest.raises(ValueError, match="`text` cannot be blank"):
        copy_message(bypassed)

    revalidated = copy_message(Message.model_construct(role="user", content=[TextPart(text="ok")]))
    assert revalidated is not bypassed
    assert revalidated.content[0].text == "ok"
    # The rebuilt message is validated, so copying it again is a no-op.
    assert copy_message(revalidated) is revalidated


def test_message_construction_owns_part_payloads() -> None:
    part = ToolCallPart(tool_call_id="call_1", tool_name="echo", arguments={"n": {"v": "original"}})
    message = Message.tool_call(calls=[part])

    part.arguments["n"]["v"] = "mutated"

    call = message.content[0]
    assert isinstance(call, ToolCallPart)
    assert call.arguments == {"n": {"v": "original"}}


def test_copy_message_part_detaches_payloads_generically() -> None:
    part = ToolResultPart(
        tool_call_id="call_1",
        tool_name="echo",
        structured={"n": {"v": "original"}},
        artifacts=[{"n": {"v": "original"}}],
    )
    copied = copy_message_part(part)
    assert isinstance(copied, ToolResultPart)

    part.structured["n"]["v"] = "mutated"  # type: ignore[index]
    part.artifacts[0]["n"]["v"] = "mutated"

    assert copied.structured == {"n": {"v": "original"}}
    assert copied.artifacts == [{"n": {"v": "original"}}]


def test_session_deep_copies_share_frozen_messages() -> None:
    message = Message.text("user", "hello")
    messages = [message, Message.text("user", "again")]

    copied = deepcopy(messages)

    assert copied is not messages
    assert copied[0] is message


def test_retry_policy_is_frozen_and_shared_instead_of_rebuilt() -> None:
    policy = RetryPolicy(max_attempts=3)

    with pytest.raises(ValidationError):
        policy.max_attempts = 5  # type: ignore[misc]

    assert copy_retry_policy(policy) is policy
    assert copy_retry_policy(None) == RetryPolicy()

    with pytest.raises(TypeError, match="RetryPolicy"):
        copy_retry_policy(object())  # type: ignore[arg-type]

    decision = RetryDecision(retry=False, attempt=1, max_attempts=3)
    with pytest.raises(ValidationError):
        decision.retry = True  # type: ignore[misc]


def test_tool_result_and_context_are_frozen() -> None:
    result = ToolResult(content="ok", structured={"n": 1})
    with pytest.raises(ValidationError):
        result.content = "changed"  # type: ignore[misc]

    ctx = ToolContext(session_id="sess_1")
    with pytest.raises(ValidationError):
        ctx.session_id = "sess_2"  # type: ignore[misc]

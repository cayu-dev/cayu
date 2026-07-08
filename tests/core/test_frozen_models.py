"""Frozen core models: copies happen once, at construction trust boundaries.

Core message/tool/retry models are immutable value objects. Construction deep
copies every caller-supplied JSON payload; after that, sharing an instance is
safe while payloads are treated as read-only, so hot-path "copies" (context
builds, per-attempt retry-policy copies) are no-ops instead of per-field
rebuilds. Isolation is explicit: `deepcopy` and `detach_message` produce
copies with detached payloads for storage/trust boundaries.
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
    detach_message,
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


def test_deepcopy_isolates_messages() -> None:
    message = Message.tool_call(
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"nested": {"value": "original"}},
    )

    copied = deepcopy(message)
    assert copied is not message
    assert copied == message
    copied.content[0].arguments["nested"]["value"] = "mutated"  # type: ignore[union-attr]
    assert message.content[0].arguments == {"nested": {"value": "original"}}  # type: ignore[union-attr]

    duplicated = message.model_copy(deep=True)
    assert duplicated is not message
    duplicated.content[0].arguments["nested"]["value"] = "mutated"  # type: ignore[union-attr]
    assert message.content[0].arguments == {"nested": {"value": "original"}}  # type: ignore[union-attr]

    # Deepcopy preserves the validated-content marker, so revalidating the
    # copy stays a no-op.
    assert copy_message(copied) is copied


def test_detach_message_isolates_all_payload_fields() -> None:
    payload = {"nested": {"value": "original"}}
    original = {"nested": {"value": "original"}}
    assistant = Message(
        role=MessageRole.ASSISTANT,
        content=(
            ToolCallPart(tool_call_id="call_1", tool_name="echo", arguments=payload),
            ProviderStatePart(provider="openai", state=payload),
            ThinkingPart(text="reason", provider_state=payload),
        ),
    )
    tool = Message.tool_result(
        tool_call_id="call_1",
        tool_name="echo",
        structured=payload,
        artifacts=[payload],
    )

    detached_assistant = detach_message(assistant)
    assert detached_assistant is not assistant
    assert detached_assistant == assistant
    detached_assistant.content[0].arguments["nested"]["value"] = "mutated"  # type: ignore[union-attr]
    detached_assistant.content[1].state["nested"]["value"] = "mutated"  # type: ignore[union-attr]
    detached_assistant.content[2].provider_state["nested"]["value"] = "mutated"  # type: ignore[index, union-attr]
    assert assistant.content[0].arguments == original  # type: ignore[union-attr]
    assert assistant.content[1].state == original  # type: ignore[union-attr]
    assert assistant.content[2].provider_state == original  # type: ignore[union-attr]

    detached_tool = detach_message(tool)
    assert detached_tool is not tool
    assert detached_tool == tool
    detached_tool.content[0].structured["nested"]["value"] = "mutated"  # type: ignore[index, union-attr]
    detached_tool.content[0].artifacts[0]["nested"]["value"] = "mutated"  # type: ignore[union-attr]
    assert tool.content[0].structured == original  # type: ignore[union-attr]
    assert tool.content[0].artifacts == [original]  # type: ignore[union-attr]

    # The detached messages are fully validated, so hot-path copies of them
    # remain no-ops.
    assert copy_message(detached_assistant) is detached_assistant
    assert copy_message(detached_tool) is detached_tool


def test_detach_message_revalidates_construct_bypass_and_rejects_non_messages() -> None:
    bypassed = Message.model_construct(
        role="user",
        content=[TextPart.model_construct(text=" ")],
    )
    with pytest.raises(ValueError, match="`text` cannot be blank"):
        detach_message(bypassed)

    with pytest.raises(TypeError, match="Message instances"):
        detach_message(object())  # type: ignore[arg-type]


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


def test_deepcopy_of_message_lists_isolates_each_message() -> None:
    message = Message.text("user", "hello")
    messages = [message, Message.text("user", "again")]

    copied = deepcopy(messages)

    assert copied is not messages
    assert copied[0] is not message
    assert copied[0] == message


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

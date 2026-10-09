"""Shared placement rules for system messages in provider payloads.

Only the leading contiguous run of system messages is the request's system
prompt. Providers send that run in their system or instructions field. Every
later system message is conversation content and stays at its position, so a
system message that changes on every turn (controller or memory state, for
example) changes only the tail of the request and the cacheable prefix stays
append-only.

The one permitted move keeps tool protocols valid: a later system message that
appears between an assistant tool-call turn and the tool results answering it
is sent immediately after those tool results.
"""

from __future__ import annotations

from collections.abc import Sequence

from cayu.messages import Message, MessageRole, TextPart, ToolCallPart

LATE_SYSTEM_NOTE_OPEN = "<system>"
LATE_SYSTEM_NOTE_CLOSE = "</system>"


def leading_system_count(messages: Sequence[Message]) -> int:
    """Return the length of the leading contiguous run of system messages."""

    count = 0
    for message in messages:
        if message.role is not MessageRole.SYSTEM:
            break
        count += 1
    return count


def system_message_text(message: Message) -> str:
    """Join one system message's text parts the way the system field joins them."""

    return "\n\n".join(part.text for part in message.content if type(part) is TextPart)


def leading_system_text(messages: Sequence[Message]) -> str:
    """Return the text of the leading system run, joined by blank lines."""

    parts: list[str] = []
    for message in messages[: leading_system_count(messages)]:
        parts.extend(part.text for part in message.content if type(part) is TextPart)
    return "\n\n".join(parts)


def late_system_note_text(message: Message) -> str:
    """Render a later system message for APIs without a mid-conversation system role."""

    return f"{LATE_SYSTEM_NOTE_OPEN}\n{system_message_text(message)}\n{LATE_SYSTEM_NOTE_CLOSE}"


def placed_conversation_messages(
    messages: Sequence[Message],
) -> list[tuple[int, Message, bool]]:
    """Return ``(index, message, is_late_system)`` for every non-leading message.

    Messages keep their order, except that a later system message found while
    tool results for the preceding assistant tool-call turn are still arriving
    is placed after those tool results. Deferral never crosses a user or
    assistant message.
    """

    placed: list[tuple[int, Message, bool]] = []
    deferred: list[tuple[int, Message, bool]] = []
    awaiting_tool_results = False
    for index in range(leading_system_count(messages), len(messages)):
        message = messages[index]
        if message.role is MessageRole.SYSTEM:
            (deferred if awaiting_tool_results else placed).append((index, message, True))
            continue
        if message.role is MessageRole.TOOL:
            placed.append((index, message, False))
            continue
        placed.extend(deferred)
        deferred.clear()
        placed.append((index, message, False))
        awaiting_tool_results = message.role is MessageRole.ASSISTANT and any(
            type(part) is ToolCallPart for part in message.content
        )
    placed.extend(deferred)
    return placed

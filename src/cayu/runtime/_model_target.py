from __future__ import annotations

from dataclasses import dataclass

from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    ThinkingPart,
    copy_message_part,
    detach_message,
)


@dataclass(frozen=True, slots=True)
class PortableTranscriptProjection:
    """Provider-neutral transcript plus evidence about removed opaque state."""

    messages: tuple[Message, ...]
    provider_state_parts_dropped: int
    thinking_parts_dropped: int
    source_prefix_count: int
    projected_prefix_count: int


def project_portable_transcript(messages: list[Message]) -> PortableTranscriptProjection:
    """Remove provider-native continuation state without changing semantic turns."""

    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return project_portable_transcript_prefix(messages, len(messages))


def project_portable_transcript_prefix(
    messages: list[Message],
    transcript_cursor: int,
) -> PortableTranscriptProjection:
    """Project only the prefix invalidated by the latest durable target switch.

    Text, neutral tool calls/results, and Cayu file references remain intact.
    Opaque response ids, cache handles, encrypted reasoning, and signed thinking
    blocks are removed. Assistant shells left with no portable parts are omitted
    from the model-facing copy; durable cursors continue to address the unchanged
    source rows rather than positions in this shorter projection.
    """

    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    if type(transcript_cursor) is not int or not 0 <= transcript_cursor <= len(messages):
        raise ValueError("Model-target projection cursor exceeds the session transcript.")
    projected: list[Message] = []
    provider_state_parts_dropped = 0
    thinking_parts_dropped = 0
    projected_prefix_count = 0
    for index, message in enumerate(messages):
        if type(message) is not Message:
            raise TypeError("Transcript messages must contain exact Message instances.")
        if index >= transcript_cursor or message.role is not MessageRole.ASSISTANT:
            projected.append(detach_message(message))
            if index < transcript_cursor:
                projected_prefix_count += 1
            continue
        parts = []
        for part in message.content:
            if type(part) is ProviderStatePart:
                provider_state_parts_dropped += 1
                continue
            if type(part) is ThinkingPart:
                thinking_parts_dropped += 1
                continue
            parts.append(copy_message_part(part))
        if not parts:
            continue
        projected.append(Message(role=message.role, content=tuple(parts)))
        projected_prefix_count += 1
    return PortableTranscriptProjection(
        messages=tuple(projected),
        provider_state_parts_dropped=provider_state_parts_dropped,
        thinking_parts_dropped=thinking_parts_dropped,
        source_prefix_count=transcript_cursor,
        projected_prefix_count=projected_prefix_count,
    )

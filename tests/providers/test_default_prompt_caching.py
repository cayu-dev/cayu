"""Prompt caching is on by default for Anthropic-shaped providers (#2289)."""

from __future__ import annotations

import pytest

from cayu import Message, ToolCallPart
from cayu.providers import (
    AnthropicProvider,
    BedrockProvider,
    CacheBreakpoint,
    CachePolicy,
    ModelRequest,
    VertexProvider,
    build_anthropic_payload,
    build_bedrock_converse_payload,
)
from cayu.providers.bedrock import bedrock_model_supports_default_prompt_caching

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]
_CLAUDE_ON_BEDROCK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _tool_turns() -> list[Message]:
    return [
        Message.text("system", "Rules."),
        Message.text("user", "Read A."),
        Message(
            role="assistant",
            content=[ToolCallPart(tool_call_id="call-1", tool_name="read_file", arguments={})],
        ),
        Message.tool_result(tool_call_id="call-1", tool_name="read_file", content="contents"),
    ]


def _anthropic_markers(payload: dict) -> list[tuple[str, int | None]]:
    markers: list[tuple[str, int | None]] = []
    if isinstance(payload.get("system"), list):
        markers.extend(("system", None) for block in payload["system"] if "cache_control" in block)
    markers.extend(("tools", None) for tool in payload.get("tools", ()) if "cache_control" in tool)
    for index, message in enumerate(payload["messages"]):
        markers.extend(
            ("messages", index) for block in message["content"] if "cache_control" in block
        )
    return markers


def test_default_cache_policy_covers_system_tools_and_conversation() -> None:
    assert CachePolicy().breakpoints == (
        CacheBreakpoint.SYSTEM_PROMPT,
        CacheBreakpoint.TOOL_DEFINITIONS,
        CacheBreakpoint.CONVERSATION_PREFIX,
    )
    assert CachePolicy().conversation_prefix_strategy == "all_but_last"


@pytest.mark.parametrize(
    "provider",
    [
        AnthropicProvider(api_key="test-key"),
        VertexProvider(project_id="test-project", credentials=object()),
    ],
    ids=["anthropic", "vertex"],
)
def test_anthropic_shaped_providers_cache_by_default(provider) -> None:
    request = ModelRequest(
        model="claude-haiku-4-5",
        messages=[*_tool_turns(), Message.text("system", "Turn state.")],
        tools=_TOOLS,
    )

    assert provider.cache_policy == CachePolicy()
    assert provider.request_cache_policy(request) == CachePolicy()
    projection = provider.request_cache_projection(request)
    assert projection is not None and projection.conversation_prefix is not None
    # The cached conversation prefix ends at the tool results, before the note.
    assert "Turn state" not in repr(projection.conversation_prefix)


def test_anthropic_markers_stay_within_the_limit_and_skip_late_notes() -> None:
    messages = [
        *_tool_turns(),
        Message.text("assistant", "Read it."),
        Message.text("system", "Turn state."),
        Message.text("user", "Next question."),
    ]
    payload = build_anthropic_payload(
        ModelRequest(model="claude-haiku-4-5", messages=messages, tools=_TOOLS),
        cache_policy=CachePolicy(),
    )

    markers = _anthropic_markers(payload)
    assert len(markers) <= 4
    # [user, assistant, user(tool result), assistant, user(note), user]:
    # all_but_last would end on the note, so the marker moves to the assistant turn.
    assert markers == [("system", None), ("tools", None), ("messages", 3)]


def test_anthropic_cache_opt_out_sends_no_markers() -> None:
    request = ModelRequest(model="claude-haiku-4-5", messages=_tool_turns(), tools=_TOOLS)
    disabled = AnthropicProvider(api_key="test-key", cache_policy=CachePolicy(breakpoints=()))

    assert disabled.request_cache_policy(request) == CachePolicy(breakpoints=())
    assert (
        _anthropic_markers(
            build_anthropic_payload(request, cache_policy=CachePolicy(breakpoints=()))
        )
        == []
    )


def _bedrock_cache_points(payload: dict) -> list[tuple[str, int | None]]:
    points: list[tuple[str, int | None]] = []
    points.extend(("system", None) for block in payload.get("system", ()) if "cachePoint" in block)
    points.extend(
        ("tools", None)
        for tool in payload.get("toolConfig", {}).get("tools", ())
        if "cachePoint" in tool
    )
    for index, message in enumerate(payload["messages"]):
        points.extend(("messages", index) for block in message["content"] if "cachePoint" in block)
    return points


def test_bedrock_places_cache_points_for_claude_by_default() -> None:
    provider = BedrockProvider(client=object(), region_name="us-east-1")
    request = ModelRequest(
        model=_CLAUDE_ON_BEDROCK,
        messages=[*_tool_turns(), Message.text("system", "Turn state.")],
        tools=_TOOLS,
    )

    policy = provider._resolved_cache_policy(request.model, request.options)
    payload = build_bedrock_converse_payload(request, cache_policy=policy)

    # [user, assistant(toolUse), user(toolResult + note)] -> the assistant turn.
    assert _bedrock_cache_points(payload) == [("system", None), ("tools", None), ("messages", 1)]
    assert payload["system"][-1] == {"cachePoint": {"type": "default"}}
    assert payload["messages"][2]["content"][-1] == {"text": "<system>\nTurn state.\n</system>"}
    assert provider.request_cache_policy(request) == CachePolicy()


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("global.anthropic.claude-opus-4-1-20250805-v1:0", True),
        ("eu.anthropic.claude-3-7-sonnet-20250219-v1:0", True),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            True,
        ),
        ("anthropic.claude-3-haiku-20240307-v1:0", False),
        ("meta.llama3-70b-instruct-v1:0", False),
        ("us.amazon.nova-lite-v1:0", False),
    ],
)
def test_bedrock_default_caching_is_limited_to_documented_claude_families(
    model: str, supported: bool
) -> None:
    assert bedrock_model_supports_default_prompt_caching(model) is supported
    provider = BedrockProvider(client=object(), region_name="us-east-1")
    expected = CachePolicy() if supported else None
    assert provider._resolved_cache_policy(model, {}) == expected


def test_bedrock_explicit_policy_applies_to_any_model_and_can_opt_out() -> None:
    request = ModelRequest(
        model="meta.llama3-70b-instruct-v1:0",
        messages=_tool_turns(),
        tools=_TOOLS,
    )
    explicit = BedrockProvider(
        client=object(),
        region_name="us-east-1",
        cache_policy=CachePolicy(breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,)),
    )
    disabled = BedrockProvider(
        client=object(), region_name="us-east-1", cache_policy=CachePolicy(breakpoints=())
    )

    assert explicit.request_cache_policy(request) == CachePolicy(
        breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,)
    )
    claude = request.model_copy(update={"model": _CLAUDE_ON_BEDROCK})
    assert disabled.request_cache_policy(claude) == CachePolicy(breakpoints=())
    off = build_bedrock_converse_payload(
        claude, cache_policy=disabled._resolved_cache_policy(claude.model, claude.options)
    )
    assert _bedrock_cache_points(off) == []


def test_bedrock_rejects_an_extended_ttl_it_cannot_map() -> None:
    with pytest.raises(ValueError, match="extended cache TTL"):
        BedrockProvider(
            client=object(), region_name="us-east-1", cache_policy=CachePolicy(ttl="extended")
        )


def test_bedrock_projects_the_cached_wire_prefix_with_attachment_digests() -> None:
    from cayu import (
        RESOLVED_FILE_ATTACHMENTS_OPTION,
        FileAttachmentKind,
        FilePart,
        TextPart,
        file_attachment,
    )

    attachment = file_attachment(
        artifact_id="image-1",
        kind=FileAttachmentKind.IMAGE,
        filename="chart.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model=_CLAUDE_ON_BEDROCK,
        messages=[
            Message.text("system", "Rules."),
            Message(
                role="user",
                content=[TextPart(text="Read the chart."), FilePart(attachment=attachment)],
            ),
            Message.text("assistant", "It shows growth."),
            Message.text("user", "Newest question."),
            Message.text("system", "Turn state."),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "image-1": {
                    "artifact_id": "image-1",
                    "kind": "image",
                    "filename": "chart.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )
    projection = BedrockProvider(client=object(), region_name="us-east-1").request_cache_projection(
        request
    )

    assert projection is not None
    assert CacheBreakpoint.CONVERSATION_PREFIX in projection.policy.breakpoints
    prefix = projection.conversation_prefix
    assert prefix is not None
    # The cache point ends on the assistant turn; the merged newest user turn
    # (question plus <system> note) is outside the cached prefix.
    assert [message["role"] for message in prefix] == ["user", "assistant"]
    assert "cachePoint" not in repr(prefix)
    assert "Newest question" not in repr(prefix) and "Turn state" not in repr(prefix)
    assert "bytes_sha256" in repr(prefix[0]["content"])

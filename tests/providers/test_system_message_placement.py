"""Per-provider placement of non-leading system messages (#2289).

The shared conformance suite proves append-only prefixes for every built-in
provider. These tests pin each provider's exact wire mapping, the
tool-protocol placement rule, and OpenAI ``prompt_cache_key`` pass-through.
"""

from __future__ import annotations

import pytest

import tests.providers.registrations as registrations_module
from cayu import Message, MessageRole, ProviderStatePart, ToolCallPart
from cayu.providers import (
    CacheBreakpoint,
    CachePolicy,
    ModelRequest,
    ModelStreamEventType,
    build_anthropic_payload,
    build_bedrock_converse_payload,
    build_chat_completions_payload,
    build_openai_payload,
)
from cayu.providers.openai import build_openai_token_count_payload

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


def _tool_call(call_id: str = "call-1") -> Message:
    return Message(
        role="assistant",
        content=[
            ToolCallPart(tool_call_id=call_id, tool_name="read_file", arguments={"path": "A"})
        ],
    )


def _tool_result(call_id: str = "call-1") -> Message:
    return Message.tool_result(tool_call_id=call_id, tool_name="read_file", content="contents")


def _request(messages: list[Message]) -> ModelRequest:
    return ModelRequest(model="test-model", messages=messages, tools=_TOOLS)


def test_openai_sends_late_system_messages_as_developer_items_in_place() -> None:
    payload = build_openai_payload(
        _request(
            [
                Message.text("system", "Rules."),
                Message.text("user", "Question."),
                Message.text("system", "Mid note."),
                Message.text("assistant", "Answer."),
                Message.text("system", "Turn state."),
            ]
        )
    )

    assert payload["instructions"] == "Rules."
    roles = [item.get("role", item.get("type")) for item in payload["input"]]
    assert roles == ["user", "developer", "assistant", "developer"]
    assert payload["input"][1] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "Mid note."}],
    }
    assert payload["input"][3]["content"] == [{"type": "input_text", "text": "Turn state."}]
    count_payload = build_openai_token_count_payload(
        _request(
            [
                Message.text("system", "Rules."),
                Message.text("user", "Question."),
                Message.text("system", "Turn state."),
            ]
        )
    )
    assert count_payload["instructions"] == "Rules."
    assert count_payload["input"][-1]["role"] == "developer"


def test_openai_places_late_system_message_after_pending_tool_outputs() -> None:
    payload = build_openai_payload(
        _request(
            [
                Message.text("user", "Read A."),
                _tool_call(),
                Message.text("system", "Turn state."),
                _tool_result(),
            ]
        )
    )

    types = [item.get("type", item.get("role")) for item in payload["input"]]
    assert types == ["user", "function_call", "function_call_output", "developer"]
    assert "instructions" not in payload


def test_openai_server_chain_sends_only_unowned_late_system_messages() -> None:
    prior_assistant = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ProviderStatePart(
                provider="openai",
                state={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ),
            ProviderStatePart(
                provider="openai",
                state={"type": "response_ref", "id": "resp_prev", "targeted_tool_marker_id": None},
            ),
        ],
    )
    payload = build_openai_payload(
        ModelRequest(
            model="gpt-test",
            messages=[
                Message.text("system", "Rules."),
                Message.text("user", "first"),
                Message.text("system", "Earlier state, already on the server."),
                prior_assistant,
                Message.text("user", "second"),
                Message.text("system", "Current state."),
            ],
        ),
        reasoning_state="server",
    )

    assert payload["previous_response_id"] == "resp_prev"
    assert payload["instructions"] == "Rules."
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
        {"role": "developer", "content": [{"type": "input_text", "text": "Current state."}]},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "registration",
    [registrations_module.OPENAI, registrations_module.OPENAI_SUBSCRIPTION],
    ids=lambda item: item.name,
)
async def test_openai_prompt_cache_key_passes_through_unchanged(registration) -> None:
    harness = await registration.factory("text")
    assert harness.sent_payloads is not None
    try:
        events = await harness.collect(
            ModelRequest(
                model=harness.model,
                messages=[Message.text("user", "hello")],
                options={"openai": {"prompt_cache_key": "lineage-7f3a"}},
            )
        )
        sent = harness.sent_payloads()
    finally:
        await harness.aclose()

    assert events[-1].type is ModelStreamEventType.COMPLETED
    assert [payload["prompt_cache_key"] for payload in sent] == ["lineage-7f3a"]


def test_chat_completions_keeps_late_system_messages_in_place() -> None:
    payload = build_chat_completions_payload(
        _request(
            [
                Message.text("system", "Rules."),
                Message.text("system", "More rules."),
                Message.text("user", "Read A."),
                _tool_call(),
                Message.text("system", "Deferred state."),
                _tool_result(),
                Message.text("system", "Turn state."),
            ]
        )
    )

    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "system",
        "system",
    ]
    assert messages[0]["content"] == "Rules.\n\nMore rules."
    assert messages[4] == {"role": "system", "content": "Deferred state."}
    assert messages[5] == {"role": "system", "content": "Turn state."}


def test_anthropic_renders_late_system_message_as_delimited_user_turn() -> None:
    payload = build_anthropic_payload(
        _request(
            [
                Message.text("system", "Rules."),
                Message.text("user", "Read A."),
                _tool_call(),
                Message.text("system", "Deferred state."),
                _tool_result(),
                Message.text("system", "Turn state."),
            ]
        )
    )

    assert payload["system"] == "Rules."
    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "user",
        "user",
    ]
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[3]["content"] == [
        {"type": "text", "text": "<system>\nDeferred state.\n</system>"}
    ]
    assert messages[4]["content"] == [{"type": "text", "text": "<system>\nTurn state.\n</system>"}]


def test_anthropic_cache_markers_stay_on_the_stable_prefix() -> None:
    policy = CachePolicy(
        breakpoints=(
            CacheBreakpoint.SYSTEM_PROMPT,
            CacheBreakpoint.TOOL_DEFINITIONS,
            CacheBreakpoint.CONVERSATION_PREFIX,
        )
    )
    payload = build_anthropic_payload(
        _request(
            [
                Message.text("system", "Rules."),
                Message.text("user", "Read A."),
                _tool_call(),
                _tool_result(),
                Message.text("system", "Turn state."),
            ]
        ),
        cache_policy=policy,
    )

    assert payload["system"] == [
        {"type": "text", "text": "Rules.", "cache_control": {"type": "ephemeral"}}
    ]
    marked = [
        (index, block)
        for index, message in enumerate(payload["messages"])
        for block in message["content"]
        if "cache_control" in block
    ]
    # The marker ends the cached history at the tool results, before the
    # trailing system note that changes every turn.
    assert [(index, block["type"]) for index, block in marked] == [(2, "tool_result")]
    assert "cache_control" not in payload["messages"][-1]["content"][0]


def test_bedrock_appends_late_system_message_to_the_user_side() -> None:
    payload = build_bedrock_converse_payload(
        ModelRequest(
            model="anthropic.claude-test",
            messages=[
                Message.text("system", "Rules."),
                Message.text("user", "Read A."),
                _tool_call(),
                Message.text("system", "Deferred state."),
                _tool_result(),
                Message.text("assistant", "Done."),
                Message.text("system", "Note before user."),
                Message.text("user", "Next."),
                Message.text("assistant", "Ok."),
                Message.text("system", "Turn state."),
            ],
            tools=_TOOLS,
        )
    )

    assert payload["system"] == [{"text": "Rules."}]
    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "toolResult" in messages[2]["content"][0]
    assert messages[2]["content"][1:] == [{"text": "<system>\nDeferred state.\n</system>"}]
    assert messages[4]["content"] == [
        {"text": "<system>\nNote before user.\n</system>"},
        {"text": "Next."},
    ]
    assert messages[6]["content"] == [{"text": "<system>\nTurn state.\n</system>"}]


@pytest.mark.parametrize(
    ("build", "items_key"),
    [
        (build_openai_payload, "input"),
        (build_chat_completions_payload, "messages"),
        (build_anthropic_payload, "messages"),
        (build_bedrock_converse_payload, "messages"),
    ],
    ids=["openai", "chat_completions", "anthropic", "bedrock"],
)
def test_prompt_cache_compaction_request_extends_the_warm_request(build, items_key) -> None:
    # PromptCacheCompactor appends one user instruction to the exact warm
    # request. A late system message must stay where the warm request sent it.
    warm_messages = [
        Message.text("system", "Rules."),
        Message.text("user", "Read A."),
        _tool_call(),
        _tool_result(),
        Message.text("system", "Turn state."),
    ]
    warm = build(_request(warm_messages))
    compaction = build(_request([*warm_messages, Message.text("user", "Summarize.")]))

    assert {key: value for key, value in warm.items() if key != items_key} == {
        key: value for key, value in compaction.items() if key != items_key
    }
    warm_items = warm[items_key]
    compaction_items = compaction[items_key]
    if build is build_bedrock_converse_payload:
        # Converse merges the appended user text into the note's user turn.
        assert compaction_items[:-1] == warm_items[:-1]
        assert compaction_items[-1]["content"][:-1] == warm_items[-1]["content"]
        return
    assert compaction_items[: len(warm_items)] == warm_items
    assert len(compaction_items) == len(warm_items) + 1


_AFFINITY_KEY = "cayu-0123456789abcdef0123456789abcdef"


def test_openai_maps_cache_affinity_key_to_prompt_cache_key() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        cache_affinity_key=_AFFINITY_KEY,
    )

    assert build_openai_payload(request)["prompt_cache_key"] == _AFFINITY_KEY
    # The Responses token-count endpoint takes no cache-routing field.
    assert "prompt_cache_key" not in build_openai_token_count_payload(request)
    overridden = request.model_copy(update={"options": {"openai": {"prompt_cache_key": "caller"}}})
    assert build_openai_payload(overridden)["prompt_cache_key"] == "caller"
    assert "prompt_cache_key" not in build_openai_payload(
        request.model_copy(update={"cache_affinity_key": None})
    )


@pytest.mark.parametrize(
    "build",
    [build_chat_completions_payload, build_anthropic_payload, build_bedrock_converse_payload],
    ids=["chat_completions", "anthropic", "bedrock"],
)
def test_non_openai_payloads_never_carry_the_cache_affinity_key(build) -> None:
    messages = [Message.text("system", "Rules."), Message.text("user", "hello")]
    with_key = build(
        ModelRequest(model="test-model", messages=messages, cache_affinity_key=_AFFINITY_KEY)
    )
    without_key = build(ModelRequest(model="test-model", messages=messages))

    assert with_key == without_key


@pytest.mark.anyio
@pytest.mark.parametrize(
    "registration",
    [
        registrations_module.OPENAI,
        registrations_module.OPENAI_SUBSCRIPTION,
        registrations_module.CHAT_COMPLETIONS,
        registrations_module.GATEWAY,
        registrations_module.ANTHROPIC,
        registrations_module.VERTEX,
        registrations_module.BEDROCK,
    ],
    ids=lambda item: item.name,
)
async def test_only_openai_responses_send_the_cache_affinity_key(registration) -> None:
    harness = await registration.factory("text")
    assert harness.sent_payloads is not None
    try:
        events = await harness.collect(
            ModelRequest(
                model=harness.model,
                messages=[Message.text("user", "hello")],
                cache_affinity_key=_AFFINITY_KEY,
            )
        )
        sent = harness.sent_payloads()
    finally:
        await harness.aclose()

    assert events[-1].type is ModelStreamEventType.COMPLETED
    openai_family = registration in {
        registrations_module.OPENAI,
        registrations_module.OPENAI_SUBSCRIPTION,
    }
    rendered = repr(sent)
    assert (_AFFINITY_KEY in rendered) is openai_family
    if openai_family:
        assert sent[0]["prompt_cache_key"] == _AFFINITY_KEY


@pytest.mark.parametrize(
    "value",
    ["", "x" * 65, "has space", "tenant/a", "caf\u00e9"],
)
def test_cache_affinity_key_rejects_non_token_values(value: str) -> None:
    with pytest.raises(ValueError, match="cache_affinity_key"):
        ModelRequest(model="m", messages=[Message.text("user", "hi")], cache_affinity_key=value)


def test_unset_cache_affinity_key_is_omitted_from_request_dumps() -> None:
    request = ModelRequest(model="m", messages=[Message.text("user", "hi")])

    assert "cache_affinity_key" not in request.model_dump(mode="json")
    keyed = request.model_copy(update={"cache_affinity_key": _AFFINITY_KEY})
    assert ModelRequest.model_validate(keyed.model_dump(mode="json")) == keyed

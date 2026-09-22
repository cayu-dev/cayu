"""Synthetic coverage of shared OpenAI stream lifecycle and retry contracts."""

from copy import deepcopy

import pytest
from tests.core import test_openai_function_ordering as function
from tests.core import test_openai_search_ordering as search
from tests.core.test_openai_overload_identity import failure
from tests.providers._responses_ordering import run_ordering_attempts

from cayu import EventType
from cayu.providers import ModelStreamEventType
from cayu.providers.openai import OpenAIProtocolError


def shifted_completion():
    reasoning = {"type": "reasoning", "id": "rs_fixture", "summary": []}
    terminal = function.terminal((1,))
    terminal["response"]["output"].insert(0, reasoning)
    return [
        function.created(),
        {"type": "response.output_item.added", "output_index": 0, "item": reasoning},
        {"type": "response.output_item.done", "output_index": 1, "item": reasoning},
        function.added(1),
        function.delta(1),
        {**function.arguments_done(1), "output_index": 4},
        {**function.done(1), "output_index": 4},
        terminal,
    ]


@pytest.mark.anyio
async def test_shifted_completions_preserve_reasoning_and_exact_function():
    seen = []
    await function.parse(shifted_completion(), seen)
    calls = [e for e in seen if e.type == ModelStreamEventType.TOOL_CALL]
    assert len(calls) == 1
    assert calls[0].payload["id"] == "call_1"
    assert calls[0].payload["arguments"] == {"text": "safe"}
    assert seen[-1].type == ModelStreamEventType.COMPLETED
    assert seen[-1].payload["provider_state"][0]["state"]["id"] == "rs_fixture"


@pytest.mark.anyio
@pytest.mark.parametrize("conflict", ["occupied", "arguments", "terminal"])
async def test_reconciliation_does_not_override_conflicting_evidence(conflict):
    raw = shifted_completion()
    if conflict == "occupied":
        raw.insert(5, function.added(4))
    elif conflict == "arguments":
        raw[5]["arguments"] = '{"text":"different"}'
    else:
        raw[-1] = deepcopy(raw[-1])
        raw[-1]["response"]["output"][0]["id"] = "other-reasoning"
    seen = []
    with pytest.raises(OpenAIProtocolError):
        await function.parse(raw, seen)
    assert not any(e.type == ModelStreamEventType.COMPLETED for e in seen)


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
async def test_searching_registration_and_completed_message_reach_terminal(tmp_path, adapter):
    raw = search.normal()
    raw[1]["item"]["status"] = "searching"
    message = deepcopy(raw[-1]["response"]["output"][-1])
    raw[-1:-1] = [
        {"type": "response.output_item.added", "output_index": 1, "item": message},
        {"type": "response.output_item.done", "output_index": 1, "item": message},
    ]
    events, durable, calls = await run_ordering_attempts(tmp_path, adapter, [raw])
    assert not calls
    assert not [e for e in durable if e.type == EventType.MODEL_ERROR]
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
async def test_overload_then_protocol_failure_can_use_third_attempt(tmp_path, adapter):
    # Unregistered identity remains rejected; the next response may be valid.
    malformed = [function.created(), function.arguments_done()]
    events, durable, calls = await run_ordering_attempts(
        tmp_path, adapter, [[failure()], malformed, search.normal()]
    )
    assert not calls
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    assert errors[1]["attempt"] == 2
    assert errors[1]["retry"] is True
    assert errors[1]["provider_retryable"] is True
    assert errors[1]["effective_max_attempts"] == 5
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
async def test_reconciled_native_stream_executes_function_exactly_once(tmp_path, adapter):
    events, durable, calls = await run_ordering_attempts(
        tmp_path, adapter, [shifted_completion(), search.normal()]
    )
    assert calls == [{"text": "safe"}]
    assert len([e for e in durable if e.type == EventType.TOOL_CALL_STARTED]) == 1
    assert not [e for e in durable if e.type == EventType.MODEL_ERROR]
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("item_index", [0, 1])
@pytest.mark.parametrize("omitted_at", ["item_done", "terminal"])
async def test_reconciled_completion_accepts_optional_status(item_index, omitted_at):
    raw = shifted_completion()
    raw[-1] = deepcopy(raw[-1])
    done_item = raw[2 if item_index == 0 else 6]["item"]
    terminal_item = raw[-1]["response"]["output"][item_index]
    done_item["status"] = terminal_item["status"] = "completed"
    (done_item if omitted_at == "item_done" else terminal_item).pop("status")
    seen = []
    await function.parse(raw, seen)
    assert seen[-1].type == ModelStreamEventType.COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("item_index", [0, 1])
async def test_reconciled_completion_rejects_nonterminal_status(item_index):
    raw = shifted_completion()
    raw[-1] = deepcopy(raw[-1])
    raw[-1]["response"]["output"][item_index]["status"] = "in_progress"
    with pytest.raises(OpenAIProtocolError) as caught:
        await function.parse(raw, [])
    assert caught.value.reason_code == "terminal_reconciled_item_conflicts_with_lifecycle_evidence"

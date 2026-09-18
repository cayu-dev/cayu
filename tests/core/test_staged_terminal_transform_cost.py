"""A terminal transition owns and parses one full checkpoint snapshot."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from tests.core.test_tool_round_publication import _lifecycle_events, _pending_round

import cayu.runtime._tool_round_recovery as recovery
from cayu.runtime._assistant_tool_round_publication import StagedToolCallTerminal


def state():
    pending = _pending_round()
    event = next(e for e in _lifecycle_events(pending) if str(e.type) == "tool.call.completed")
    pending.staged_terminals = [StagedToolCallTerminal(tool_call_id="call-a", event=event)]
    return (
        {
            "retained": {"nested": ["unchanged"]},
            "pending_tool_round": pending.model_dump(mode="json"),
        },
        event,
        recovery.pending_tool_round_identity(pending),
    )


@pytest.mark.parametrize("kind", ["projection", "completion", "timing"])
def test_terminal_transition_parses_owner_once_and_detaches(monkeypatch, kind):
    checkpoint, event, identity = state()
    before = deepcopy(checkpoint)
    seen = []
    parse = recovery._pending_tool_round_from_owned_checkpoint

    def count(*args, **kwargs):
        seen.append(1)
        return parse(*args, **kwargs)

    monkeypatch.setattr(recovery, "_pending_tool_round_from_owned_checkpoint", count)
    if kind == "timing":
        now = datetime.now(UTC)
        transform = recovery.started_staged_terminal_publication_transform(
            tool_round_identity=identity,
            tool_call_id="call-a",
            event=event,
            payload_bytes=10,
            effect_completed_at=now,
            staged_at=now,
            publication_started_at=now,
        )
    else:
        factory = (
            recovery.projected_staged_terminal_transform
            if kind == "projection"
            else recovery.completed_staged_terminal_transform
        )
        transform = factory(tool_round_identity=identity, event=event)
    updated = transform(None, checkpoint)
    assert len(seen) == 1
    assert checkpoint == before
    updated["retained"]["nested"].append("changed")
    assert checkpoint == before
    fresh = transform(None, checkpoint)
    assert fresh["retained"] == before["retained"]
    assert fresh["pending_tool_round"]["staged_terminals"][0]["hooks_state"] == (
        "completed" if kind == "completion" else "pending"
    )


@pytest.mark.parametrize("change", ["unrelated_invalid", "foreign_round", "foreign_call"])
def test_transition_retains_complete_document_and_identity_checks(change):
    checkpoint, event, identity = state()
    if change == "unrelated_invalid":
        checkpoint["retained"]["bad"] = object()
    elif change == "foreign_round":
        identity = identity.model_copy(update={"tool_round_id": "tround_" + "9" * 32})
    else:
        event = event.model_copy(update={"payload": {**event.payload, "tool_call_id": "foreign"}})
    transform = recovery.completed_staged_terminal_transform(
        tool_round_identity=identity, event=event
    )
    with pytest.raises((ValueError, RuntimeError)):
        transform(None, checkpoint)


def test_owned_json_parse_skips_second_stage_validation_but_never_caches(monkeypatch):
    checkpoint, _, _ = state()
    original = StagedToolCallTerminal.model_validate
    copies = []

    def count(*args, **kwargs):
        copies.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(StagedToolCallTerminal, "model_validate", count)
    first = recovery.pending_tool_round_from_checkpoint(checkpoint)
    assert first is not None
    assert copies == []
    checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = "new query"
    second = recovery.pending_tool_round_from_checkpoint(checkpoint)
    assert second.tool_calls[0].arguments["query"] == "new query"
    assert first.tool_calls[0].arguments["query"] == "alpha"
    second.tool_calls[0].arguments["query"] = "result mutation"
    assert checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] == "new query"
    # Public construction with pre-existing model instances still detaches and
    # revalidates those instances; it cannot claim the private JSON boundary.
    recovery.PendingToolRound.model_validate(
        {**checkpoint["pending_tool_round"], "staged_terminals": first.staged_terminals}
    )
    assert copies


def test_public_model_constructed_stage_is_not_trusted():
    checkpoint, event, _ = state()
    invalid = StagedToolCallTerminal.model_construct(tool_call_id="other", event=event)
    checkpoint["pending_tool_round"]["staged_terminals"] = [invalid]
    with pytest.raises(ValueError):
        recovery.PendingToolRound.model_validate(checkpoint["pending_tool_round"])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("count", [1, 10, 20])
def test_native_round_delivers_all_results_with_bounded_parse_work(
    tmp_path, monkeypatch, backend, count
):
    import asyncio

    from tests.core._event_projection_support import private_events_for_public_events
    from tests.core.test_tool_round_runtime_publication import _EchoTool

    from cayu import (
        AgentSpec,
        CayuApp,
        InMemorySessionStore,
        Message,
        ModelStreamEvent,
        RunRequest,
        ScriptedModelProvider,
    )
    from cayu.messages import ToolResultPart
    from cayu.storage.sqlite import SQLiteSessionStore

    parses = 0
    original = recovery._pending_tool_round_from_owned_checkpoint

    def observed(*args, **kwargs):
        nonlocal parses
        parses += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "_pending_tool_round_from_owned_checkpoint", observed)

    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "round.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(
                            name="echo", id=f"call-{i}", arguments={"text": f"result-{i}"}
                        )
                        for i in range(count)
                    ],
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="test"), tools=[_EchoTool()])
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="worker", messages=[Message.text("user", "Run the calls.")]
                    )
                )
            ]
            assert str(events[-1].type) == "session.completed"
            events = await private_events_for_public_events(store, events)
            completed = [event for event in events if str(event.type) == "tool.call.completed"]
            assert {event.payload["tool_call_id"] for event in completed} == {
                f"call-{i}" for i in range(count)
            }
            assert len(completed) == count
            # Verify the continuation receives every result, not only events.
            assert len(provider.requests) == 2
            results = [
                part
                for message in provider.requests[-1].messages
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(results) == count
            assert {part.tool_call_id: part.content for part in results} == {
                f"call-{i}": f"result-{i}" for i in range(count)
            }
            # Bound actual round parses separately for each native backend.
            # Base e829882e5 needs 29 + 30*N (memory), 9 + 13*N (SQLite).
            assert parses <= (29 + 22 * count if backend == "memory" else 9 + 11 * count)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())

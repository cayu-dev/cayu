from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest, SessionStatus
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _ScriptedProvider(ModelProvider):
    name = "scripted-model-publication"

    def __init__(self, attempts: list[list[ModelStreamEvent]]) -> None:
        self._attempts = attempts
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        attempt = self._attempts[self.calls]
        self.calls += 1
        for event in attempt:
            yield event


class _EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo one value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        value = str(args["value"])
        self.calls.append(value)
        return ToolResult(content=value)


class _PromotionAcknowledgementLostStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False

    async def _promote_model_completion_stage_atomic(self, **kwargs):
        result = await super()._promote_model_completion_stage_atomic(**kwargs)
        if not self.lost_acknowledgement and result.replayed is False:
            self.lost_acknowledgement = True
            raise ConnectionError("model promotion acknowledgement lost")
        return result


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _completed_payload() -> dict:
    return {
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        },
    }


def test_session_engine_publishes_one_authoritative_assistant_turn() -> None:
    store = InMemorySessionStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("durable answer"),
                ModelStreamEvent.completed(_completed_payload()),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-publication-no-tools",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    completed_event = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    logical_step_id = completed_event.payload["model_step_id"]
    receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "model-publication-no-tools",
            logical_step_id,
        )
    )
    transcript = asyncio.run(store.load_transcript("model-publication-no-tools"))
    durable_events = asyncio.run(store.load_events("model-publication-no-tools"))

    assert provider.calls == 1
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].text == "durable answer"
    assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 1
    assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1
    assert receipt is not None
    assert receipt.kind == "model-step"
    assert receipt.transcript_start_cursor == 1
    assert receipt.transcript_end_cursor == 2
    pointer = model_completion_publication.model_step_publication_from_checkpoint(
        asyncio.run(store.load_checkpoint("model-publication-no-tools"))
    )
    assert pointer is not None
    assert pointer.logical_step_id == logical_step_id
    assert pointer.transcript_end_cursor == 2
    assert pointer.tool_round_id is None
    assert (
        asyncio.run(store.load_active_model_completion_stage("model-publication-no-tools")) is None
    )


def test_model_publication_redacts_provider_metadata_before_durable_commit() -> None:
    secret = "model-completion-durable-secret"
    store = InMemorySessionStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("safe answer"),
                ModelStreamEvent.completed(
                    {
                        **_completed_payload(),
                        "provider_debug": {"credential": secret},
                    }
                ),
            ]
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-publication-redacted-metadata",
                messages=[Message.text("user", "answer")],
            ),
        )
    )
    durable_events = asyncio.run(store.load_events("model-publication-redacted-metadata"))
    returned = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    durable = next(event for event in durable_events if event.type == EventType.MODEL_COMPLETED)

    assert returned == durable
    assert durable.payload["provider_debug"] == {"credential": REDACTED_SECRET}
    assert secret not in durable.model_dump_json()


def test_model_publication_redacts_ordinary_assistant_before_durable_commit() -> None:
    secret = "ordinary-assistant-durable-secret"
    store = InMemorySessionStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta(f"answer with {secret}"),
                ModelStreamEvent.completed(_completed_payload()),
            ]
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "model-publication-redacted-assistant"

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "answer")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert transcript[-1].content[0].text == f"answer with {REDACTED_SECRET}"
    assert secret not in str([message.model_dump(mode="json") for message in transcript])


def test_model_publication_preserves_runtime_classification_when_secret_overlaps() -> None:
    store = InMemorySessionStore()
    provider = _ScriptedProvider([[ModelStreamEvent.completed(_completed_payload())]])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor("id"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-protocol-secret-overlap",
                messages=[Message.text("user", "answer")],
            ),
        )
    )
    durable_event = next(
        event
        for event in asyncio.run(store.load_events("model-protocol-secret-overlap"))
        if event.type == EventType.MODEL_COMPLETED
    )
    pointer = model_completion_publication.model_step_publication_from_checkpoint(
        asyncio.run(store.load_checkpoint("model-protocol-secret-overlap"))
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert durable_event.payload["step_classification"]["type"] == "invalid"
    assert pointer is not None
    assert pointer.classification["type"] == "invalid"


def test_session_engine_links_model_and_tool_round_publications() -> None:
    store = InMemorySessionStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-echo",
                    name="echo",
                    arguments={"value": "once"},
                ),
                ModelStreamEvent.completed(
                    {
                        **_completed_payload(),
                        "finish_reason": "tool_calls",
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("finished"),
                ModelStreamEvent.completed(_completed_payload()),
            ],
        ]
    )
    tool = _EchoTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-publication-tools",
                messages=[Message.text("user", "use echo")],
            ),
        )
    )

    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    first_step_id = completed_events[0].payload["model_step_id"]
    second_step_id = completed_events[1].payload["model_step_id"]
    tool_round_id = completed_events[0].payload["tool_round_id"]
    first_receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "model-publication-tools",
            first_step_id,
        )
    )
    tool_receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "model-publication-tools",
            f"tool-round:{tool_round_id}",
        )
    )
    second_receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "model-publication-tools",
            second_step_id,
        )
    )
    transcript = asyncio.run(store.load_transcript("model-publication-tools"))

    assert provider.calls == 2
    assert tool.calls == ["once"]
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert first_receipt is not None
    assert first_receipt.transcript_start_cursor == 1
    assert first_receipt.transcript_end_cursor == 2
    assert tool_receipt is not None
    assert tool_receipt.transcript_start_cursor == 2
    assert tool_receipt.transcript_end_cursor == 3
    assert second_receipt is not None
    assert second_receipt.transcript_start_cursor == 3
    assert second_receipt.transcript_end_cursor == 4
    pointer = model_completion_publication.model_step_publication_from_checkpoint(
        asyncio.run(store.load_checkpoint("model-publication-tools"))
    )
    assert pointer is not None
    assert pointer.logical_step_id == second_step_id
    assert pointer.transcript_end_cursor == 4
    assert pointer.tool_round_id is None


def test_model_promotion_acknowledgement_loss_replays_without_duplication() -> None:
    store = _PromotionAcknowledgementLostStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("committed before the lost acknowledgement"),
                ModelStreamEvent.completed(_completed_payload()),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-promotion-ack-loss",
                messages=[Message.text("user", "answer once")],
            ),
        )
    )

    logical_step_id = next(
        event.payload["model_step_id"]
        for event in events
        if event.type == EventType.MODEL_COMPLETED
    )
    session = asyncio.run(store.load("model-promotion-ack-loss"))
    transcript = asyncio.run(store.load_transcript("model-promotion-ack-loss"))
    durable_events = asyncio.run(store.load_events("model-promotion-ack-loss"))
    receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "model-promotion-ack-loss",
            logical_step_id,
        )
    )

    assert provider.calls == 1
    assert store.lost_acknowledgement is True
    assert session is not None and session.status == SessionStatus.COMPLETED
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 1
    assert receipt is not None
    assert asyncio.run(store.load_active_model_completion_stage(session.id)) is None

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EventQuery,
    EventRecord,
    InMemorySessionStore,
    RunRequest,
    SessionStatus,
)
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime._event_projection import public_event_sequence
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
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False

    async def _promote_model_completion_stage_atomic(self, **kwargs):
        result = await super()._promote_model_completion_stage_atomic(**kwargs)
        if not self.lost_acknowledgement and result.replayed is False:
            self.lost_acknowledgement = True
            raise ConnectionError("model promotion acknowledgement lost")
        return result


class _LegacyPublicationOverrideStore(InMemorySessionStore):
    """Exercise the established public publication signatures without catch-all kwargs."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.publications = 0
        self.stage_completions = 0
        self.stage_promotions = 0

    async def publish_runtime_publication(
        self,
        session_id,
        *,
        request,
        expected_statuses=None,
        expected_run_epoch=None,
        expected_transcript_cursor=None,
    ):
        self.publications += 1
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def complete_model_completion_stage(
        self,
        session_id,
        *,
        stage_id,
        publication,
    ):
        self.stage_completions += 1
        return await super().complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )

    async def promote_model_completion_stage(
        self,
        session_id,
        *,
        stage_id,
        expected_run_epoch,
    ):
        self.stage_promotions += 1
        return await super().promote_model_completion_stage(
            session_id,
            stage_id=stage_id,
            expected_run_epoch=expected_run_epoch,
        )


class _PreAssistantPublicationStageStore(InMemorySessionStore):
    """Store one terminal tool-round stage in the additive pre-field v2 shape."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.stripped_legacy_field = False

    async def complete_model_completion_stage(
        self,
        session_id,
        *,
        stage_id,
        publication,
    ):
        operations = []
        for operation in publication.mutation.operations:
            if operation.key != "pending_tool_round" or type(operation.value) is not dict:
                operations.append(operation)
                continue
            legacy_value = dict(operation.value)
            if "assistant_publication" in legacy_value:
                self.stripped_legacy_field = True
                legacy_value.pop("assistant_publication")
            operations.append(operation.model_copy(update={"value": legacy_value}, deep=True))
        legacy_mutation = publication.mutation.model_copy(
            update={"operations": tuple(operations)},
            deep=True,
        )
        legacy_publication = publication.model_copy(
            update={"mutation": legacy_mutation},
            deep=True,
        )
        return await super().complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=legacy_publication,
        )


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


async def _model_completion_records(
    store: InMemorySessionStore,
    session_id: str,
) -> list[EventRecord]:
    return await store.query_events(
        EventQuery(
            session_id=session_id,
            event_type=EventType.MODEL_COMPLETED,
            limit=100,
        )
    )


def _completed_payload() -> dict:
    return {
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        },
    }


def test_runtime_checkpoint_codec_preserves_legacy_publication_overrides() -> None:
    store = _LegacyPublicationOverrideStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="legacy-publication-call",
                    name="echo",
                    arguments={"value": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
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
                session_id="legacy-publication-overrides",
                messages=[Message.text("user", "echo")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert tool.calls == ["first"]
    assert store.publications == 1
    assert store.stage_completions == 2
    assert store.stage_promotions == 2


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

    completion_records = asyncio.run(_model_completion_records(store, "model-publication-no-tools"))
    logical_step_id = completion_records[0].event.payload["model_step_id"]
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

    completion_record = asyncio.run(
        _model_completion_records(store, "model-publication-redacted-metadata")
    )[0]
    assert completion_record.event == durable
    assert completion_record.sequence == public_event_sequence(returned.id)
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

    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-publication-tools",
                messages=[Message.text("user", "use echo")],
            ),
        )
    )

    completed_records = asyncio.run(_model_completion_records(store, "model-publication-tools"))
    first_step_id = completed_records[0].event.payload["model_step_id"]
    second_step_id = completed_records[1].event.payload["model_step_id"]
    tool_round_id = completed_records[0].event.payload["tool_round_id"]
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
    assert first_receipt.transcript_end_cursor == 1
    assert tool_receipt is not None
    assert tool_receipt.transcript_start_cursor == 1
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


def test_pre_upgrade_v2_tool_round_stage_promotes_without_redispatch() -> None:
    store = _PreAssistantPublicationStageStore()
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="legacy-v2-call",
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
                session_id="pre-assistant-publication-v2",
                messages=[Message.text("user", "use echo")],
            ),
        )
    )

    assert store.stripped_legacy_field is True
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert provider.calls == 2
    assert tool.calls == ["once"]
    assert [
        message.role
        for message in asyncio.run(store.load_transcript("pre-assistant-publication-v2"))
    ] == ["user", "assistant", "tool", "assistant"]


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

    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="model-promotion-ack-loss",
                messages=[Message.text("user", "answer once")],
            ),
        )
    )

    session = asyncio.run(store.load("model-promotion-ack-loss"))
    transcript = asyncio.run(store.load_transcript("model-promotion-ack-loss"))
    durable_events = asyncio.run(store.load_events("model-promotion-ack-loss"))
    logical_step_id = next(
        event.payload["model_step_id"]
        for event in durable_events
        if event.type == EventType.MODEL_COMPLETED
    )
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

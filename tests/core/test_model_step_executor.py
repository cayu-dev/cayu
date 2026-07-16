from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import cayu.runtime._model_step_executor as model_step_executor_module
from cayu.artifacts import (
    ArtifactStoreUnavailableError,
    LocalArtifactStore,
    file_attachment,
)
from cayu.core import AgentSpec, Event, EventType, Message, TextPart
from cayu.core.messages import FilePart
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    CayuApp,
    EventOrder,
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionIdentity,
    StructuredOutputSpec,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


def test_context_usage_state_uses_one_latest_completed_event_query() -> None:
    class TrackingStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.event_queries: list[EventQuery] = []

        async def query_events(self, query: EventQuery | None = None):
            if query is not None:
                self.event_queries.append(query)
            return await super().query_events(query)

    store = TrackingStore()
    session_id = "usage_context_policy_latest_event"

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    payload={
                        "model": "fake-model",
                        "usage": {"input_tokens": index, "output_tokens": 1},
                    },
                )
                for index in range(1, 106)
            ],
        )
        return await model_step_executor_module._context_usage_state_for_session(
            session_store=store,
            session_id=session_id,
        )

    usage = asyncio.run(run())

    assert usage.last_input_tokens == 105
    assert usage.last_output_tokens == 1
    assert usage.last_total_tokens == 106
    assert usage.last_provider_name is None
    assert usage.last_requested_model is None
    assert usage.last_model == "fake-model"
    assert len(store.event_queries) == 1
    assert store.event_queries[0].limit == 1
    assert store.event_queries[0].order_by == EventOrder.SEQUENCE_DESC


def test_model_step_executor_builds_defensive_structured_output_request() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={"nested": {"value": "agent"}},
            provider_options={"nested": {"value": "provider"}},
        )
    )
    registered_agent = app._agents["assistant"]
    session = Session(
        id="sess_request",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="sess_request",
    )
    messages = [Message.text("user", "answer as JSON")]
    spec = StructuredOutputSpec(
        name="answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    request = asyncio.run(
        app._model_step_executor.build_request(
            session=session,
            registered_agent=registered_agent,
            registered_environment=None,
            context_messages=messages,
            structured_output=spec,
            thinking=None,
            step=1,
        )
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert request.messages[0].role == "system"
    assert request.messages[1].model_dump() == messages[0].model_dump()
    assert request.tools[-1]["name"] == STRUCTURED_OUTPUT_TOOL_NAME
    assert request.options["structured_output"]["name"] == "answer"

    request.options["nested"]["value"] = "mutated"
    request.options["agent_metadata"]["nested"]["value"] = "mutated"
    assert registered_agent.spec.provider_options == {"nested": {"value": "provider"}}
    assert registered_agent.spec.metadata == {"nested": {"value": "agent"}}


def test_model_step_executor_retries_synchronous_stream_construction_failure() -> None:
    class RetryOnceProvider(ModelProvider):
        name = "retry-once"

        def __init__(self) -> None:
            self.calls = 0

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 1:
                raise ModelProviderError(
                    "temporary provider failure",
                    provider=self.name,
                    retryable=True,
                )

            async def events() -> AsyncIterator[ModelStreamEvent]:
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

            return events()

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = RetryOnceProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_executor",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        )
        user_message = Message.text("user", "hello")
        await store.append_transcript_messages(session.id, [user_message])
        registered_agent = app._agents["assistant"]
        registered_provider = app._providers[provider.name]
        request = await app._model_step_executor.build_request(
            session=session,
            registered_agent=registered_agent,
            registered_environment=None,
            context_messages=[user_message],
            structured_output=None,
            thinking=None,
            step=1,
        )
        preparation_calls = 0
        dispatch_calls = 0
        completed_events: list[Event] = []

        async def prepare_provider_dispatch():
            nonlocal preparation_calls
            preparation_calls += 1
            return [], None, None

        async def before_provider_dispatch() -> None:
            nonlocal dispatch_calls
            dispatch_calls += 1

        events: list[Event] = []
        result = None
        async for event, step_result in app._model_step_executor.run_with_retries(
            provider=provider,
            model_request=request,
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=None,
            step=1,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            transcript_cursor_before_request=1,
            record_model_completion=completed_events.append,
            prepare_provider_dispatch=prepare_provider_dispatch,
            before_provider_dispatch=before_provider_dispatch,
        ):
            if event is not None:
                events.append(event)
            if step_result is not None:
                result = step_result

        return (
            provider,
            events,
            result,
            completed_events,
            preparation_calls,
            dispatch_calls,
            await store.load_transcript(session.id),
        )

    (
        provider,
        events,
        result,
        completed_events,
        preparation_calls,
        dispatch_calls,
        transcript,
    ) = asyncio.run(run())

    assert provider.calls == 2
    assert preparation_calls == 2
    assert dispatch_calls == 2
    assert [event.type for event in events].count(EventType.MODEL_STARTED) == 2
    assert EventType.MODEL_RETRY in [event.type for event in events]
    assert EventType.MODEL_ATTEMPT_DISCARDED in [event.type for event in events]
    assert len(completed_events) == 1
    assert result is not None
    assert result.text_content == "done"
    assert transcript == [Message.text("user", "hello")]


def test_file_attachment_refs_include_prompt_and_tool_result_provenance() -> None:
    user_attachment = file_attachment(
        artifact_id="art_user",
        kind="image",
        filename="photo.png",
        content_type="image/png",
        size_bytes=5,
    )
    tool_attachment = file_attachment(
        artifact_id="art_tool",
        kind="document",
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=9,
    )
    messages = [
        Message(
            role="user",
            content=[TextPart(text="look"), FilePart(attachment=user_attachment)],
        ),
        Message.tool_call(tool_call_id="call_1", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="read_file",
            content="attached",
            artifacts=[tool_attachment],
        ),
    ]

    refs, prompt_ids, tool_ids = model_step_executor_module._file_attachment_refs(messages)

    assert [ref.artifact_id for ref in refs] == ["art_user", "art_tool"]
    assert prompt_ids == {"art_user"}
    assert tool_ids == {"art_tool"}

    conflicting = file_attachment(
        artifact_id="art_user",
        kind="image",
        filename="different.png",
        content_type="image/png",
        size_bytes=7,
    )
    with pytest.raises(RuntimeError, match="Conflicting file attachment"):
        model_step_executor_module._file_attachment_refs(
            [
                *messages,
                Message(role="user", content=[FilePart(attachment=conflicting)]),
            ]
        )


def test_attachment_resolution_fails_open_only_for_prompt_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    prompt_missing_id = f"art_{'a' * 32}"
    tool_missing_id = f"art_{'b' * 32}"
    shared_missing_id = f"art_{'c' * 32}"
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    registered_environment = app._environments["local"]
    session = Session(
        id="sess_resolve",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="sess_resolve",
    )

    def missing(artifact_id: str):
        return file_attachment(
            artifact_id=artifact_id,
            kind="image",
            filename="ghost.png",
            content_type="image/png",
            size_bytes=5,
        )

    async def resolve(messages: list[Message]):
        return await model_step_executor_module._resolved_file_attachments(
            messages=messages,
            session=session,
            registered_environment=registered_environment,
            max_file_attachment_bytes=8_000_000,
            max_total_file_attachment_bytes=32_000_000,
            max_file_attachments_per_request=20,
        )

    prompt_only = [Message(role="user", content=[FilePart(attachment=missing(prompt_missing_id))])]
    resolved, unresolvable = asyncio.run(resolve(prompt_only))
    assert resolved == {}
    assert unresolvable == {prompt_missing_id}

    malformed = [Message(role="user", content=[FilePart(attachment=missing("nope"))])]
    resolved, unresolvable = asyncio.run(resolve(malformed))
    assert resolved == {}
    assert unresolvable == {"nope"}

    tool_only = [
        Message.tool_call(tool_call_id="c1", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="c1",
            tool_name="read_file",
            content="x",
            artifacts=[missing(tool_missing_id)],
        ),
    ]
    with pytest.raises(FileNotFoundError):
        asyncio.run(resolve(tool_only))

    shared = [
        Message(role="user", content=[FilePart(attachment=missing(shared_missing_id))]),
        Message.tool_call(tool_call_id="c2", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="c2",
            tool_name="read_file",
            content="x",
            artifacts=[missing(shared_missing_id)],
        ),
    ]
    with pytest.raises(FileNotFoundError):
        asyncio.run(resolve(shared))

    async def permission_failure(_artifact_id: str, *, max_bytes: int | None = None):
        raise PermissionError("Artifact backend permission denied.")

    monkeypatch.setattr(store, "read_bytes", permission_failure)
    with pytest.raises(PermissionError, match="permission denied"):
        asyncio.run(resolve(prompt_only))

    async def unavailable_failure(_artifact_id: str, *, max_bytes: int | None = None):
        raise ArtifactStoreUnavailableError("Artifact backend unavailable.")

    monkeypatch.setattr(store, "read_bytes", unavailable_failure)
    with pytest.raises(ArtifactStoreUnavailableError, match="backend unavailable"):
        asyncio.run(resolve(prompt_only))

    async def invalid_store_failure(_artifact_id: str, *, max_bytes: int | None = None):
        raise ValueError("Artifact store returned invalid data.")

    monkeypatch.setattr(store, "read_bytes", invalid_store_failure)
    with pytest.raises(ValueError, match="invalid data"):
        asyncio.run(resolve(prompt_only))

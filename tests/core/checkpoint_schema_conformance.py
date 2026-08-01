from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cayu.core import AgentSpec, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    PendingActionQuery,
    RunRequest,
    SessionStatus,
    UserInputResponse,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CheckpointCompatibilityError,
)
from cayu.runtime.sessions import SessionStore
from cayu.tools.user_input import UserInputTool


class _PauseForInputProvider(ModelProvider):
    name = "checkpoint-conformance"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.tool_call(
            id="checkpoint_conformance_input",
            name="ask_user",
            arguments={"question": "Continue?"},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _CompleteProvider(ModelProvider):
    name = "checkpoint-conformance"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        yield ModelStreamEvent.text_delta("continued")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(store: SessionStore, provider: ModelProvider) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="checkpoint-agent", model="checkpoint-model"),
        tools=[UserInputTool()],
    )
    return app


async def _pause_session(
    store: SessionStore,
    *,
    session_id: str,
) -> str:
    provider = _PauseForInputProvider()
    events = [
        event
        async for event in _app(store, provider).run(
            RunRequest(
                agent_name="checkpoint-agent",
                session_id=session_id,
                messages=[Message.text("user", "pause")],
            )
        )
    ]
    assert provider.calls == 1
    awaiting = next(
        event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
    )
    return awaiting.payload["input_id"]


async def assert_versionless_checkpoint_resume_conformance(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    input_id = await _pause_session(store, session_id=session_id)
    versionless = await store.load_checkpoint(session_id)
    assert versionless is not None
    versionless.pop(CHECKPOINT_SCHEMA_VERSION_KEY)
    versionless["future_additive_field"] = {"kept": True}
    await store.checkpoint(session_id, versionless)
    pending = await _app(
        store,
        _CompleteProvider(),
    )._runtime_session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id),
    )
    assert pending.inspected_candidate_count == 1
    assert len(pending.actions) + len(pending.issues) == 1

    provider = _CompleteProvider()
    events = [
        event
        async for event in _app(store, provider).resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=input_id,
                answer="yes",
            )
        )
    ]

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert provider.calls == 1
    checkpoint = await store.load_checkpoint(session_id)
    assert checkpoint is not None
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == 1
    assert checkpoint["future_additive_field"] == {"kept": True}
    assert "pending_user_input" not in checkpoint


async def assert_versionless_noop_transform_stamps_conformance(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    app = _app(store, _CompleteProvider())
    _ = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="checkpoint-agent",
                session_id=session_id,
                messages=[Message.text("user", "complete")],
            )
        )
    ]
    await store.checkpoint(session_id, {"preserved": {"value": True}})

    await app._runtime_session_store.transform_checkpoint(
        session_id,
        lambda _session, _checkpoint: None,
    )

    checkpoint = await store.load_checkpoint(session_id)
    assert checkpoint == {
        CHECKPOINT_SCHEMA_VERSION_KEY: 1,
        "preserved": {"value": True},
    }


async def assert_future_checkpoint_rejection_conformance(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    input_id = await _pause_session(store, session_id=session_id)
    future = await store.load_checkpoint(session_id)
    assert future is not None
    future[CHECKPOINT_SCHEMA_VERSION_KEY] = 2
    await store.checkpoint(session_id, future)

    provider = _CompleteProvider()
    with pytest.raises(CheckpointCompatibilityError) as pending_error:
        await _app(
            store,
            provider,
        )._runtime_session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id),
        )
    assert pending_error.value.reason == "checkpoint_schema_version_too_new"
    with pytest.raises(CheckpointCompatibilityError) as caught:
        _ = [
            event
            async for event in _app(store, provider).resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="yes",
                )
            )
        ]

    session = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    assert caught.value.reason == "checkpoint_schema_version_too_new"
    assert provider.calls == 0
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert checkpoint is not None
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == 2
    assert "pending_user_input" in checkpoint

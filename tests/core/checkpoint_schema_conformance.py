from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cayu.core import AgentSpec, EventType, Message
from cayu.core.messages import ProviderStatePart, ToolCallPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    PendingActionQuery,
    RunRequest,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    SessionIdentity,
    SessionStatus,
    UserInputResponse,
    runtime_publication_checkpoint_value_digest,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.runtime.execution_units import new_model_step_identity
from cayu.runtime.sessions import SessionStore
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor


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


def _app(
    store: SessionStore,
    provider: ModelProvider,
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="checkpoint-agent", model="checkpoint-model"),
        tools=[UserInputTool()],
    )
    return app


async def assert_current_checkpoint_publication_upgrade_conformance(
    store: SessionStore,
    *,
    session_id_prefix: str,
) -> None:
    """Current runtime publications atomically upcast supported legacy roots."""

    current_store = runtime_checkpoint_session_store(store)
    legacy_sources = (
        ("absent", None, None),
        ("versionless", {"publication_phase": "before"}, "before"),
        (
            "v1",
            {
                CHECKPOINT_SCHEMA_VERSION_KEY: 1,
                "publication_phase": "before",
            },
            "before",
        ),
    )
    for suffix, source_checkpoint, source_phase in legacy_sources:
        session_id = f"{session_id_prefix}-{suffix}"
        await store.create(
            RunRequest(agent_name="checkpoint-agent", session_id=session_id, messages=[]),
            identity=SessionIdentity(
                provider_name="checkpoint-conformance",
                model="checkpoint-model",
            ),
        )
        if source_checkpoint is not None:
            await store.checkpoint(session_id, source_checkpoint)
        request = RuntimePublicationRequest(
            publication_id=f"checkpoint-current-writer-{suffix}",
            kind="approval-open",
            intent={"kind": "checkpoint-current-writer-conformance"},
            mutation=RuntimePublicationMutation(
                operations=(
                    RuntimePublicationCheckpointOperation(
                        key="publication_phase",
                        expected_value_digest=(
                            None
                            if source_phase is None
                            else runtime_publication_checkpoint_value_digest(source_phase)
                        ),
                        action="set",
                        value="published",
                    ),
                )
            ),
            transcript_messages=(),
            events=(),
        )

        result = await current_store.publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses={SessionStatus.PENDING},
            expected_run_epoch=0,
            expected_transcript_cursor=0,
        )

        assert result.replayed is False
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "publication_phase": "published",
        }


async def assert_assistant_publication_checkpoint_conformance(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    """Sealed assistant projections survive backend serialization and restart."""

    await store.create(
        RunRequest(agent_name="checkpoint-agent", session_id=session_id, messages=[]),
        identity=SessionIdentity(
            provider_name="checkpoint-conformance",
            model="checkpoint-model",
        ),
    )
    secret = "checkpoint-late-secret-canary"
    tool_call = runtime_records.ToolCallRequest(
        id="checkpoint-publication-call",
        name="checkpoint-tool",
        arguments={"provided": secret},
    )
    identity = new_model_step_identity().new_attempt().new_tool_round()
    checkpoint, _pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="checkpoint-agent",
        environment_name=None,
        task_id=None,
        tool_calls=[tool_call],
        policy_outcomes=None,
        assistant_message_state="quarantined",
        secret_resolution_scope="dynamic",
        quarantined_assistant_message=Message(
            role="assistant",
            content=(
                ProviderStatePart(provider="vendor", state={"opaque": "byte-stable"}),
                ToolCallPart(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    **identity.payload(),
                ),
            ),
        ),
        tool_round_identity=identity,
    )
    runtime_store = runtime_checkpoint_session_store(store)
    await runtime_store.checkpoint(session_id, checkpoint)
    await runtime_store.transform_checkpoint(
        session_id,
        tool_round_recovery.assistant_publication_snapshot_transform(
            tool_round_identity=identity,
            tool_call_id=tool_call.id,
            redactor=SecretRedactor(secret),
            unsafe_output=False,
        ),
    )

    restarted_store = runtime_checkpoint_session_store(store)
    recovered = tool_round_recovery.pending_tool_round_from_checkpoint(
        await restarted_store.load_checkpoint(session_id)
    )
    assert recovered is not None
    assert recovered.assistant_publication is not None
    assert recovered.assistant_publication.secret_resolution_scope == "dynamic"
    message = tool_round_recovery.ready_assistant_publication_message(recovered)
    assert secret not in repr(message)
    provider_state = next(part for part in message.content if type(part) is ProviderStatePart)
    assert provider_state.state == {"opaque": "byte-stable"}


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
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
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
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
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
    future[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
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
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    assert "pending_user_input" in checkpoint

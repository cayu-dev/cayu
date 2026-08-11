from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu import SQLiteSessionStore
from cayu.core import Event, EventType, Message
from cayu.core.events import event_with_runtime_payload_authority
from cayu.core.messages import ProviderStatePart, TextPart, ThinkingPart
from cayu.runtime import (
    EventQuery,
    InMemorySessionStore,
    ModelTarget,
    RunRequest,
    SessionIdentity,
    SessionModelTransition,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.runtime.sessions import (
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    SessionStore,
    session_input_messages_sha256,
)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _assert_model_switch_store_conformance(store: SessionStore, suffix: str) -> None:
    session_id = f"model-switch-store-{suffix}-{uuid4().hex}"
    source_transcript = [
        Message.text("user", "first"),
        Message(
            role="assistant",
            content=(
                ThinkingPart(
                    text="opaque reasoning",
                    provider_state={"signature": "source-signature"},
                ),
                TextPart(text="portable answer"),
                ProviderStatePart(
                    provider="source",
                    state={"type": "response_ref", "id": "source-response"},
                ),
            ),
        ),
    ]
    created = await store.create(
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="source", model="source-model"),
    )
    await store.append_transcript_messages(session_id, source_transcript)
    await store.checkpoint(session_id, {"preserved": True})
    await store.update_status(session_id, SessionStatus.COMPLETED)
    transcript_window = await store.load_transcript_window(
        session_id,
        start_index=1,
        limit=1,
    )
    assert transcript_window.cursor == 2
    assert [record.index for record in transcript_window.records] == [1]
    empty_window = await store.load_transcript_window(
        session_id,
        start_index=2,
        limit=1,
    )
    assert empty_window.cursor == 2
    assert empty_window.records == []

    target = ModelTarget(provider_name="target", model="target-model")
    stale_interaction_id = f"interaction-stale-{uuid4().hex}"
    stale_source = source_transcript
    stale_switch_event = event_with_runtime_payload_authority(
        Event(
            id=f"switch-stale-{uuid4().hex}",
            type=EventType.SESSION_MODEL_SWITCHED,
            session_id=session_id,
            interaction_id=stale_interaction_id,
            payload={
                "source_provider_name": "source",
                "source_model": "source-model",
                "target_provider_name": "target",
                "target_model": "target-model",
                "provider_changed": True,
                "model_changed": True,
                "provider_state_parts_dropped": 0,
                "thinking_parts_dropped": 0,
                "source_transcript_cursor": 1,
                "cache_state_dropped": True,
                "full_transcript_projection": True,
            },
        ),
        "source_provider_name",
        "source_model",
        "target_provider_name",
        "target_model",
    )
    with pytest.raises(SessionStatusConflict, match="transcript cursor changed"):
        await store.transition_status_and_checkpoint(
            session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda _session, checkpoint: {
                **({} if checkpoint is None else checkpoint),
                "must_not_commit": True,
            },
            interaction_started_event=Event(
                id=f"interaction-started-stale-{uuid4().hex}",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=stale_interaction_id,
            ),
            interaction_source_messages=[Message.text("user", "must not append")],
            model_transition=SessionModelTransition(
                target=target,
                event=stale_switch_event,
                source_transcript_digest=session_input_messages_sha256(stale_source),
                source_transcript_cursor=1,
            ),
        )
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.COMPLETED
    assert (unchanged.provider_name, unchanged.model) == ("source", "source-model")
    assert await store.load_checkpoint(session_id) == {"preserved": True}
    assert await store.load_transcript(session_id) == source_transcript
    assert await store.query_events(EventQuery(session_id=session_id)) == []

    interaction_id = f"interaction-{uuid4().hex}"
    switch_event = event_with_runtime_payload_authority(
        Event(
            id=f"switch-{uuid4().hex}",
            type=EventType.SESSION_MODEL_SWITCHED,
            session_id=session_id,
            interaction_id=interaction_id,
            payload={
                "source_provider_name": "source",
                "source_model": "source-model",
                "target_provider_name": "target",
                "target_model": "target-model",
                "provider_changed": True,
                "model_changed": True,
                "provider_state_parts_dropped": 1,
                "thinking_parts_dropped": 1,
                "source_transcript_cursor": 2,
                "cache_state_dropped": True,
                "full_transcript_projection": True,
            },
        ),
        "source_provider_name",
        "source_model",
        "target_provider_name",
        "target_model",
    )
    interaction_event = Event(
        id=f"interaction-started-{uuid4().hex}",
        type=EventType.INTERACTION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
    )
    appended_message = Message.text("user", "continue")
    transitioned = await store.transition_status_and_checkpoint(
        session_id,
        from_statuses={SessionStatus.COMPLETED},
        to_status=SessionStatus.RUNNING,
        checkpoint_transform=lambda _session, checkpoint: {
            **({} if checkpoint is None else checkpoint),
            "claimed": True,
        },
        interaction_started_event=interaction_event,
        interaction_source_messages=[appended_message],
        model_transition=SessionModelTransition(
            target=target,
            event=switch_event,
            source_transcript_digest=session_input_messages_sha256(source_transcript),
            source_transcript_cursor=2,
        ),
    )

    assert transitioned.status is SessionStatus.RUNNING
    assert transitioned.run_epoch == created.run_epoch + 1
    assert (transitioned.provider_name, transitioned.model) == ("target", "target-model")
    assert transitioned.metadata[MODEL_TARGET_PROJECTION_METADATA_KEY] == {
        "record_type": "cayu.model-target-projection",
        "schema_version": 1,
        "provider_name": "target",
        "model": "target-model",
        "transcript_cursor": 2,
    }
    assert await store.load_checkpoint(session_id) == {
        "preserved": True,
        "claimed": True,
    }
    assert await store.load_transcript(session_id) == [*source_transcript, appended_message]
    records = await store.query_events(EventQuery(session_id=session_id))
    assert [record.event.id for record in records[-2:]] == [
        switch_event.id,
        interaction_event.id,
    ]
    await store.release_run_fence(session_id)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_model_switch_transition_is_atomic_in_local_stores(store_kind: str, tmp_path) -> None:
    async def run() -> None:
        store: SessionStore
        if store_kind == "memory":
            store = InMemorySessionStore()
        else:
            store = SQLiteSessionStore(tmp_path / "model-switch.sqlite")
        try:
            await _assert_model_switch_store_conformance(store, store_kind)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_model_switch_transition_is_atomic_in_postgres(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_model_switch_store_conformance(store, "postgres")
        finally:
            await store.close()

    asyncio.run(run())

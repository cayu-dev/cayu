from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu import EXECUTION_PROFILE_METADATA_KEY, SQLiteSessionStore
from cayu.core import Event, EventType
from cayu.runtime import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    build_execution_profile_identity,
)


def _profile(*, tool_name: str) -> ExecutionProfileIdentity:
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="durable instructions",
        direct_tools=[
            {
                "name": tool_name,
                "description": "Record execution.",
                "schema": {"type": "object", "properties": {}},
                "parallel_safe": True,
                "effect": "external",
            }
        ],
    )


def _rejection_event(
    *,
    session_id: str,
    expected: ExecutionProfileIdentity,
    candidate: ExecutionProfileIdentity,
) -> Event:
    return Event(
        id=f"epr_{uuid4().hex}",
        type=EventType.SESSION_EXECUTION_PROFILE_REJECTED,
        session_id=session_id,
        agent_name="assistant",
        payload={
            "expected_profile_fingerprint": expected.fingerprint,
            "candidate_profile_fingerprint": candidate.fingerprint,
            "changed_component_classes": ["direct_tools"],
        },
    )


async def _assert_execution_profile_store_conformance(
    store: SessionStore,
    suffix: str,
) -> None:
    session_id = f"profile-store-{suffix}-{uuid4().hex}"
    expected = _profile(tool_name="original_tool")
    candidate = _profile(tool_name="replacement_tool")
    created = await store.create(
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            execution_profile=expected,
        ),
    )
    await store.update_status(session_id, SessionStatus.COMPLETED)
    completed = await store.load(session_id)
    assert completed is not None
    profile_record = completed.metadata[EXECUTION_PROFILE_METADATA_KEY]
    assert profile_record["baseline"] == expected.model_dump(mode="json")
    assert profile_record["expected"] == expected.model_dump(mode="json")
    event = _rejection_event(
        session_id=session_id,
        expected=expected,
        candidate=candidate,
    )

    first = await store.reject_execution_profile_resume(
        session_id,
        expected_statuses={SessionStatus.COMPLETED},
        expected_run_epoch=completed.run_epoch,
        expected_profile=expected,
        candidate_profile=candidate,
        event=event,
    )
    assert first.replayed is False
    assert first.event == event
    unchanged = await store.load(session_id)
    assert unchanged is not None
    assert unchanged.status is SessionStatus.COMPLETED
    assert unchanged.run_epoch == created.run_epoch

    replay = await store.reject_execution_profile_resume(
        session_id,
        expected_statuses={SessionStatus.COMPLETED},
        expected_run_epoch=completed.run_epoch,
        expected_profile=expected,
        candidate_profile=candidate,
        event=event.model_copy(update={"timestamp": event.timestamp.replace(microsecond=0)}),
    )
    assert replay.replayed is True
    assert [item.id for item in await store.load_events(session_id)].count(event.id) == 1

    with pytest.raises(SessionRunFenced):
        await store.reject_execution_profile_resume(
            session_id,
            expected_statuses={SessionStatus.COMPLETED},
            expected_run_epoch=completed.run_epoch + 1,
            expected_profile=expected,
            candidate_profile=candidate,
            event=_rejection_event(
                session_id=session_id,
                expected=expected,
                candidate=candidate,
            ),
        )
    assert len(await store.load_events(session_id)) == 1

    transform_called = False

    def unexpected_transform(_session, checkpoint):
        nonlocal transform_called
        transform_called = True
        return checkpoint

    with pytest.raises(SessionStatusConflict, match="execution profile changed"):
        await store.admit_execution_profile_resume(
            session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=unexpected_transform,
            execution_profile=candidate,
        )
    assert transform_called is False
    still_unchanged = await store.load(session_id)
    assert still_unchanged is not None
    assert still_unchanged.status is SessionStatus.COMPLETED
    assert still_unchanged.run_epoch == completed.run_epoch
    assert len(await store.load_events(session_id)) == 1

    admitted = await store.admit_execution_profile_resume(
        session_id,
        from_statuses={SessionStatus.COMPLETED},
        to_status=SessionStatus.RUNNING,
        checkpoint_transform=lambda _session, checkpoint: checkpoint,
        execution_profile=expected,
    )
    assert admitted.status is SessionStatus.RUNNING
    assert admitted.run_epoch == completed.run_epoch + 1
    await store.release_run_fence(session_id)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_execution_profile_rejection_is_atomic_in_local_stores(
    store_kind: str,
    tmp_path,
) -> None:
    async def run() -> None:
        store: SessionStore
        if store_kind == "memory":
            store = InMemorySessionStore()
        else:
            store = SQLiteSessionStore(tmp_path / "execution-profile.sqlite")
        try:
            await _assert_execution_profile_store_conformance(store, store_kind)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_execution_profile_rejection_is_atomic_in_postgres(postgres_dsn: str) -> None:
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
            await _assert_execution_profile_store_conformance(store, "postgres")
        finally:
            await store.close()

    asyncio.run(run())

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu import EXECUTION_PROFILE_METADATA_KEY, SQLiteSessionStore
from cayu.core import Event, EventType
from cayu.runtime import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    InMemorySessionStore,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    build_execution_profile_identity,
    execution_profile_decision_payload,
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


def _adoption_decision(
    *,
    session_id: str,
    expected: ExecutionProfileIdentity,
    candidate: ExecutionProfileIdentity,
) -> ExecutionProfileDecision:
    adoption_request_fingerprint = "a" * 64
    actor = ResolutionActor(
        subject="store-conformance",
        source=ResolutionActorSource.REQUEST,
    )
    payload = execution_profile_decision_payload(
        kind=ExecutionProfileDecisionKind.ADOPTED,
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=(ExecutionProfileComponentClass.DIRECT_TOOLS,),
        policy_identity="test:store-adoption:v1",
        policy_reason="Authorized by the conformance policy.",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        idempotency_identity=f"adoption-{session_id}",
        adoption_request_fingerprint=adoption_request_fingerprint,
        actor=actor,
        reason="Adopt the conformance candidate.",
    )
    return ExecutionProfileDecision(
        kind=ExecutionProfileDecisionKind.ADOPTED,
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=(ExecutionProfileComponentClass.DIRECT_TOOLS,),
        policy_identity="test:store-adoption:v1",
        policy_reason="Authorized by the conformance policy.",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        idempotency_identity=f"adoption-{session_id}",
        adoption_request_fingerprint=adoption_request_fingerprint,
        actor=actor,
        reason="Adopt the conformance candidate.",
        event=Event(
            id=f"epd_{uuid4().hex}",
            type=EventType.SESSION_EXECUTION_PROFILE_DECIDED,
            session_id=session_id,
            agent_name="assistant",
            payload=payload,
        ),
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

    rollback_session_id = f"profile-store-rollback-{suffix}-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=rollback_session_id, messages=[]),
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            execution_profile=expected,
        ),
    )
    await store.update_status(rollback_session_id, SessionStatus.COMPLETED)
    rollback_before = await store.load(rollback_session_id)
    assert rollback_before is not None
    rollback_decision = _adoption_decision(
        session_id=rollback_session_id,
        expected=expected,
        candidate=candidate,
    )

    def fail_after_profile_validation(_session, _checkpoint):
        raise RuntimeError("injected checkpoint failure")

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        await store.admit_execution_profile_resume(
            rollback_session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=fail_after_profile_validation,
            execution_profile=candidate,
            execution_profile_decision=rollback_decision,
            interaction_started_event=Event(
                type=EventType.INTERACTION_STARTED,
                session_id=rollback_session_id,
                interaction_id=f"interaction-{uuid4().hex}",
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    rollback_after = await store.load(rollback_session_id)
    assert rollback_after is not None
    assert rollback_after.status is rollback_before.status
    assert rollback_after.run_epoch == rollback_before.run_epoch
    assert rollback_after.metadata == rollback_before.metadata
    assert await store.load_events(rollback_session_id) == []

    source_less_actor = ResolutionActor(subject="store-conformance")
    source_less_decision = rollback_decision.model_copy(
        update={
            "actor": source_less_actor,
            "event": rollback_decision.event.model_copy(
                update={
                    "payload": {
                        **rollback_decision.event.payload,
                        "actor": {
                            "subject": source_less_actor.subject,
                            "tenant": None,
                            "source": None,
                        },
                    }
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="actor provenance source"):
        await store.admit_execution_profile_resume(
            rollback_session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda _session, checkpoint: checkpoint,
            execution_profile=candidate,
            execution_profile_decision=source_less_decision,
            interaction_started_event=Event(
                type=EventType.INTERACTION_STARTED,
                session_id=rollback_session_id,
                interaction_id=f"interaction-{uuid4().hex}",
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    source_less_after = await store.load(rollback_session_id)
    assert source_less_after is not None
    assert source_less_after.status is rollback_before.status
    assert source_less_after.run_epoch == rollback_before.run_epoch
    assert source_less_after.metadata == rollback_before.metadata
    assert await store.load_events(rollback_session_id) == []

    adoption_session_id = f"profile-store-adoption-{suffix}-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=adoption_session_id, messages=[]),
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            execution_profile=expected,
        ),
    )
    await store.update_status(adoption_session_id, SessionStatus.COMPLETED)
    decision = _adoption_decision(
        session_id=adoption_session_id,
        expected=expected,
        candidate=candidate,
    )
    interaction = Event(
        type=EventType.INTERACTION_STARTED,
        session_id=adoption_session_id,
        interaction_id=f"interaction-{uuid4().hex}",
        agent_name="assistant",
    )
    adopted = await store.admit_execution_profile_resume(
        adoption_session_id,
        from_statuses={SessionStatus.COMPLETED},
        to_status=SessionStatus.RUNNING,
        checkpoint_transform=lambda _session, checkpoint: checkpoint,
        execution_profile=candidate,
        execution_profile_decision=decision,
        interaction_started_event=interaction,
        interaction_source_messages=[],
    )
    assert adopted.metadata[EXECUTION_PROFILE_METADATA_KEY]["baseline"] == expected.model_dump(
        mode="json"
    )
    assert adopted.metadata[EXECUTION_PROFILE_METADATA_KEY]["expected"] == candidate.model_dump(
        mode="json"
    )
    adoption_events = await store.load_events(adoption_session_id)
    assert [item.type for item in adoption_events] == [
        EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        EventType.INTERACTION_STARTED,
    ]
    await store.release_run_fence(adoption_session_id)

    concurrent_session_id = f"profile-store-concurrent-{suffix}-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=concurrent_session_id, messages=[]),
        identity=SessionIdentity(
            provider_name="fake",
            model="fake-model",
            execution_profile=expected,
        ),
    )
    await store.update_status(concurrent_session_id, SessionStatus.COMPLETED)
    first_decision = _adoption_decision(
        session_id=concurrent_session_id,
        expected=expected,
        candidate=candidate,
    )
    second_decision = first_decision.model_copy(
        update={
            "idempotency_identity": f"second-{concurrent_session_id}",
            "event": first_decision.event.model_copy(
                update={
                    "id": f"epd_{uuid4().hex}",
                    "payload": {
                        **first_decision.event.payload,
                        "idempotency_identity": f"second-{concurrent_session_id}",
                    },
                }
            ),
        }
    )

    async def admit(decision: ExecutionProfileDecision):
        return await store.admit_execution_profile_resume(
            concurrent_session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda _session, checkpoint: checkpoint,
            execution_profile=candidate,
            execution_profile_decision=decision,
            interaction_started_event=Event(
                type=EventType.INTERACTION_STARTED,
                session_id=concurrent_session_id,
                interaction_id=f"interaction-{decision.idempotency_identity}",
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )

    outcomes = await asyncio.gather(
        admit(first_decision),
        admit(second_decision),
        return_exceptions=True,
    )
    assert sum(isinstance(outcome, SessionStatusConflict) for outcome in outcomes) == 1
    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
    concurrent_events = await store.load_events(concurrent_session_id)
    assert (
        sum(
            event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED for event in concurrent_events
        )
        == 1
    )
    await store.release_run_fence(concurrent_session_id)


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

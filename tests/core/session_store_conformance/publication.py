"""Shared session-store publication contract cases."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
)
from tests.core.session_store_conformance.support import (
    _RUNTIME_PUBLICATION_PREFIX,
    _assistant_model_completion_publication,
    _close_store,
    _ConformancePartialOverlapCompactor,
    _corrupt_runtime_publication_material,
    _expire_compaction_claim_for_test,
    _identity,
    _open_store,
    _publication_tool_round_identity,
    _reopen_store,
    _replace_runtime_publication_record,
    _runtime_publication_key,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ResolutionActor,
)
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
)
from cayu.sessions._operation_records import (
    RuntimePublicationOperationRecordMutation,
    SessionOperationPublication,
    runtime_publication_operation_record_value_digest,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest,
    runtime_publication_event_reference,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectStatus,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import (
    CompactSessionRequest,
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_input import SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.workspaces.branch_lifecycle import SessionWorkspaceBranchStore

pytestmark = pytest.mark.postgres


def test_session_store_conformance_model_completion_accepts_root_schema_stamp(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_root_schema"
            logical_step_id = "model-step:root-schema"
            stage_id = f"{logical_step_id}:attempt-0"
            intent = {"logical_step": logical_step_id}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.checkpoint(
                session_id,
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: (CURRENT_CHECKPOINT_SCHEMA_VERSION),
                },
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            valid = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id="root-schema-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
            )
            publication = RuntimePublicationRequest(
                publication_id=valid.publication_id,
                kind=valid.kind,
                intent=valid.intent,
                mutation=RuntimePublicationMutation(
                    operations=(
                        *valid.mutation.operations,
                        RuntimePublicationCheckpointOperation(
                            key=CHECKPOINT_SCHEMA_VERSION_KEY,
                            expected_value_digest=(
                                runtime_publication_checkpoint_value_digest(
                                    CURRENT_CHECKPOINT_SCHEMA_VERSION,
                                )
                            ),
                            action="set",
                            value=CURRENT_CHECKPOINT_SCHEMA_VERSION,
                        ),
                    )
                ),
                transcript_messages=valid.transcript_messages,
                events=valid.events,
                referenced_events=valid.referenced_events,
            )

            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )

            assert completed.stage.state == "completed"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_atomically_mutates_operation_record(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_model_completion_operation_mutation"
        operation_key = "test:hosted-discovery-view"
        source_operation = {"schema_version": 1, "revision": 0, "grants": []}
        desired_operation = {
            "schema_version": 1,
            "revision": 1,
            "grants": [{"tool_name": "remember_knowledge"}],
        }
        logical_step_id = "model-step:operation-mutation"
        stage_id = f"{logical_step_id}:attempt-0"
        intent = {"logical_step": logical_step_id}
        try:
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
                operation_initializer=lambda _session: {operation_key: source_operation},
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id="operation-mutation-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
                operation_record_mutations=(
                    RuntimePublicationOperationRecordMutation(
                        key=operation_key,
                        expected_value_digest=(
                            runtime_publication_operation_record_value_digest(source_operation)
                        ),
                        value=desired_operation,
                    ),
                ),
            )

            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.stage.publication == publication
            assert await store.load_session_operation(session_id, operation_key) == source_operation

            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert (
                await store.load_session_operation(session_id, operation_key) == desired_operation
            )
            assert await store.load_transcript(session_id) == [
                Message.text("assistant", "authoritative")
            ]
            assert [event.id for event in await store.load_events(session_id)] == [
                "operation-mutation-completed"
            ]

            store = await _reopen_store(session_store_case, store)
            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert replayed.replayed is True
            assert replayed.receipt == promoted.receipt
            assert (
                await store.load_session_operation(session_id, operation_key) == desired_operation
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_operation_record_cas_rejects_whole_publication(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_runtime_publication_stale_operation"
        operation_key = "test:stale-hosted-discovery-view"
        source_operation = {"schema_version": 1, "revision": 0}
        try:
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
                operation_initializer=lambda _session: {operation_key: source_operation},
            )
            request = RuntimePublicationRequest(
                publication_id="stale-operation-record",
                kind="model-step",
                intent={"logical_step": "stale-operation-record"},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=(Message.text("assistant", "must not publish"),),
                events=(
                    Event(
                        id="stale-operation-record-completed",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    ),
                ),
                operation_record_mutations=(
                    RuntimePublicationOperationRecordMutation(
                        key=operation_key,
                        expected_value_digest=(
                            runtime_publication_operation_record_value_digest(
                                {"schema_version": 1, "revision": 99}
                            )
                        ),
                        value={"schema_version": 1, "revision": 1},
                    ),
                ),
            )

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="operation record changed before publication",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.PENDING},
                    expected_run_epoch=0,
                    expected_transcript_cursor=0,
                )

            assert await store.load_session_operation(session_id, operation_key) == source_operation
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_first_commit_and_stale_replay(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_commit"
            publication_id = "model-step:1"
            fixed_at = datetime(2026, 7, 22, 9, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "publish atomically")],
                ),
                identity=_identity(),
            )
            await store.publish_checkpoint_and_events(
                session_id,
                checkpoint_transform=lambda _session, _checkpoint: {
                    "phase": "before",
                    "nested": {"owned": True},
                },
                events=[],
            )
            referenced_event = Event(
                id="model-requested",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                timestamp=fixed_at,
                interaction_id="interaction-publication",
            )
            await store.append_event(session_id, referenced_event)
            before = await store.load(session_id)
            assert before is not None

            intent_source = {"logical_step": {"number": 1}}
            structured_source = {"stable": "original"}
            event_payload_source = {"usage": {"total_tokens": 3}}
            transcript_message = Message.tool_result(
                tool_call_id="structured-output-1",
                tool_name="final_output",
                content="complete",
                structured=structured_source,
            )
            completed_event = Event(
                id="model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                timestamp=fixed_at + timedelta(seconds=1),
                interaction_id="interaction-publication",
                payload=event_payload_source,
            )
            source_checkpoint = {
                "phase": "before",
                "nested": {"owned": True},
            }
            published_checkpoint = {
                "phase": "published",
                "model_step": 1,
                "budget_reservation_id": "reservation-1",
            }
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-publication",
                intent=intent_source,
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    published_checkpoint,
                ),
                transcript_messages=[transcript_message],
                events=[completed_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            intent_source["logical_step"]["number"] = 99
            structured_source["stable"] = "caller-mutated"
            event_payload_source["usage"]["total_tokens"] = 99

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert published.session.updated_at > before.updated_at
            assert published.session.updated_at == published.session.last_activity_at
            assert published.receipt.published_at == published.session.updated_at
            assert published.receipt.session_id == session_id
            assert published.receipt.publication_id == publication_id
            assert published.receipt.kind == "model-step"
            assert published.receipt.interaction_id == "interaction-publication"
            assert published.receipt.intent == {"logical_step": {"number": 1}}
            assert published.receipt.source_status is SessionStatus.PENDING
            assert published.receipt.source_run_epoch == 0
            assert published.receipt.transcript_start_cursor == 0
            assert published.receipt.transcript_end_cursor == 1
            assert published.receipt.appended_event_ids == (completed_event.id,)
            assert published.receipt.referenced_events == (
                runtime_publication_event_reference(referenced_event),
            )
            assert len(published.receipt.request_digest) == 64
            assert len(published.receipt.publication_digest) == 64
            assert await store.load_checkpoint(session_id) == published_checkpoint
            loaded_session = await store.load(session_id)
            assert loaded_session is not None
            assert "must_not_persist" not in loaded_session.metadata
            loaded_transcript = await store.load_transcript(session_id)
            assert loaded_transcript == [transcript_message]
            assert loaded_transcript[0].content[0].structured == {"stable": "original"}
            loaded_events = await store.load_events(session_id)
            assert [event.id for event in loaded_events] == [
                referenced_event.id,
                completed_event.id,
            ]
            assert loaded_events[-1].payload == {"usage": {"total_tokens": 3}}
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id=completed_event.id,
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
                == published.receipt
            )

            request.intent["logical_step"]["number"] = 500
            request.transcript_messages[0].content[0].structured["stable"] = "changed"
            request.events[0].payload["usage"]["total_tokens"] = 500
            assert (await store.load_transcript(session_id))[0].content[0].structured == {
                "stable": "original"
            }
            assert (await store.load_events(session_id))[-1].payload == {
                "usage": {"total_tokens": 3}
            }
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
            ).intent == {"logical_step": {"number": 1}}

            equivalent_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-publication",
                intent={"logical_step": {"number": 1}},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    published_checkpoint,
                ),
                transcript_messages=[transcript_message],
                events=[completed_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            assert running.run_epoch == 1
            await store.release_run_fence(session_id)
            await store.update_status(session_id, SessionStatus.COMPLETED)
            later_message = Message.text("user", "state advanced after commit")
            await store.append_transcript_messages(session_id, [later_message])
            store = await _reopen_store(session_store_case, store)

            before_replay_session = await store.load(session_id)
            before_replay_checkpoint = await store.load_checkpoint(session_id)
            before_replay_transcript = await store.load_transcript(session_id)
            before_replay_events = await store.load_events(session_id)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=equivalent_request,
                expected_statuses={SessionStatus.COMPLETED},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=2,
            )
            assert replayed.replayed is True
            assert replayed.session == before_replay_session
            assert replayed.receipt == published.receipt
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events

            conflicting_request = equivalent_request.model_copy(
                update={"intent": {"logical_step": {"number": 2}}}
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_request,
                    expected_statuses={SessionStatus.COMPLETED},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=2,
                )
            conflicting_attribution = equivalent_request.model_copy(
                update={
                    "interaction_id": "interaction-other",
                    "events": [
                        event.model_copy(
                            update={"interaction_id": "interaction-other"},
                            deep=True,
                        )
                        for event in equivalent_request.events
                    ],
                },
                deep=True,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_attribution,
                    expected_statuses={SessionStatus.COMPLETED},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=2,
                )
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_sanitizes_caller_authority(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent_session_id = f"publication-origin-parent-{session_store_case[0]}"
            child_session_id = f"publication-origin-child-{session_store_case[0]}"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=parent_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )

            caller_authored_origin = Event(
                id="caller-authored-session-started",
                type=EventType.SESSION_STARTED,
                session_id=child_session_id,
                payload={
                    "parent_session_id": parent_session_id,
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                        "v1:0:1:original:text:sha256:" + "0" * 64
                    ),
                },
            )
            request_template = RuntimePublicationRequest(
                publication_id="caller-authored-origin-publication",
                kind="model-step",
                intent={"purpose": "origin-lineage-conformance"},
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(),
            )
            # Bypass the request validator deliberately. publish_runtime_publication()
            # must reconstruct and sanitize the request before digesting or storing it.
            bypassed_request = request_template.model_copy(
                update={"events": (caller_authored_origin,)},
            )
            assert bypassed_request.events[0].payload["parent_session_id"] == parent_session_id

            published = await store.publish_runtime_publication(
                child_session_id,
                request=bypassed_request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert published.receipt.appended_event_ids == (caller_authored_origin.id,)

            store = await _reopen_store(session_store_case, store)
            records = await store.query_events(
                EventQuery(
                    session_id=child_session_id,
                    event_type=EventType.SESSION_STARTED,
                    limit=2,
                )
            )
            assert len(records) == 1
            assert "parent_session_id" not in records[0].event.payload
            assert SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in records[0].event.payload

            equivalent_request = RuntimePublicationRequest(
                publication_id="caller-authored-origin-publication",
                kind="model-step",
                intent={"purpose": "origin-lineage-conformance"},
                mutation=RuntimePublicationMutation(),
                transcript_messages=(),
                events=(caller_authored_origin,),
            )
            assert "parent_session_id" not in equivalent_request.events[0].payload
            assert (
                SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY
                not in equivalent_request.events[0].payload
            )
            replayed = await store.publish_runtime_publication(
                child_session_id,
                request=equivalent_request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reclaimed_partial_publication_has_one_prefix(
    session_store_case,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        store = await _open_store(session_store_case)
        try:
            compactor = _ConformancePartialOverlapCompactor()

            def configured_app(now: datetime) -> CayuApp:
                app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=compactor,
                        max_user_turns=1,
                    ),
                )
                return app

            first_app = configured_app(accepted_at)
            reclaimed_app = configured_app(accepted_at + timedelta(minutes=6))
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_partial_reclaim_conformance",
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                    app=first_app,
                ),
            )
            transcript = [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            request = CompactSessionRequest(
                session_id=created.id,
                idempotency_key="partial-reclaim",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
                requested_by=ResolutionActor(subject="operator-a"),
            )

            async def collect(app: CayuApp, requested_by: str) -> list[Event]:
                attempted = request.model_copy(
                    update={"requested_by": ResolutionActor(subject=requested_by)}
                )
                return [event async for event in app.compact_session(attempted)]

            first = asyncio.create_task(collect(first_app, "operator-a"))
            await asyncio.wait_for(compactor.started[0].wait(), timeout=10)
            await _expire_compaction_claim_for_test(store, created.id, request.idempotency_key)
            reclaimed = asyncio.create_task(collect(reclaimed_app, "operator-b"))
            await asyncio.wait_for(compactor.started[1].wait(), timeout=10)
            compactor.release[1].set()
            reclaimed_events = await reclaimed
            compactor.release[0].set()
            with pytest.raises(RuntimeError, match="superseded"):
                await first

            store = await _reopen_store(session_store_case, store)
            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
            assert checkpoint["context_compaction"]["summary"] == "partial-1"
            assert await store.load_transcript(created.id) == transcript
            assert (
                sum(
                    event.type == EventType.CONTEXT_COMPACTION_COMPLETED
                    for event in reclaimed_events
                )
                == 1
            )
            durable_events = [
                record.event
                for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            ]
            assert (
                sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_runtime_publication_request_rejects_id_only_event_references() -> None:
    with pytest.raises(ValueError, match="referenced_event_ids"):
        RuntimePublicationRequest.model_validate(
            {
                "publication_id": "id-only-reference",
                "kind": "tool-round",
                "intent": {},
                "transcript_messages": [],
                "events": [],
                "referenced_event_ids": ["unbound-event-id"],
            }
        )


@pytest.mark.parametrize("failure", ["missing", "wrong-digest", "wrong-content"])
def test_session_store_conformance_runtime_publication_rejects_unbound_event_references(
    session_store_case,
    failure: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_runtime_publication_reference_{failure}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            durable_event = Event(
                id="referenced-terminal-event",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-1", "result": "authoritative"},
            )
            if failure != "missing":
                await store.append_event(session_id, durable_event)

            reference_source = durable_event
            if failure == "missing":
                reference_source = durable_event.model_copy(
                    update={"id": "event-that-was-never-persisted"},
                    deep=True,
                )
            elif failure == "wrong-content":
                reference_source = durable_event.model_copy(
                    update={"payload": {"tool_call_id": "call-1", "result": "forged"}},
                    deep=True,
                )
            reference = runtime_publication_event_reference(reference_source)
            if failure == "wrong-digest":
                reference = reference.model_copy(update={"event_digest": "0" * 64})

            round_identity = _publication_tool_round_identity(f"reference-{failure}")
            round_id = round_identity["tool_round_id"]
            request = RuntimePublicationRequest(
                publication_id=f"tool-round:{round_id}",
                kind="tool-round",
                intent={
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                },
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="must not publish",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[reference],
            )

            expected_error = "not durable" if failure == "missing" else "does not match"
            with pytest.raises(ValueError, match=expected_error):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                )
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == (
                [] if failure == "missing" else [durable_event]
            )
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_reference_order_is_durable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_reference_order"
            publication_id = "model-step:ordered-references"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            first = Event(
                id="ordered-reference-first",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-1", "order": 1},
            )
            second = Event(
                id="ordered-reference-second",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                payload={"tool_call_id": "call-2", "order": 2},
            )
            await store.append_event(session_id, first)
            await store.append_event(session_id, second)
            references = (
                runtime_publication_event_reference(first),
                runtime_publication_event_reference(second),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"tool_call_ids": ["call-1", "call-2"]},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[],
                events=[],
                referenced_events=references,
            )

            published = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            assert published.replayed is False
            assert published.receipt.referenced_events == references

            store = await _reopen_store(session_store_case, store)
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt

            reordered_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=request.mutation,
                transcript_messages=request.transcript_messages,
                events=request.events,
                referenced_events=tuple(reversed(references)),
            )
            with pytest.raises(SessionRuntimePublicationConflict, match="different request"):
                await store.publish_runtime_publication(
                    session_id,
                    request=reordered_request,
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_recovers_commit_then_cancel(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_commit_then_cancel"
            round_identity = _publication_tool_round_identity("commit-then-cancel")
            round_id = round_identity["tool_round_id"]
            publication_id = f"tool-round:{round_id}"
            pending_round = {
                **round_identity,
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "lookup",
                        "arguments": {},
                    }
                ],
            }
            source_checkpoint = {"pending_tool_round": pending_round}
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover the ambiguous commit")],
                ),
                identity=_identity(),
            )
            await store.checkpoint(session_id, source_checkpoint)
            started_event = Event(
                id="commit-then-cancel-started",
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **round_identity,
                    "tool_call_id": "call-1",
                    "idempotency_key": "tool-call:call-1",
                },
            )
            await store.append_event(session_id, started_event)
            completed_result = {
                "content": "completed once",
                "structured": None,
                "artifacts": [],
                "is_error": False,
            }
            completed_event = Event(
                id="commit-then-cancel-event",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id=session_id,
                tool_name="lookup",
                payload={
                    **round_identity,
                    "tool_call_id": "call-1",
                    "idempotency_key": "tool-call:call-1",
                    "result": completed_result,
                },
            )
            await store.append_event(session_id, completed_event)
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="tool-round",
                intent={
                    "schema_version": 1,
                    "round_id": round_id,
                    **round_identity,
                    "tool_call_ids": ["call-1"],
                    "pending_round_digest": runtime_publication_checkpoint_value_digest(
                        pending_round
                    ),
                },
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {},
                ),
                transcript_messages=[
                    Message.tool_result(
                        tool_call_id="call-1",
                        tool_name="lookup",
                        content="completed once",
                        **round_identity,
                    )
                ],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(started_event),
                    runtime_publication_event_reference(completed_event),
                ],
            )

            original_atomic_publish = store._publish_runtime_publication_atomic
            publication_committed = asyncio.Event()
            hold_acknowledgement = asyncio.Event()

            async def commit_then_hold_acknowledgement(
                prepared: Any,
            ) -> Any:
                result = await original_atomic_publish(prepared)
                assert result.replayed is False
                publication_committed.set()
                await hold_acknowledgement.wait()
                return result

            with monkeypatch.context() as publication_patch:
                publication_patch.setattr(
                    store,
                    "_publish_runtime_publication_atomic",
                    commit_then_hold_acknowledgement,
                )
                publishing = asyncio.create_task(
                    store.publish_runtime_publication(
                        session_id,
                        request=request,
                        expected_statuses={SessionStatus.PENDING},
                        expected_run_epoch=0,
                        expected_transcript_cursor=0,
                    )
                )
                await asyncio.wait_for(publication_committed.wait(), timeout=5)
                publishing.cancel("runtime publication acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await publishing
                assert cancellation.value.args == ("runtime publication acknowledgement lost",)

            committed_receipt = await store.load_runtime_publication_receipt(
                session_id,
                publication_id,
            )
            assert committed_receipt is not None
            assert await store.load_checkpoint(session_id) == {}
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [started_event, completed_event]
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id=completed_event.id,
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING

            store = await _reopen_store(session_store_case, store)
            before_replay_session = await store.load(session_id)
            before_replay_checkpoint = await store.load_checkpoint(session_id)
            before_replay_transcript = await store.load_transcript(session_id)
            before_replay_events = await store.load_events(session_id)

            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert replayed.replayed is True
            assert replayed.receipt == committed_receipt
            assert replayed.session == before_replay_session
            assert await store.load(session_id) == before_replay_session
            assert await store.load_checkpoint(session_id) == before_replay_checkpoint
            assert await store.load_transcript(session_id) == before_replay_transcript
            assert await store.load_events(session_id) == before_replay_events
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "transcript",
        "transcript-attribution",
        "event",
        "reference",
        "reference-content",
    ],
)
def test_session_store_conformance_runtime_publication_replay_fails_closed_on_material_drift(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_runtime_publication_material_{corruption}"
            publication_id = f"model-step:material-{corruption}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            referenced_event = Event(
                id=f"material-reference-{corruption}",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id="interaction-material",
            )
            await store.append_event(session_id, referenced_event)
            appended_event = Event(
                id=f"material-completion-{corruption}",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                interaction_id="interaction-material",
                payload={"usage": {"input_tokens": 2, "output_tokens": 1}},
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                interaction_id="interaction-material",
                intent={"logical_step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "authoritative material")],
                events=[appended_event],
                referenced_events=[runtime_publication_event_reference(referenced_event)],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=0,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False

            corrupted_event_id = (
                referenced_event.id
                if corruption in {"reference", "reference-content"}
                else appended_event.id
            )
            await _corrupt_runtime_publication_material(
                session_store_case,
                store,
                session_id=session_id,
                corruption=corruption,
                event_id=corrupted_event_id,
            )
            expected_error = {
                "transcript": "transcript segment conflicts",
                "transcript-attribution": "transcript attribution conflicts",
                "event": "event batch conflicts",
                "reference": "references a missing event",
                "reference-content": "referenced event content conflicts",
            }[corruption]
            with pytest.raises(SessionRuntimePublicationConflict, match=expected_error):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            with pytest.raises(SessionRuntimePublicationConflict, match=expected_error):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.PENDING},
                    expected_run_epoch=0,
                    expected_transcript_cursor=0,
                )
            assert await store.load_checkpoint(session_id) == {"phase": "published"}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_rejects_before_mutation_and_recovers(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_rollback"
            fixed_at = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "roll back invalid publication")],
                ),
                identity=_identity(),
            )
            await store.publish_checkpoint_and_events(
                session_id,
                checkpoint_transform=lambda _session, _checkpoint: {"phase": "before"},
                events=[],
            )
            before_session = await store.load(session_id)
            before_checkpoint = await store.load_checkpoint(session_id)
            before_transcript = await store.load_transcript(session_id)
            before_events = await store.load_events(session_id)

            malformed_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=RuntimePublicationMutation.model_construct(
                    operations=(
                        {
                            "key": "non_finite",
                            "expected_value_digest": None,
                            "action": "set",
                            "value": float("nan"),
                        },
                    )
                ),
                transcript_messages=(Message.text("assistant", "must not persist"),),
                events=(
                    Event(
                        id="malformed-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    ),
                ),
                referenced_events=(),
            )

            with pytest.raises(ValueError, match="mutation is malformed"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    malformed_request.publication_id,
                )
                is None
            )

            malformed_operation_mutation = RuntimePublicationRequest.model_construct(
                publication_id="malformed-operation-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=(Message.text("assistant", "must not persist"),),
                events=(
                    Event(
                        id="malformed-operation-mutation-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    ),
                ),
                operation_record_mutations=(
                    RuntimePublicationOperationRecordMutation.model_construct(
                        key="test:malformed-operation-mutation",
                        expected_value_digest="not-a-digest",
                        value={"schema_version": 1},
                    ),
                ),
                referenced_events=(),
            )
            with pytest.raises(
                ValueError,
                match="operation record mutation is malformed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_operation_mutation,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events

            stale_request = RuntimePublicationRequest(
                publication_id="stale-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "different-source"},
                    {"phase": "must-not-publish"},
                ),
                transcript_messages=[Message.text("assistant", "must not persist")],
                events=[
                    Event(
                        id="stale-mutation-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    )
                ],
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=stale_request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events

            malformed_boundary_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-request",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "wrong"},
                ),
                transcript_messages=(Message.text("assistant", "invalid boundary"),),
                events=(
                    Event(
                        id="invalid-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id="different-session",
                    ),
                ),
                referenced_events=(),
            )
            with pytest.raises(ValueError, match="does not match target session"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_boundary_request,
                )
            assert await store.load(session_id) == before_session

            malformed_reference = runtime_publication_event_reference(
                Event(
                    id="malformed-reference",
                    type=EventType.MODEL_STARTED,
                    session_id=session_id,
                )
            ).model_copy(update={"event_digest": "not-a-digest"})
            malformed_reference_request = RuntimePublicationRequest.model_construct(
                publication_id="malformed-reference-request",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "wrong"},
                ),
                transcript_messages=(),
                events=(),
                referenced_events=(malformed_reference,),
            )
            with pytest.raises(ValueError, match="malformed reference"):
                await store.publish_runtime_publication(
                    session_id,
                    request=malformed_reference_request,
                )
            assert await store.load(session_id) == before_session

            recovered_request = RuntimePublicationRequest(
                publication_id="recovered-mutation",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    {"phase": "before"},
                    {"phase": "recovered"},
                ),
                transcript_messages=[Message.text("assistant", "recovered")],
                events=[
                    Event(
                        id="recovered-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at + timedelta(seconds=1),
                    )
                ],
            )
            recovered = await store.publish_runtime_publication(
                session_id,
                request=recovered_request,
                expected_transcript_cursor=0,
            )
            assert recovered.replayed is False
            assert await store.load_checkpoint(session_id) == {"phase": "recovered"}
            assert await store.load_transcript(session_id) == [
                Message.text("assistant", "recovered")
            ]
            assert [event.id for event in await store.load_events(session_id)] == [
                "recovered-event"
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_serializes_concurrent_commits(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_concurrent"
            fixed_at = datetime(2026, 7, 22, 11, 0, tzinfo=UTC)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "publish once concurrently")],
                ),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id="shared-model-step",
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"winner": "shared"},
                ),
                transcript_messages=[Message.text("assistant", "only once")],
                events=[
                    Event(
                        id="shared-model-completed",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=fixed_at,
                    )
                ],
            )
            results = await asyncio.gather(
                *[
                    store.publish_runtime_publication(
                        session_id,
                        request=request,
                        expected_statuses={SessionStatus.PENDING},
                        expected_run_epoch=0,
                        expected_transcript_cursor=0,
                    )
                    for _ in range(8)
                ]
            )
            assert sum(not result.replayed for result in results) == 1
            assert sum(result.replayed for result in results) == 7
            assert len({result.receipt.publication_digest for result in results}) == 1
            assert await store.load_transcript(session_id) == [
                Message.text("assistant", "only once")
            ]
            assert [event.id for event in await store.load_events(session_id)] == [
                "shared-model-completed"
            ]

            fenced_session_id = "sess_runtime_publication_cursor_fence"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=fenced_session_id,
                    messages=[Message.text("user", "serialize different identities")],
                ),
                identity=_identity(),
            )
            assert await store.load_checkpoint(session_id) == {"winner": "shared"}
            contender_results = await asyncio.gather(
                *[
                    store.publish_runtime_publication(
                        fenced_session_id,
                        request=RuntimePublicationRequest(
                            publication_id=f"contender:{contender}",
                            kind="model-step",
                            intent={"round": contender},
                            mutation=runtime_publication_checkpoint_mutation(
                                None,
                                {"winner": contender},
                            ),
                            transcript_messages=[Message.text("assistant", contender)],
                            events=[
                                Event(
                                    id=f"event-{contender}",
                                    type=EventType.MODEL_COMPLETED,
                                    session_id=fenced_session_id,
                                    timestamp=fixed_at + timedelta(seconds=1),
                                )
                            ],
                        ),
                        expected_transcript_cursor=0,
                    )
                    for contender in ("a", "b")
                ],
                return_exceptions=True,
            )
            successes = [
                result for result in contender_results if not isinstance(result, BaseException)
            ]
            failures = [result for result in contender_results if isinstance(result, BaseException)]
            assert len(successes) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], ValueError)
            assert "transcript cursor is stale" in str(failures[0])
            assert len(await store.load_transcript(fenced_session_id)) == 1
            assert len(await store.load_events(fenced_session_id)) == 1
            winning_contender = successes[0].receipt.intent["round"]
            assert await store.load_checkpoint(fenced_session_id) == {"winner": winning_contender}
            receipts = [
                await store.load_runtime_publication_receipt(
                    fenced_session_id,
                    f"contender:{contender}",
                )
                for contender in ("a", "b")
            ]
            assert sum(receipt is not None for receipt in receipts) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_references_and_namespace(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_namespace"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "enforce receipt namespace")],
                ),
                identity=_identity(),
            )
            missing_reference_request = RuntimePublicationRequest(
                publication_id="missing-reference",
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "not published")],
                events=[],
                referenced_events=[
                    runtime_publication_event_reference(
                        Event(
                            id="not-durable",
                            type=EventType.MODEL_STARTED,
                            session_id=session_id,
                        )
                    )
                ],
            )
            with pytest.raises(ValueError, match="not durable"):
                await store.publish_runtime_publication(
                    session_id,
                    request=missing_reference_request,
                )
            assert await store.load_transcript(session_id) == []
            assert await store.load_checkpoint(session_id) is None

            overlap_event = Event(
                id="overlap-event",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
            )
            with pytest.raises(ValueError, match="cannot overlap"):
                RuntimePublicationRequest(
                    publication_id="overlap-reference",
                    kind="model-step",
                    intent={"round": 1},
                    mutation=runtime_publication_checkpoint_mutation(
                        None,
                        {"phase": "published"},
                    ),
                    transcript_messages=[],
                    events=[overlap_event],
                    referenced_events=[runtime_publication_event_reference(overlap_event)],
                )
            assert await store.load_events(session_id) == []

            reserved_key = _RUNTIME_PUBLICATION_PREFIX + "caller-owned"
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.load_session_operation(session_id, reserved_key)
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key=reserved_key,
                    operation_transform=lambda _session, _checkpoint, _record: (
                        SessionOperationPublication(checkpoint={})
                    ),
                    events=[],
                )

            bypassed_publication = SessionOperationPublication.model_construct(
                checkpoint={"phase": "legacy"},
                operation_records={reserved_key: {"status": "completed"}},
            )
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key="ordinary-operation",
                    operation_transform=lambda _session, _checkpoint, _record: bypassed_publication,
                    events=[],
                )
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_session_operation(session_id, "ordinary-operation") is None

            durable_reference = Event(
                id="durable-reference",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
            )
            await store.append_event(session_id, durable_reference)
            duplicate_event_request = RuntimePublicationRequest(
                publication_id="duplicate-appended-event",
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[],
                events=[durable_reference],
            )
            with pytest.raises(ValueError, match="Event already exists"):
                await store.publish_runtime_publication(
                    session_id,
                    request=duplicate_event_request,
                )
            assert await store.load_checkpoint(session_id) is None

            publication_id = "referenced-model-step"
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"round": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "referenced")],
                events=[],
                referenced_events=[runtime_publication_event_reference(durable_reference)],
            )
            result = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert result.receipt.referenced_events == (
                runtime_publication_event_reference(durable_reference),
            )
            assert result.receipt.appended_event_ids == ()
            assert await store.load_checkpoint(session_id) == {"phase": "published"}
            with pytest.raises(ValueError, match="reserved runtime publication namespace"):
                await store.load_session_operation(
                    session_id,
                    _runtime_publication_key(publication_id),
                )
            assert (
                await store.load_runtime_publication_receipt(session_id, publication_id)
                == result.receipt
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_fails_closed_on_receipt_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_receipt_drift"
            publication_id = "receipt-drift"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "verify receipt")],
                ),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"step": 1},
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {"phase": "published"},
                ),
                transcript_messages=[Message.text("assistant", "published")],
                events=[
                    Event(
                        id="receipt-drift-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
            )
            before_session = await store.load(session_id)
            before_checkpoint = await store.load_checkpoint(session_id)
            before_transcript = await store.load_transcript(session_id)
            before_events = await store.load_events(session_id)
            drifted_record = published.receipt.model_dump(mode="json")
            drifted_record["record_type"] = "caller-owned-record"
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record=drifted_record,
            )

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record="nested-json-is-not-a-receipt",
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)

            drifted_record = published.receipt.model_dump(mode="json")
            drifted_record["checkpoint_digest"] = "0" * 64
            await _replace_runtime_publication_record(
                session_store_case,
                store,
                session_id=session_id,
                publication_id=publication_id,
                record=drifted_record,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.load_runtime_publication_receipt(session_id, publication_id)
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="malformed or conflicts",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                )
            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == before_checkpoint
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_preserves_unrelated_keys(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_unrelated"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            source_checkpoint = {
                "owned": {"version": 1},
                "unrelated": {"version": 1},
            }
            target_checkpoint = {
                "owned": {"version": 2},
                "unrelated": {"version": 1},
            }
            await store.checkpoint(session_id, source_checkpoint)
            request = RuntimePublicationRequest(
                publication_id="checkpoint-unrelated",
                kind="model-step",
                intent={"round": "checkpoint-unrelated"},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    target_checkpoint,
                ),
                transcript_messages=[Message.text("assistant", "publish owned state")],
                events=[
                    Event(
                        id="checkpoint-unrelated-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            assert tuple(operation.key for operation in request.mutation.operations) == ("owned",)

            await store.transform_checkpoint(
                session_id,
                lambda _session, checkpoint: {
                    **(checkpoint or {}),
                    "unrelated": {"version": 2},
                    "late_runtime_state": {"preserved": True},
                },
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {
                "owned": {"version": 2},
                "unrelated": {"version": 2},
                "late_runtime_state": {"preserved": True},
            }
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == list(request.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_rejects_touched_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_touched_drift"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            source_checkpoint = {
                "owned": {"version": 1},
                "unrelated": {"stable": True},
            }
            await store.checkpoint(session_id, source_checkpoint)
            request = RuntimePublicationRequest(
                publication_id="checkpoint-touched-drift",
                kind="model-step",
                intent={"round": "checkpoint-touched-drift"},
                mutation=runtime_publication_checkpoint_mutation(
                    source_checkpoint,
                    {
                        "owned": {"version": 2},
                        "unrelated": {"stable": True},
                    },
                ),
                transcript_messages=[Message.text("assistant", "must not publish")],
                events=[
                    Event(
                        id="checkpoint-touched-drift-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            drifted_checkpoint = {
                "owned": {"version": 99},
                "unrelated": {"stable": True},
                "late_runtime_state": True,
            }
            await store.checkpoint(session_id, drifted_checkpoint)
            before_session = await store.load(session_id)

            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=request,
                    expected_transcript_cursor=0,
                )

            assert await store.load(session_id) == before_session
            assert await store.load_checkpoint(session_id) == drifted_checkpoint
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert (
                await store.load_runtime_publication_receipt(
                    session_id,
                    request.publication_id,
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_patch_distinguishes_null(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_null"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await store.checkpoint(session_id, {"nullable": None})

            absent_mutation = runtime_publication_checkpoint_mutation(
                {},
                {"nullable": "must-not-publish"},
            )
            assert absent_mutation.operations[0].expected_value_digest is None
            absent_request = RuntimePublicationRequest(
                publication_id="checkpoint-null-expected-absent",
                kind="model-step",
                intent={"round": "null-expected-absent"},
                mutation=absent_mutation,
                transcript_messages=[Message.text("assistant", "must not publish")],
                events=[
                    Event(
                        id="checkpoint-null-expected-absent-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="checkpoint key changed",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=absent_request,
                    expected_transcript_cursor=0,
                )
            assert await store.load_checkpoint(session_id) == {"nullable": None}
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            present_null_mutation = runtime_publication_checkpoint_mutation(
                {"nullable": None},
                {"nullable": "updated"},
            )
            assert present_null_mutation.operations[0].expected_value_digest is not None
            present_null_request = RuntimePublicationRequest(
                publication_id="checkpoint-null-expected-present",
                kind="model-step",
                intent={"round": "null-expected-present"},
                mutation=present_null_mutation,
                transcript_messages=[Message.text("assistant", "published")],
                events=[
                    Event(
                        id="checkpoint-null-expected-present-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=present_null_request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {"nullable": "updated"}
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_checkpoint_operations_normalize_and_bind(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_order"
            publication_id = "checkpoint-operation-order"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            first = RuntimePublicationCheckpointOperation(
                key="first",
                expected_value_digest=None,
                action="set",
                value={"version": 1},
            )
            second = RuntimePublicationCheckpointOperation(
                key="second",
                expected_value_digest=None,
                action="set",
                value={"version": 2},
            )
            reverse_order = RuntimePublicationMutation(operations=(second, first))
            forward_order = RuntimePublicationMutation(operations=(first, second))
            assert reverse_order == forward_order
            assert tuple(operation.key for operation in reverse_order.operations) == (
                "first",
                "second",
            )
            event = Event(
                id="checkpoint-operation-order-event",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"round": "checkpoint-operation-order"},
                mutation=reverse_order,
                transcript_messages=[Message.text("assistant", "ordered mutation")],
                events=[event],
            )
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) == {
                "first": {"version": 1},
                "second": {"version": 2},
            }

            equivalent_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=forward_order,
                transcript_messages=request.transcript_messages,
                events=request.events,
            )
            replayed = await store.publish_runtime_publication(
                session_id,
                request=equivalent_request,
                expected_transcript_cursor=99,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt

            conflicting_request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind=request.kind,
                intent=request.intent,
                mutation=runtime_publication_checkpoint_mutation(
                    None,
                    {
                        "first": {"version": 1},
                        "second": {"version": 3},
                    },
                ),
                transcript_messages=request.transcript_messages,
                events=request.events,
            )
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different request",
            ):
                await store.publish_runtime_publication(
                    session_id,
                    request=conflicting_request,
                )
            assert await store.load_checkpoint(session_id) == {
                "first": {"version": 1},
                "second": {"version": 2},
            }
            assert await store.load_transcript(session_id) == list(request.transcript_messages)
            assert await store.load_events(session_id) == [event]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_runtime_publication_empty_checkpoint_mutation_preserves_none(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_runtime_publication_checkpoint_empty"
            publication_id = "checkpoint-empty-mutation"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            request = RuntimePublicationRequest(
                publication_id=publication_id,
                kind="model-step",
                intent={"logical_step": "checkpoint-empty-mutation"},
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[Message.text("assistant", "no checkpoint state")],
                events=[
                    Event(
                        id="checkpoint-empty-mutation-event",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            assert request.mutation.operations == ()
            published = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=0,
            )
            assert published.replayed is False
            assert await store.load_checkpoint(session_id) is None

            store = await _reopen_store(session_store_case, store)
            assert await store.load_checkpoint(session_id) is None
            replayed = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_transcript_cursor=99,
            )
            assert replayed.replayed is True
            assert replayed.receipt == published.receipt
            assert await store.load_checkpoint(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_operation_commit_guard_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_operation_commit_guard_conformance",
                    messages=[],
                ),
                identity=_identity(),
            )
            event = Event(
                type=EventType.CONTEXT_COMPACTION_STARTED,
                session_id=created.id,
                agent_name="assistant",
                payload={"operation_id": "guarded-operation"},
            )

            def transform(_session, checkpoint, _persisted_record):
                updated = {} if checkpoint is None else dict(checkpoint)
                updated["guarded_operation"] = True
                return SessionOperationPublication(
                    checkpoint=updated,
                    operation_records={
                        "guarded-request": {
                            "operation_id": "guarded-operation",
                            "status": "completed",
                        }
                    },
                )

            def reject_commit() -> None:
                raise RuntimeError("operation commit guard rejected publication")

            with pytest.raises(RuntimeError, match="commit guard rejected"):
                await store.publish_session_operation_guarded(
                    created.id,
                    idempotency_key="guarded-request",
                    operation_transform=transform,
                    commit_guard=reject_commit,
                    events=[event],
                )

            assert await store.load_checkpoint(created.id) is None
            assert await store.load_session_operation(created.id, "guarded-request") is None
            assert await store.load_events(created.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_workspace_branch_store_guard_does_not_block_event_loop(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        release_guard = threading.Event()
        guard_started = asyncio.Event()
        loop = asyncio.get_running_loop()
        publication: asyncio.Task | None = None
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_operation_commit_guard_off_loop",
                    messages=[],
                ),
                identity=_identity(),
            )
            branch_store = SessionWorkspaceBranchStore(store)

            def transform(persisted_record):
                assert persisted_record is None
                return {"status": "completed"}

            def blocking_guard() -> None:
                loop.call_soon_threadsafe(guard_started.set)
                if not release_guard.wait(timeout=5):
                    raise TimeoutError("test did not release commit guard")

            publication = asyncio.create_task(
                branch_store.publish_workspace_branch_record(
                    created.id,
                    "guarded-request",
                    record_transform=transform,
                    commit_guard=blocking_guard,
                    expected_run_epoch=0,
                )
            )
            await asyncio.wait_for(guard_started.wait(), timeout=10)
            await asyncio.sleep(0)
            assert not publication.done()
            release_guard.set()
            await publication
            publication = None
        finally:
            release_guard.set()
            if publication is not None:
                await asyncio.gather(publication, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_future_checkpoint_before_operation_load(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_future_operation_load",
                    messages=[],
                ),
                identity=_identity(),
            )

            def transform(_session, checkpoint, _persisted_record):
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else dict(checkpoint),
                    operation_records={
                        "completed-request": {
                            "status": "completed",
                            "private_detail": "must-not-be-interpreted",
                        }
                    },
                )

            await store.publish_session_operation(
                created.id,
                idempotency_key="completed-request",
                operation_transform=transform,
                events=[],
            )
            with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
                await store.checkpoint(
                    created.id,
                    {
                        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                        "private_checkpoint_detail": "must-not-be-reported",
                    },
                )

            runtime_store = CayuApp(
                session_store=store,
                enable_logging=False,
            )._runtime_session_store
            with pytest.raises(CheckpointCompatibilityError) as caught:
                await runtime_store.load_session_operation(
                    created.id,
                    "completed-request",
                )

            assert caught.value.reason == "checkpoint_schema_version_too_new"
            assert "must-not-be-interpreted" not in str(caught.value)
            assert "must-not-be-reported" not in str(caught.value)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_guarded_operation_publication_requires_native_commit_boundary() -> None:
    class LegacyOverrideStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.publication_calls = 0

        async def publish_session_operation(
            self,
            session_id: str,
            *,
            idempotency_key: str,
            operation_transform,
            events: list[Event],
            expected_statuses: set[SessionStatus] | None = None,
            expected_run_epoch: int | None = None,
            expected_transcript_cursor: int | None = None,
        ) -> Session:
            self.publication_calls += 1
            return await super().publish_session_operation(
                session_id,
                idempotency_key=idempotency_key,
                operation_transform=operation_transform,
                events=events,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )

    async def run() -> None:
        store = LegacyOverrideStore()
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_guarded_legacy_override",
                messages=[],
            ),
            identity=_identity(),
        )
        guard_calls = 0

        def transform(_session, checkpoint, _persisted_record):
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["legacy_guarded_operation"] = True
            return SessionOperationPublication(checkpoint=updated)

        def commit_guard() -> None:
            nonlocal guard_calls
            guard_calls += 1

        await store.publish_session_operation_guarded(
            created.id,
            idempotency_key="guarded-request",
            operation_transform=transform,
            commit_guard=commit_guard,
            events=[],
        )

        assert store.publication_calls == 0
        assert guard_calls == 1
        assert await store.load_checkpoint(created.id) == {"legacy_guarded_operation": True}

        with pytest.raises(NotImplementedError, match="atomic guarded operation publication"):
            await SessionStore.publish_session_operation_guarded(
                store,
                created.id,
                idempotency_key="unsupported-guarded-request",
                operation_transform=transform,
                commit_guard=commit_guard,
                events=[],
            )

    asyncio.run(run())

"""Shared session-store model stages contract cases."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.core.session_store_conformance.support import (
    _MODEL_COMPLETION_ACTIVE_KEY,
    _assistant_model_completion_publication,
    _close_store,
    _delete_raw_session_operation_record,
    _identity,
    _load_raw_session_operation_record,
    _model_completion_abandonment_key,
    _model_completion_publication_checkpoint,
    _model_completion_stage_key,
    _model_completion_winner_key,
    _open_store,
    _reopen_store,
    _replace_model_completion_stage_record,
    _runtime_publication_key,
    _set_raw_session_operation_record,
)

import cayu.sessions.base as sessions_module
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.runtime.authority import SessionRunFenced
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions._checkpoint_mutations import (
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelCompletionStageDisposition,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES,
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
    SessionModelCompletionStageIncomplete,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectStatus,
)
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    RunRequest,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_model_completion_stage_reopens_and_promotes_exactly_once(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_lifecycle"
            stage_id = "model-step:run-1:cursor-0"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "stage the provider completion")],
                ),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            stage_request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1, "request_fingerprint": "request-1"},
                reservation_ids=["reservation-app", "reservation-session"],
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=stage_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.replayed is False
            assert prepared.dispatch_authorized is True
            assert prepared.stage.state == "in_flight"
            assert prepared.stage.source_status is SessionStatus.RUNNING
            assert prepared.stage.source_run_epoch == running.run_epoch
            assert prepared.stage.source_transcript_cursor == 0
            assert prepared.stage.reservation_ids == (
                "reservation-app",
                "reservation-session",
            )
            assert prepared.stage.publication is None
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert active.marker_digest
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            preparation_replay = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": 1, "request_fingerprint": "request-1"},
                    reservation_ids=["reservation-app", "reservation-session"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert preparation_replay.replayed is True
            assert preparation_replay.dispatch_authorized is False
            assert preparation_replay.stage == prepared.stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1, "request_fingerprint": "request-1"},
                completion_event_id="staged-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "durably staged answer"),
                event_payload={
                    "step": 1,
                    "usage": {"input_tokens": 4, "output_tokens": 3},
                },
                reservation_ids=("reservation-app", "reservation-session"),
            )
            before_completion_session = await store.load(session_id)
            assert before_completion_session is not None
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.replayed is False
            assert completed.dispatch_authorized is False
            assert completed.stage.state == "completed"
            assert completed.stage.publication == publication
            assert completed.stage.completed_at is not None
            assert completed.stage.completed_at >= completed.stage.prepared_at
            after_completion_session = await store.load(session_id)
            assert after_completion_session is not None
            assert after_completion_session.updated_at == completed.stage.completed_at
            assert after_completion_session.last_activity_at == completed.stage.completed_at
            assert (
                after_completion_session.last_activity_at
                > before_completion_session.last_activity_at
            )
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == completed.stage

            completion_replay = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=RuntimePublicationRequest(
                    publication_id=publication.publication_id,
                    kind=publication.kind,
                    intent=publication.intent,
                    mutation=publication.mutation,
                    transcript_messages=publication.transcript_messages,
                    events=publication.events,
                    referenced_events=publication.referenced_events,
                ),
            )
            assert completion_replay.replayed is True
            assert completion_replay.dispatch_authorized is False
            assert completion_replay.stage == completed.stage
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="must be published before another dispatch",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{stage_id}:retry-after-terminal",
                        logical_step_id=stage_id,
                        dispatch_ordinal=1,
                        intent={"logical_step": 1, "request_fingerprint": "request-2"},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            store = await _reopen_store(session_store_case, store)
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage

            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert promoted.receipt.publication_id == stage_id
            assert promoted.receipt.appended_event_ids == ("staged-model-completed",)
            assert promoted.receipt.transcript_start_cursor == 0
            assert promoted.receipt.transcript_end_cursor == 1
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_active_model_completion_stage(session_id) is None
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id="staged-model-completed",
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.PENDING

            store = await _reopen_store(session_store_case, store)
            later_preparation = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:later-logical-step:attempt-0",
                    logical_step_id="model-step:later-logical-step",
                    dispatch_ordinal=0,
                    intent={"logical_step": 2, "request_fingerprint": "request-2"},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=1,
            )
            assert later_preparation.dispatch_authorized is True
            later_active = await store.load_active_model_completion_stage(session_id)
            assert later_active is not None
            assert later_active.stage == later_preparation.stage
            store = await _reopen_store(session_store_case, store)
            before_replay_session = await store.load(session_id)
            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch + 100,
            )
            assert replayed.replayed is True
            assert replayed.receipt == promoted.receipt
            assert replayed.session == before_replay_session
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage
            assert await store.load_active_model_completion_stage(session_id) == later_active
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_bounds_model_completion_recovery_context(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_recovery_context_bound"
            stage_id = "model-step:recovery-context-bound"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            recovery_context = {
                "schema_version": 1,
                "request_metadata": {"payload": "x" * 1024},
            }
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"recovery_context": recovery_context},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.stage.intent["recovery_context"] == recovery_context
            assert (await store.load_model_completion_stage(session_id, stage_id)) == (
                prepared.stage
            )

            with pytest.raises(ValidationError, match="exceeds the durable byte limit"):
                ModelCompletionStageRequest(
                    stage_id="model-step:oversized-recovery-context",
                    logical_step_id="model-step:oversized-recovery-context",
                    dispatch_ordinal=1,
                    intent={
                        "recovery_context": {
                            "payload": "x" * MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES
                        }
                    },
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_abandonment_rejects_wrong_digest_and_terminal(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_abandonment_refusal"
            stage_id = "model-step:abandonment-refusal:attempt-0"
            logical_step_id = "model-step:abandonment-refusal"
            intent = {"logical_step": "abandonment-refusal", "request_fingerprint": "request-1"}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=intent,
                    reservation_ids=("reservation-1",),
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="preparation digest is stale",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest="0" * 64,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
            with pytest.raises(SessionRunFenced, match="stage run epoch is stale"):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch + 1,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            assert await store.load_active_model_completion_stage(session_id) == active

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=intent,
                completion_event_id="abandonment-refusal-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "terminal evidence"),
                reservation_ids=("reservation-1",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="terminal model-completion stage cannot be abandoned",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == completed.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("publication_state", ["winner", "receipt"])
def test_session_store_conformance_model_completion_abandonment_rejects_publication_state(
    session_store_case,
    publication_state: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_abandonment_{publication_state}"
            stage_id = f"model-step:abandonment-{publication_state}:attempt-0"
            logical_step_id = f"model-step:abandonment-{publication_state}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": publication_state},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication_storage_key = (
                _model_completion_winner_key(logical_step_id)
                if publication_state == "winner"
                else _runtime_publication_key(logical_step_id)
            )
            await _set_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=publication_storage_key,
                record={"malformed_but_durable": publication_state},
            )

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="durable publication state cannot be abandoned",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == prepared.stage
            assert (
                await _load_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_abandonment_key(stage_id),
                )
                is None
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_abandonment_replays_and_reprepares_safely(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_abandonment_replay"
            stage_id = "model-step:abandonment-replay:attempt-0"
            logical_step_id = "model-step:abandonment-replay"
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": "abandonment-replay", "request_fingerprint": "exact"},
                reservation_ids=("reservation-1", "reservation-2"),
            )
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            first_active = await store.load_active_model_completion_stage(session_id)
            assert first_active is not None

            abandoned = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=first.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert abandoned.replayed is False
            assert abandoned.abandonment.session_id == session_id
            assert abandoned.abandonment.stage_id == stage_id
            assert abandoned.abandonment.logical_step_id == logical_step_id
            assert abandoned.abandonment.preparation_request_digest == (
                first.stage.preparation_request_digest
            )
            assert abandoned.abandonment.preparation_digest == first.stage.preparation_digest
            assert abandoned.abandonment.active_marker_digest == first_active.marker_digest
            assert abandoned.abandonment.source_run_epoch == running.run_epoch
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None
            tombstone = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_model_completion_abandonment_key(stage_id),
            )
            assert tombstone is not None
            assert tombstone["preparation_digest"] == first.stage.preparation_digest
            assert tombstone["record_digest"] == abandoned.abandonment.abandonment_digest

            store = await _reopen_store(session_store_case, store)
            before_replay = await store.load(session_id)
            replayed = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=first.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert replayed.replayed is True
            assert replayed.abandonment == abandoned.abandonment
            assert await store.load(session_id) == before_replay
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None

            conflicting_request = request.model_copy(
                update={"intent": {"logical_step": "abandonment-replay", "changed": True}},
                deep=True,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="only be reused for its exact request",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=conflicting_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) is None

            second = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert second.replayed is False
            assert second.dispatch_authorized is True
            assert second.stage.preparation_digest != first.stage.preparation_digest
            second_active = await store.load_active_model_completion_stage(session_id)
            assert second_active is not None
            assert second_active.stage == second.stage

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="preparation digest is stale",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=first.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == second.stage
            assert await store.load_active_model_completion_stage(session_id) == second_active
            unchanged_tombstone = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_model_completion_abandonment_key(stage_id),
            )
            assert unchanged_tombstone == tombstone

            second_abandonment = await store.abandon_model_completion_stage(
                session_id,
                stage_id=stage_id,
                preparation_digest=second.stage.preparation_digest,
                expected_run_epoch=running.run_epoch,
            )
            assert second_abandonment.replayed is False
            assert second_abandonment.abandonment.preparation_digest == (
                second.stage.preparation_digest
            )
            assert await store.load_model_completion_stage(session_id, stage_id) is None
            assert await store.load_active_model_completion_stage(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["missing-active", "malformed-active", "forged-winner"])
def test_session_store_conformance_model_completion_preparation_replay_fails_closed(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_prepare_replay_{corruption}"
            logical_step_id = f"model-step:prepare-replay-{corruption}"
            stage_id = f"{logical_step_id}:attempt-0"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id},
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert prepared.dispatch_authorized is True

            if corruption == "missing-active":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                )
            elif corruption == "malformed-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record={"malformed": True},
                )
            else:
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                    record={"forged": True},
                )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            durable_stage = await store.load_model_completion_stage(session_id, stage_id)
            assert durable_stage == prepared.stage
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
            assert await store.load_runtime_publication_receipt(session_id, logical_step_id) is None
            if corruption == "malformed-active":
                before_completion = await store.load(session_id)
                assert before_completion is not None
                terminal = await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=_assistant_model_completion_publication(
                        session_id=session_id,
                        stage_id=stage_id,
                        logical_step_id=logical_step_id,
                        intent=request.intent,
                        completion_event_id="malformed-active-terminal-evidence",
                        source_transcript_cursor=0,
                        assistant_message=Message.text(
                            "assistant",
                            "retain terminal evidence",
                        ),
                    ),
                )
                assert terminal.stage.state == "completed"
                after_completion = await store.load(session_id)
                assert after_completion is not None
                assert after_completion.updated_at == terminal.stage.completed_at
                assert after_completion.updated_at > before_completion.updated_at
                assert after_completion.last_activity_at == before_completion.last_activity_at
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "winner-active",
        "malformed-active",
        "malformed-winner",
        "missing-terminal",
        "missing-receipt",
        "missing-winner",
    ],
)
def test_session_store_conformance_model_completion_publication_state_fails_closed(
    session_store_case,
    corruption: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_publication_state_{corruption}"
            logical_step_id = f"model-step:publication-state-{corruption}"
            stage_id = f"{logical_step_id}:attempt-0"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            stage_request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id},
            )
            await store.prepare_model_completion_stage(
                session_id,
                request=stage_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            active_record = await _load_raw_session_operation_record(
                session_store_case,
                store,
                session_id=session_id,
                storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
            )
            assert active_record is not None
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=logical_step_id,
                intent=stage_request.intent,
                completion_event_id=f"publication-state-event-{corruption}",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative answer"),
            )
            await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False

            if corruption == "winner-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record=active_record,
                )
            elif corruption == "malformed-active":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_MODEL_COMPLETION_ACTIVE_KEY,
                    record={"malformed": True},
                )
            elif corruption == "malformed-winner":
                await _set_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                    record={"malformed": True},
                )
            elif corruption == "missing-terminal":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_stage_key(stage_id, terminal=True),
                )
            elif corruption == "missing-receipt":
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_runtime_publication_key(logical_step_id),
                )
            else:
                await _delete_raw_session_operation_record(
                    session_store_case,
                    store,
                    session_id=session_id,
                    storage_key=_model_completion_winner_key(logical_step_id),
                )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(SessionModelCompletionStageConflict):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=stage_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )

            expected_error = (
                (SessionModelCompletionStageConflict, SessionModelCompletionStageIncomplete)
                if corruption == "missing-terminal"
                else SessionModelCompletionStageConflict
            )
            with pytest.raises(expected_error):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch + 100,
                )
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_preserves_non_turn_completion(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_non_turn"
            stage_id = "model-step:contentless-attempt"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            intent = {"logical_step": 1, "attempt": 1}
            await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent=intent,
                    reservation_ids=["reservation-contentless-attempt"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent=intent,
                completion_event_id="contentless-model-completed",
                source_transcript_cursor=0,
                assistant_message=None,
                classification={
                    "type": "invalid",
                    "reason": "assistant produced no content",
                },
                event_payload={
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
                reservation_ids=("reservation-contentless-attempt",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.stage.state == "completed"

            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert promoted.receipt.transcript_start_cursor == 0
            assert promoted.receipt.transcript_end_cursor == 0
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == list(publication.events)

            store = await _reopen_store(session_store_case, store)
            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=running.run_epoch + 1,
            )
            assert replayed.replayed is True
            assert replayed.receipt == promoted.receipt
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("missing", "atomically publish exactly"),
        ("stage-id", "pointer conflicts"),
        ("source-boundary", "pointer conflicts"),
        ("completion-event", "pointer conflicts"),
        ("classification", "pointer conflicts"),
    ],
)
def test_session_store_conformance_model_completion_stage_rejects_invalid_publication_pointer(
    session_store_case,
    corruption: str,
    expected_error: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_model_completion_pointer_{corruption}"
            logical_step_id = f"model-step:pointer-{corruption}"
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
                completion_event_id=f"pointer-completed-{corruption}",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
            )
            if corruption == "missing":
                mutation = runtime_publication_checkpoint_mutation(None, None)
            else:
                (pointer_operation,) = valid.mutation.operations
                assert type(pointer_operation.value) is dict
                pointer = dict(pointer_operation.value)
                if corruption == "stage-id":
                    pointer["stage_id"] = f"{stage_id}:forged"
                elif corruption == "source-boundary":
                    pointer["source_transcript_cursor"] = 1
                    pointer["transcript_end_cursor"] = 2
                elif corruption == "completion-event":
                    pointer["completion_event_id"] = "forged-model-completed"
                else:
                    pointer["classification"] = {
                        "type": "final",
                        "reason": "forged",
                    }
                mutation = runtime_publication_checkpoint_mutation(
                    None,
                    {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer},
                )
            publication = RuntimePublicationRequest(
                publication_id=valid.publication_id,
                kind=valid.kind,
                intent=valid.intent,
                mutation=mutation,
                transcript_messages=valid.transcript_messages,
                events=valid.events,
                referenced_events=valid.referenced_events,
            )

            with pytest.raises(ValueError, match=expected_error):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )
            stage = await store.load_model_completion_stage(session_id, stage_id)
            assert stage is not None
            assert stage.state == "in_flight"
            assert await store.load_checkpoint(session_id) is None
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_preserves_purpose_validation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_compaction_purpose"
            stage_id = "context-compaction:attempt-0"
            logical_step_id = "context-compaction:logical-0"
            intent = {"operation_id": "compaction-0", "input_digest": "input-0"}
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
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
                    purpose="context-compaction",
                    intent=intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            event = Event(
                id="context-compaction-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={"purpose": "context_compaction", "usage": {"input_tokens": 2}},
            )
            with pytest.raises(ValueError, match="kind must be 'context-compaction'"):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=RuntimePublicationRequest(
                        publication_id=logical_step_id,
                        kind="model-step",
                        intent=intent,
                        mutation=runtime_publication_checkpoint_mutation(None, None),
                        transcript_messages=[],
                        events=[event],
                    ),
                )
            publication = RuntimePublicationRequest(
                publication_id=logical_step_id,
                kind="context-compaction",
                intent=intent,
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[],
                events=[event],
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert completed.stage.state == "completed"
            assert completed.stage.purpose == "context-compaction"
            assert completed.stage.publication == publication
            assert await store.load_transcript(session_id) == []
            assert await store.load_events(session_id) == []

            assistant_session_id = "sess_model_completion_stage_assistant_purpose"
            assistant_stage_id = "model-step:assistant-purpose:attempt-0"
            assistant_logical_step_id = "model-step:assistant-purpose"
            assistant_intent = {"logical_step": "assistant-purpose"}
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=assistant_session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            assistant_running = await store.transition_status(
                assistant_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                assistant_session_id,
                request=ModelCompletionStageRequest(
                    stage_id=assistant_stage_id,
                    logical_step_id=assistant_logical_step_id,
                    dispatch_ordinal=0,
                    purpose="assistant-turn",
                    intent=assistant_intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=assistant_running.run_epoch,
                expected_transcript_cursor=0,
            )
            with pytest.raises(
                ValueError,
                match="cannot carry context-compaction evidence",
            ):
                await store.complete_model_completion_stage(
                    assistant_session_id,
                    stage_id=assistant_stage_id,
                    publication=RuntimePublicationRequest(
                        publication_id=assistant_logical_step_id,
                        kind="model-step",
                        intent=assistant_intent,
                        mutation=runtime_publication_checkpoint_mutation(None, None),
                        transcript_messages=[Message.text("assistant", "not a compaction")],
                        events=[
                            Event(
                                id="assistant-purpose-model-completed",
                                type=EventType.MODEL_COMPLETED,
                                session_id=assistant_session_id,
                                payload={"purpose": "context_compaction"},
                            )
                        ],
                    ),
                )
            assistant_stage = await store.load_model_completion_stage(
                assistant_session_id,
                assistant_stage_id,
            )
            assert assistant_stage is not None
            assert assistant_stage.state == "in_flight"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_fences_conflicts_and_drift(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_conflicts"
            stage_id = "model-step:conflict"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1},
                reservation_ids=["reservation-1"],
            )

            with pytest.raises(SessionRunFenced, match="run epoch is stale"):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch + 1,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(ValueError, match="transcript cursor is stale"):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=1,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) is None

            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different logical model completion",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:unrelated",
                        logical_step_id="model-step:unrelated",
                        dispatch_ordinal=0,
                        intent={"logical_step": "unrelated"},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="advance dispatch_ordinal monotonically",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:non-monotonic-retry",
                        logical_step_id=stage_id,
                        dispatch_ordinal=0,
                        intent={"logical_step": 1, "retry": True},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different request",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=stage_id,
                        logical_step_id=stage_id,
                        dispatch_ordinal=0,
                        intent={"logical_step": 2},
                        reservation_ids=["reservation-1"],
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            with pytest.raises(SessionModelCompletionStageIncomplete, match="still in flight"):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch,
                )

            with pytest.raises(ValueError, match="requires a non-turn"):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=_assistant_model_completion_publication(
                        session_id=session_id,
                        stage_id=stage_id,
                        logical_step_id=stage_id,
                        intent={"logical_step": 1},
                        completion_event_id="unclassified-contentless-completion",
                        source_transcript_cursor=0,
                        assistant_message=None,
                        classification={"type": "invalid"},
                        reservation_ids=("reservation-1",),
                        include_classification=False,
                    ),
                )

            private_key = _model_completion_stage_key(stage_id, terminal=False)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.load_session_operation(session_id, private_key)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key=private_key,
                    operation_transform=lambda _session, _checkpoint, _record: (
                        SessionOperationPublication(checkpoint={})
                    ),
                    events=[],
                )

            wrong_intent = RuntimePublicationRequest(
                publication_id=stage_id,
                kind="model-step",
                intent={"logical_step": 2},
                mutation=runtime_publication_checkpoint_mutation(None, None),
                transcript_messages=[Message.text("assistant", "wrong")],
                events=[
                    Event(
                        id="wrong-intent-completion",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                    )
                ],
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="intent conflicts",
            ):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=wrong_intent,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == prepared.stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="authoritative-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "authoritative"),
                reservation_ids=("reservation-1",),
            )
            completed = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            conflicting_publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="contradictory-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "contradictory"),
                reservation_ids=("reservation-1",),
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different terminal material",
            ):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=conflicting_publication,
                )
            assert await store.load_model_completion_stage(session_id, stage_id) == completed.stage

            await _replace_model_completion_stage_record(
                session_store_case,
                store,
                session_id=session_id,
                stage_id=stage_id,
                terminal=True,
                record={"record_type": "corrupted-terminal"},
            )
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="terminal model completion is malformed",
            ):
                await store.load_model_completion_stage(session_id, stage_id)

            cursor_session_id = "sess_model_completion_stage_cursor_fence"
            await store.create(
                RunRequest(agent_name="assistant", session_id=cursor_session_id, messages=[]),
                identity=_identity(),
            )
            cursor_running = await store.transition_status(
                cursor_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                cursor_session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:cursor-fence",
                    logical_step_id="model-step:cursor-fence",
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=cursor_running.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.append_transcript_messages(
                cursor_session_id,
                [Message.text("user", "concurrent input")],
            )
            cursor_publication = _assistant_model_completion_publication(
                session_id=cursor_session_id,
                stage_id="model-step:cursor-fence",
                logical_step_id="model-step:cursor-fence",
                intent={"logical_step": 1},
                completion_event_id="cursor-fenced-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "stale"),
            )
            cursor_completion = await store.complete_model_completion_stage(
                cursor_session_id,
                stage_id="model-step:cursor-fence",
                publication=cursor_publication,
            )
            assert cursor_completion.stage.state == "completed"
            assert await store.load_events(cursor_session_id) == []
            with pytest.raises(ValueError, match="transcript cursor is stale"):
                await store.promote_model_completion_stage(
                    cursor_session_id,
                    stage_id="model-step:cursor-fence",
                    expected_run_epoch=cursor_running.run_epoch,
                )
            assert await store.load_checkpoint(cursor_session_id) is None

            epoch_session_id = "sess_model_completion_stage_epoch_fence"
            await store.create(
                RunRequest(agent_name="assistant", session_id=epoch_session_id, messages=[]),
                identity=_identity(),
            )
            first_epoch = await store.transition_status(
                epoch_session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.prepare_model_completion_stage(
                epoch_session_id,
                request=ModelCompletionStageRequest(
                    stage_id="model-step:epoch-fence",
                    logical_step_id="model-step:epoch-fence",
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=first_epoch.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.release_run_fence(epoch_session_id)
            await store.update_status(epoch_session_id, SessionStatus.INTERRUPTED)
            second_epoch = await store.transition_status(
                epoch_session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            assert second_epoch.run_epoch > first_epoch.run_epoch
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different logical model completion",
            ):
                await store.prepare_model_completion_stage(
                    epoch_session_id,
                    request=ModelCompletionStageRequest(
                        stage_id="model-step:epoch-fence:retry",
                        logical_step_id="model-step:epoch-fence",
                        dispatch_ordinal=1,
                        intent={"logical_step": 1, "attempt": 2},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=second_epoch.run_epoch,
                    expected_transcript_cursor=0,
                )
            active_after_takeover = await store.load_active_model_completion_stage(epoch_session_id)
            assert active_after_takeover is not None
            assert active_after_takeover.stage.stage_id == "model-step:epoch-fence"
            epoch_publication = _assistant_model_completion_publication(
                session_id=epoch_session_id,
                stage_id="model-step:epoch-fence",
                logical_step_id="model-step:epoch-fence",
                intent={"logical_step": 1},
                completion_event_id="epoch-fenced-completion",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "recovered completion"),
            )
            before_epoch_completion = await store.load(epoch_session_id)
            assert before_epoch_completion is not None
            epoch_completion = await store.complete_model_completion_stage(
                epoch_session_id,
                stage_id="model-step:epoch-fence",
                publication=epoch_publication,
            )
            assert epoch_completion.stage.state == "completed"
            assert epoch_completion.dispatch_authorized is False
            assert epoch_completion.stage.source_run_epoch == first_epoch.run_epoch
            after_epoch_completion = await store.load(epoch_session_id)
            assert after_epoch_completion is not None
            assert after_epoch_completion.updated_at == epoch_completion.stage.completed_at
            assert after_epoch_completion.updated_at > before_epoch_completion.updated_at
            assert (
                after_epoch_completion.last_activity_at == before_epoch_completion.last_activity_at
            )
            assert await store.load_transcript(epoch_session_id) == []
            assert await store.load_events(epoch_session_id) == []
            with pytest.raises(SessionRunFenced, match="run epoch is stale"):
                await store.promote_model_completion_stage(
                    epoch_session_id,
                    stage_id="model-step:epoch-fence",
                    expected_run_epoch=first_epoch.run_epoch,
                )
            assert await store.load_checkpoint(epoch_session_id) is None

            recovered_promotion = await store.promote_model_completion_stage(
                epoch_session_id,
                stage_id="model-step:epoch-fence",
                expected_run_epoch=second_epoch.run_epoch,
            )
            assert recovered_promotion.replayed is False
            assert recovered_promotion.receipt.source_run_epoch == second_epoch.run_epoch
            assert await store.load_checkpoint(
                epoch_session_id
            ) == _model_completion_publication_checkpoint(epoch_publication)
            assert await store.load_transcript(epoch_session_id) == list(
                epoch_publication.transcript_messages
            )
            assert await store.load_events(epoch_session_id) == list(epoch_publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_late_superseded_completion_liveness(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_superseded_liveness"
            logical_step_id = "model-step:superseded-liveness"
            first_stage_id = f"{logical_step_id}:attempt-0"
            retry_stage_id = f"{logical_step_id}:attempt-1"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first_request = ModelCompletionStageRequest(
                stage_id=first_stage_id,
                logical_step_id=logical_step_id,
                dispatch_ordinal=0,
                intent={"logical_step": logical_step_id, "attempt": 0},
            )
            first = await store.prepare_model_completion_stage(
                session_id,
                request=first_request,
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert first.dispatch_authorized is True
            retry = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=retry_stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=1,
                    intent={"logical_step": logical_step_id, "attempt": 1},
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert retry.dispatch_authorized is True
            before_completion = await store.load(session_id)
            assert before_completion is not None

            late_completion = await store.complete_model_completion_stage(
                session_id,
                stage_id=first_stage_id,
                publication=_assistant_model_completion_publication(
                    session_id=session_id,
                    stage_id=first_stage_id,
                    logical_step_id=logical_step_id,
                    intent=first_request.intent,
                    completion_event_id="superseded-liveness-model-completed",
                    source_transcript_cursor=0,
                    assistant_message=Message.text("assistant", "late first attempt"),
                ),
            )
            assert late_completion.replayed is False
            assert late_completion.dispatch_authorized is False
            after_completion = await store.load(session_id)
            assert after_completion is not None
            assert after_completion.updated_at == late_completion.stage.completed_at
            assert after_completion.updated_at > before_completion.updated_at
            assert after_completion.last_activity_at == before_completion.last_activity_at
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage == retry.stage

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="no longer the exact active dispatch",
            ):
                await store.prepare_model_completion_stage(
                    session_id,
                    request=first_request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert await store.load_active_model_completion_stage(session_id) == active
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_allows_trusted_live_retry(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_live_retry"
            first_stage_id = "model-step:logical-1:attempt-1"
            retry_stage_id = "model-step:logical-1:attempt-2"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )

            first = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=first_stage_id,
                    logical_step_id="model-step:logical-1",
                    dispatch_ordinal=0,
                    intent={"logical_step": "logical-1", "attempt": 1},
                    reservation_ids=["reservation-attempt-1"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert first.stage.state == "in_flight"
            first_active = await store.load_active_model_completion_stage(session_id)
            assert first_active is not None
            assert first_active.stage == first.stage

            # A runtime that directly observed a terminal provider error may start
            # a new attempt. The old identity remains dispatch-ambiguous for
            # process-loss recovery and is never reused for provider dispatch.
            retry = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=retry_stage_id,
                    logical_step_id="model-step:logical-1",
                    dispatch_ordinal=1,
                    intent={"logical_step": "logical-1", "attempt": 2},
                    reservation_ids=["reservation-attempt-2"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            assert retry.stage.state == "in_flight"
            retry_active = await store.load_active_model_completion_stage(session_id)
            assert retry_active is not None
            assert retry_active.stage == retry.stage
            assert retry_active.stage.dispatch_ordinal == 1
            assert (
                await store.load_model_completion_stage(session_id, first_stage_id) == first.stage
            )

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=retry_stage_id,
                logical_step_id="model-step:logical-1",
                intent={"logical_step": "logical-1", "attempt": 2},
                completion_event_id="live-retry-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "retry succeeded"),
                reservation_ids=("reservation-attempt-2",),
            )
            await store.complete_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                publication=publication,
            )
            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
            assert await store.load_active_model_completion_stage(session_id) is None
            durable_first = await store.load_model_completion_stage(
                session_id,
                first_stage_id,
            )
            assert durable_first is not None
            assert durable_first.state == "in_flight"

            late_first_completion = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=first_stage_id,
                logical_step_id="model-step:logical-1",
                intent={"logical_step": "logical-1", "attempt": 1},
                completion_event_id="late-first-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "late first response"),
                reservation_ids=("reservation-attempt-1",),
            )
            late_completion = await store.complete_model_completion_stage(
                session_id,
                stage_id=first_stage_id,
                publication=late_first_completion,
            )
            assert late_completion.stage.state == "completed"
            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="different model-completion winner",
            ):
                await store.promote_model_completion_stage(
                    session_id,
                    stage_id=first_stage_id,
                    expected_run_epoch=running.run_epoch,
                )
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_records_exact_model_stage_dispatch(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_exact_dispatch"
            interaction_id = "interaction-model-completion-exact-dispatch"
            logical_step_id = "mstep_33333333333333333333333333333333"
            stage_id = f"{logical_step_id}:dispatch:0"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            prepared = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": interaction_id,
                        "model_step_id": logical_step_id,
                        "model_attempt_id": "matt_33333333333333333333333333333333",
                        "request_fingerprint": "3" * 64,
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )

            dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(stage_id)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.load_session_operation(session_id, dispatch_key)
            with pytest.raises(ValueError, match="reserved model-completion stage namespace"):
                await store.publish_session_operation(
                    session_id,
                    idempotency_key="caller-owned-dispatch-collision",
                    operation_transform=lambda _session, checkpoint, _record: (
                        SessionOperationPublication(
                            checkpoint={} if checkpoint is None else checkpoint,
                            operation_records={dispatch_key: {"owner": "caller"}},
                        )
                    ),
                    events=[],
                )

            dispatch = await store.mark_model_completion_stage_dispatched(
                session_id,
                stage=prepared.stage,
            )
            assert dispatch.stage_id == stage_id
            assert dispatch.logical_step_id == logical_step_id
            assert dispatch.interaction_id == interaction_id
            assert dispatch.model_attempt_id == "matt_33333333333333333333333333333333"
            assert dispatch.request_fingerprint == "3" * 64
            assert dispatch.provider_effect_id == f"request:{'3' * 64}"
            assert dispatch.preparation_digest == prepared.stage.preparation_digest
            assert dispatch.source_run_epoch == running.run_epoch

            with pytest.raises(
                sessions_module.SessionModelCompletionDispatchAlreadyAuthorized,
                match="Provider dispatch was authorized",
            ):
                await store.transition_status_and_checkpoint(
                    session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=lambda _session, checkpoint: (
                        {} if checkpoint is None else checkpoint
                    ),
                    require_no_active_model_completion_dispatch=True,
                )
            after_conflict = await store.load(session_id)
            assert after_conflict is not None
            assert after_conflict.status is SessionStatus.RUNNING
            assert after_conflict.run_epoch == running.run_epoch

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="dispatched model-completion stage cannot be abandoned",
            ):
                await store.abandon_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    preparation_digest=prepared.stage.preparation_digest,
                    expected_run_epoch=running.run_epoch,
                )
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None and active.stage == prepared.stage

            store = await _reopen_store(session_store_case, store)
            assert (
                await store.load_model_completion_stage_dispatch(session_id, stage_id) == dispatch
            )
            assert (
                await store.mark_model_completion_stage_dispatched(
                    session_id,
                    stage=prepared.stage,
                )
                == dispatch
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_dispatch_fence_is_atomic_with_retry(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_dispatch_retry_race"
            interaction_id = "interaction-model-completion-dispatch-retry-race"
            logical_step_id = "mstep_44444444444444444444444444444444"
            first_stage_id = f"{logical_step_id}:dispatch:0"
            retry_stage_id = f"{logical_step_id}:dispatch:1"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=first_stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": interaction_id,
                        "model_step_id": logical_step_id,
                        "model_attempt_id": "matt_44444444444444444444444444444444",
                        "request_fingerprint": "4" * 64,
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            original_load_active = store.load_active_model_completion_stage
            retry_prepared = False

            async def load_active_with_retry(stage_session_id: str):
                nonlocal retry_prepared
                active = await original_load_active(stage_session_id)
                if not retry_prepared:
                    retry_prepared = True
                    await store.prepare_model_completion_stage(
                        session_id,
                        request=ModelCompletionStageRequest(
                            stage_id=retry_stage_id,
                            logical_step_id=logical_step_id,
                            dispatch_ordinal=1,
                            intent={
                                "interaction_id": interaction_id,
                                "model_step_id": logical_step_id,
                                "model_attempt_id": "matt_55555555555555555555555555555555",
                                "request_fingerprint": "5" * 64,
                            },
                        ),
                        expected_statuses={SessionStatus.RUNNING},
                        expected_run_epoch=running.run_epoch,
                        expected_transcript_cursor=0,
                    )
                return active

            monkeypatch.setattr(
                store,
                "load_active_model_completion_stage",
                load_active_with_retry,
            )
            dispatch = await store.mark_model_completion_stage_dispatched(
                session_id,
                stage=first.stage,
            )
            if not retry_prepared:
                await store.load_active_model_completion_stage(session_id)

            settlement = await store.load_model_completion_stage_settlement(
                session_id,
                first_stage_id,
            )
            assert retry_prepared is True
            assert settlement is not None
            assert settlement.disposition is ModelCompletionStageDisposition.SUPERSEDED
            assert settlement.superseding_stage_id == retry_stage_id
            assert dispatch.dispatched_at < settlement.settled_at
            active = await original_load_active(session_id)
            assert active is not None and active.stage.stage_id == retry_stage_id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_exact_retry_supersedes_prior_model_stage(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_exact_retry_settlement"
            interaction_id = "interaction-model-completion-exact-retry"
            logical_step_id = "mstep_11111111111111111111111111111111"
            first_stage_id = f"{logical_step_id}:dispatch:0"
            retry_stage_id = f"{logical_step_id}:dispatch:1"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            first_intent = {
                "interaction_id": interaction_id,
                "model_step_id": logical_step_id,
                "model_attempt_id": "matt_11111111111111111111111111111111",
                "request_fingerprint": "1" * 64,
            }
            retry_intent = {
                "interaction_id": interaction_id,
                "model_step_id": logical_step_id,
                "model_attempt_id": "matt_22222222222222222222222222222222",
                "request_fingerprint": "2" * 64,
            }
            first = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=first_stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=0,
                    intent=first_intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            retry = await store.prepare_model_completion_stage(
                session_id,
                request=ModelCompletionStageRequest(
                    stage_id=retry_stage_id,
                    logical_step_id=logical_step_id,
                    dispatch_ordinal=1,
                    intent=retry_intent,
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )

            settlement = await store.load_model_completion_stage_settlement(
                session_id,
                first_stage_id,
            )
            assert settlement is not None
            assert settlement.disposition is ModelCompletionStageDisposition.SUPERSEDED
            assert settlement.superseding_stage_id == retry_stage_id
            assert settlement.model_attempt_id == first_intent["model_attempt_id"]
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None and active.stage == retry.stage

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="already terminally settled as superseded",
            ):
                await store.complete_model_completion_stage(
                    session_id,
                    stage_id=first_stage_id,
                    publication=_assistant_model_completion_publication(
                        session_id=session_id,
                        stage_id=first_stage_id,
                        logical_step_id=logical_step_id,
                        intent=first_intent,
                        completion_event_id="superseded-model-completed",
                        source_transcript_cursor=0,
                        assistant_message=Message.text("assistant", "stale"),
                        event_payload={
                            "model_step_id": first_intent["model_step_id"],
                            "model_attempt_id": first_intent["model_attempt_id"],
                        },
                    ),
                )
            assert (
                await store.load_model_completion_stage(
                    session_id,
                    first_stage_id,
                )
                == first.stage
            )
            assert await store.load_active_model_completion_stage(session_id) == active

            retry_publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=retry_stage_id,
                logical_step_id=logical_step_id,
                intent=retry_intent,
                completion_event_id="retry-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "winner"),
                event_payload={
                    "model_step_id": retry_intent["model_step_id"],
                    "model_attempt_id": retry_intent["model_attempt_id"],
                },
            )
            await store.complete_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                publication=retry_publication,
            )
            promoted = await store.promote_model_completion_stage(
                session_id,
                stage_id=retry_stage_id,
                expected_run_epoch=running.run_epoch,
            )
            assert promoted.replayed is False
            assert await store.load_active_model_completion_stage(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_concurrent_replay(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_concurrency"
            stage_id = "model-step:concurrent"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            request = ModelCompletionStageRequest(
                stage_id=stage_id,
                logical_step_id=stage_id,
                dispatch_ordinal=0,
                intent={"logical_step": 1, "request_fingerprint": "shared"},
                reservation_ids=["reservation-shared"],
            )

            async def prepare():
                return await store.prepare_model_completion_stage(
                    session_id,
                    request=request,
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )

            preparation_results = await asyncio.gather(prepare(), prepare())
            assert sorted(result.replayed for result in preparation_results) == [False, True]
            assert sorted(result.dispatch_authorized for result in preparation_results) == [
                False,
                True,
            ]
            assert preparation_results[0].stage == preparation_results[1].stage

            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1, "request_fingerprint": "shared"},
                completion_event_id="concurrent-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text("assistant", "one authoritative answer"),
                reservation_ids=("reservation-shared",),
            )

            async def complete():
                return await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )

            completion_results = await asyncio.gather(complete(), complete())
            assert sorted(result.replayed for result in completion_results) == [False, True]
            assert all(result.dispatch_authorized is False for result in completion_results)
            assert completion_results[0].stage == completion_results[1].stage

            async def promote():
                return await store.promote_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    expected_run_epoch=running.run_epoch,
                )

            promotion_results = await asyncio.gather(promote(), promote())
            assert sorted(result.replayed for result in promotion_results) == [False, True]
            assert promotion_results[0].receipt == promotion_results[1].receipt
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_zero_message_model_retry_has_one_logical_winner(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_zero_message_model_retry_winner"
            logical_step_id = "model-step:zero-message-winner"
            stage_ids = (
                "model-step:zero-message:attempt-0",
                "model-step:zero-message:attempt-1",
            )
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            running = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            for ordinal, stage_id in enumerate(stage_ids):
                prepared = await store.prepare_model_completion_stage(
                    session_id,
                    request=ModelCompletionStageRequest(
                        stage_id=stage_id,
                        logical_step_id=logical_step_id,
                        dispatch_ordinal=ordinal,
                        intent={"logical_step": "zero-message", "attempt": ordinal},
                    ),
                    expected_statuses={SessionStatus.RUNNING},
                    expected_run_epoch=running.run_epoch,
                    expected_transcript_cursor=0,
                )
            assert prepared.dispatch_authorized is True
            for ordinal, stage_id in enumerate(stage_ids):
                publication = _assistant_model_completion_publication(
                    session_id=session_id,
                    stage_id=stage_id,
                    logical_step_id=logical_step_id,
                    intent={"logical_step": "zero-message", "attempt": ordinal},
                    completion_event_id=f"zero-message-completed-{ordinal}",
                    source_transcript_cursor=0,
                    assistant_message=None,
                    classification={
                        "type": "invalid",
                        "reason": "assistant produced no content",
                    },
                )
                completed = await store.complete_model_completion_stage(
                    session_id,
                    stage_id=stage_id,
                    publication=publication,
                )
                assert completed.dispatch_authorized is False

            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None
            assert active.stage.stage_id == stage_ids[1]
            assert active.stage.dispatch_ordinal == 1

            outcomes = await asyncio.gather(
                *(
                    store.promote_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        expected_run_epoch=running.run_epoch,
                    )
                    for stage_id in stage_ids
                ),
                return_exceptions=True,
            )
            winners = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
            conflicts = [
                outcome
                for outcome in outcomes
                if isinstance(outcome, SessionModelCompletionStageConflict)
            ]
            assert len(winners) == 1
            assert len(conflicts) == 1
            winner = winners[0]
            assert winner.replayed is False
            assert winner.receipt.publication_id == logical_step_id
            assert winner.receipt.transcript_start_cursor == 0
            assert winner.receipt.transcript_end_cursor == 0
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == []
            assert [event.id for event in await store.load_events(session_id)] == [
                "zero-message-completed-1"
            ]
            assert await store.load_active_model_completion_stage(session_id) is None

            replayed = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_ids[1],
                expected_run_epoch=running.run_epoch + 100,
            )
            assert replayed.replayed is True
            assert replayed.receipt == winner.receipt
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_model_completion_stage_recovers_lost_acknowledgements(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_model_completion_stage_lost_ack"
            stage_id = "model-step:lost-ack"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
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
                    logical_step_id=stage_id,
                    dispatch_ordinal=0,
                    intent={"logical_step": 1},
                    reservation_ids=["reservation-lost-ack"],
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=running.run_epoch,
                expected_transcript_cursor=0,
            )
            publication = _assistant_model_completion_publication(
                session_id=session_id,
                stage_id=stage_id,
                logical_step_id=stage_id,
                intent={"logical_step": 1},
                completion_event_id="lost-ack-model-completed",
                source_transcript_cursor=0,
                assistant_message=Message.text(
                    "assistant",
                    "committed before cancellation",
                ),
                reservation_ids=("reservation-lost-ack",),
            )

            original_complete = store._complete_model_completion_stage_atomic
            completion_committed = asyncio.Event()
            hold_completion_ack = asyncio.Event()

            async def complete_then_hold_ack(prepared):
                result = await original_complete(prepared)
                assert result.replayed is False
                completion_committed.set()
                await hold_completion_ack.wait()
                return result

            with monkeypatch.context() as completion_patch:
                completion_patch.setattr(
                    store,
                    "_complete_model_completion_stage_atomic",
                    complete_then_hold_ack,
                )
                completing = asyncio.create_task(
                    store.complete_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        publication=publication,
                    )
                )
                await asyncio.wait_for(completion_committed.wait(), timeout=5)
                completing.cancel("terminal stage acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await completing
                assert cancellation.value.args == ("terminal stage acknowledgement lost",)

            completed_stage = await store.load_model_completion_stage(session_id, stage_id)
            assert completed_stage is not None
            assert completed_stage.state == "completed"
            store = await _reopen_store(session_store_case, store)
            terminal_replay = await store.complete_model_completion_stage(
                session_id,
                stage_id=stage_id,
                publication=publication,
            )
            assert terminal_replay.replayed is True
            assert terminal_replay.stage == completed_stage

            original_promote = store._promote_model_completion_stage_atomic
            promotion_committed = asyncio.Event()
            hold_promotion_ack = asyncio.Event()

            async def promote_then_hold_ack(**kwargs):
                result = await original_promote(**kwargs)
                assert result.replayed is False
                promotion_committed.set()
                await hold_promotion_ack.wait()
                return result

            with monkeypatch.context() as promotion_patch:
                promotion_patch.setattr(
                    store,
                    "_promote_model_completion_stage_atomic",
                    promote_then_hold_ack,
                )
                promoting = asyncio.create_task(
                    store.promote_model_completion_stage(
                        session_id,
                        stage_id=stage_id,
                        expected_run_epoch=running.run_epoch,
                    )
                )
                await asyncio.wait_for(promotion_committed.wait(), timeout=5)
                promoting.cancel("promotion acknowledgement lost")
                with pytest.raises(asyncio.CancelledError) as cancellation:
                    await promoting
                assert cancellation.value.args == ("promotion acknowledgement lost",)

            committed_receipt = await store.load_runtime_publication_receipt(
                session_id,
                stage_id,
            )
            assert committed_receipt is not None
            assert await store.load_checkpoint(
                session_id
            ) == _model_completion_publication_checkpoint(publication)
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)

            store = await _reopen_store(session_store_case, store)

            promotion_replay = await store.promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=0,
            )
            assert promotion_replay.replayed is True
            assert promotion_replay.receipt == committed_receipt
            assert await store.load_transcript(session_id) == list(publication.transcript_messages)
            assert await store.load_events(session_id) == list(publication.events)
        finally:
            await _close_store(store)

    asyncio.run(run())

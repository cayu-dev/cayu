"""Stage transaction conformance; this is not app.run failover acceptance."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from importlib.metadata import version

import pytest
from tests.core._execution_profile_fixtures import (
    admit_test_invocation,
    create_admitted_session,
    interrupt_and_release_test_invocation,
    runtime_interaction_started_event,
)
from tests.core.test_model_failover_profiles import _plan, _profile

from cayu import AgentSpec, CayuApp, Event, EventType, Message
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelProviderError
from cayu.runtime._execution_profile_admission import bind_model_failover_execution_profile
from cayu.runtime._invocation_lifecycle import (
    AdmittedInvocationBinding,
    _authenticated_invocation_context,
)
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.runtime._model_failover import FailoverObservation
from cayu.runtime._model_failover_stage import ModelFailoverStageAdmission
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.execution_units import new_model_step_identity
from cayu.runtime.retry_policy import RetryPolicy, retry_decision
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY, ModelFailoverProgress
from cayu.sessions.base import (
    InMemorySessionStore,
    ModelCompletionStageDisposition,
    ModelCompletionStageRequest,
    RunRequest,
    RuntimePublicationRequest,
    SessionModelCompletionStageConflict,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    model_completion_stage_settlement_request,
    runtime_publication_checkpoint_mutation,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.exposure import ToolCapabilityCeiling

# Existing fixture names now refer to the actual supported native implementations.
_StageMemoryStore = InMemorySessionStore
_StageSQLiteStore = SQLiteSessionStore


class _LosePreparationAcknowledgement(SessionStore):
    fail_after_commit = True

    async def _prepare_model_completion_stage_atomic(self, prepared):
        result = await super()._prepare_model_completion_stage_atomic(prepared)
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise OSError("lost preparation acknowledgement")
        return result


class _CommitThenRaiseMemory(_LosePreparationAcknowledgement, _StageMemoryStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


class _CommitThenRaiseSQLite(_LosePreparationAcknowledgement, _StageSQLiteStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


async def _initial_admission(
    store,
    *,
    provider=None,
    context_overflow_policy=None,
    max_total_attempts=5,
    session_id="session",
):
    primary = _profile("primary", "small", runtime_version=version("cayu"))
    backup = _profile("backup", "large", runtime_version=version("cayu"))
    plan = _plan(primary, backup)
    plan = type(plan).model_validate({**plan.payload(), "max_total_attempts": max_total_attempts})
    bound = bind_model_failover_execution_profile(plan=plan, candidate_profiles=(primary, backup))
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([], name="primary") if provider is None else provider,
        default=True,
    )
    app.register_agent(
        AgentSpec(name="agent", model="small"), context_overflow_policy=context_overflow_policy
    )
    fixture = await create_admitted_session(
        store,
        request=RunRequest(agent_name="agent", session_id=session_id, messages=[]),
        provider_name="primary",
        model="small",
        execution_profile=bound,
        durable_system_prompt="answer safely",
        interaction_id="interaction",
        app=app,
    )
    session = fixture.session
    active = fixture.active_invocation_profile
    context = _authenticated_invocation_context(
        active_profile=active,
        binding=AdmittedInvocationBinding(
            session_id=session.id,
            session_instance_id=session.instance_id,
            interaction_id=active.interaction_id,
            run_epoch=session.run_epoch,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=None,
        ),
        registered_agent=app._agents["agent"],
        registered_provider=app._providers["primary"],
        registered_environment=None,
        validated_profile=active.profile,
        runtime_hooks=(),
        loop_policies=(),
        request_loop_policies=(),
        budget_policy=None,
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
    )
    attempt = new_model_step_identity().new_attempt()
    cursor = await store.load_transcript_cursor(session.id)
    progress = ModelFailoverProgress(
        session_id=session.id,
        session_instance_id=session.instance_id,
        interaction_id=active.interaction_id,
        execution_profile_fingerprint=bound.fingerprint,
        plan=plan,
        generation=1,
        candidate_index=0,
        logical_step_id=attempt.model_step_id,
        stage_id="initial-stage",
        request_fingerprint="d" * 64,
        source_run_epoch=session.run_epoch,
        source_transcript_cursor=cursor,
        projection_cursor=0,
        dispatch_ordinal=0,
        attempts_used=1,
        candidate_attempt=1,
        provider_effect_observed=False,
    )
    admission = ModelFailoverStageAdmission(
        invocation_context=context,
        candidate_profiles=(primary, backup),
        expected=None,
        successor=progress,
        transition="initial",
        source_preparation_digest=None,
    )
    return admission, attempt


def _request(admission, attempt):
    progress = admission.successor
    target = progress.plan.candidates[progress.candidate_index]
    return ModelCompletionStageRequest(
        stage_id=progress.stage_id,
        logical_step_id=progress.logical_step_id,
        dispatch_ordinal=progress.dispatch_ordinal,
        intent={
            **attempt.payload(),
            "interaction_id": progress.interaction_id,
            "provider_name": target.provider_name,
            "requested_model": target.model,
            "request_fingerprint": progress.request_fingerprint,
            "recovery_context": {
                "execution_profile_fingerprint": progress.execution_profile_fingerprint
            },
        },
    )


async def _prepare(store, admission, attempt):
    progress = admission.successor
    return await store._prepare_model_completion_stage_with_failover(
        progress.session_id,
        request=_request(admission, attempt),
        admission=admission,
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=progress.source_run_epoch,
        expected_transcript_cursor=progress.source_transcript_cursor,
    )


async def _complete_stage(store, stage, attempt):
    checkpoint = await store.load_checkpoint(stage.session_id)
    classification = {"type": "final"}
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id=stage.session_id,
        interaction_id=stage.intent["interaction_id"],
        payload={
            **attempt.payload(),
            "provider_name": stage.intent["provider_name"],
            "model": stage.intent["requested_model"],
            "requested_model": stage.intent["requested_model"],
            "transcript_cursor": stage.source_transcript_cursor + 1,
            "step_classification": classification,
        },
    )
    target = dict(checkpoint or {})
    target[LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY] = ModelStepPublicationCheckpoint(
        logical_step_id=stage.logical_step_id,
        stage_id=stage.stage_id,
        source_transcript_cursor=stage.source_transcript_cursor,
        transcript_end_cursor=stage.source_transcript_cursor + 1,
        completion_event_id=event.id,
        classification=classification,
        assistant_message_published=True,
    ).model_dump(mode="json")
    return await store.complete_model_completion_stage(
        stage.session_id,
        stage_id=stage.stage_id,
        publication=RuntimePublicationRequest(
            publication_id=stage.logical_step_id,
            kind="model-step",
            interaction_id=stage.intent["interaction_id"],
            intent=stage.intent,
            mutation=runtime_publication_checkpoint_mutation(checkpoint, target),
            transcript_messages=(Message.text("assistant", "done"),),
            events=(event,),
        ),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("predecessor", ["completed", "abandoned", "settled"])
def test_new_admitted_interaction_inherits_published_candidate_without_reset(
    tmp_path, backend, predecessor
):
    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "continued-route.sqlite")
        )
        try:
            initial, attempt = await _initial_admission(store)
            first = await _prepare(store, initial, attempt)
            await store.mark_model_completion_stage_dispatched("session", stage=first.stage)
            fallback = replace(
                initial,
                expected=initial.successor,
                successor=ModelFailoverProgress.model_validate(
                    {
                        **initial.successor.payload(),
                        "candidate_index": 1,
                        "generation": 2,
                        "stage_id": "fallback-stage",
                        "dispatch_ordinal": 1,
                        "attempts_used": 2,
                        "projection_cursor": initial.successor.source_transcript_cursor,
                    }
                ),
                transition="fallback",
                source_preparation_digest=first.stage.preparation_digest,
                failure=ModelProviderError(
                    "unavailable", provider="primary", status_code=503, retryable=True
                ),
                retry=retry_decision(
                    policy=RetryPolicy(max_attempts=1),
                    attempt=1,
                    error="unavailable",
                    status_code=503,
                    retryable=True,
                ),
                observation=FailoverObservation(
                    provider_name="primary",
                    caller_cancelled=False,
                    completion_observed=False,
                    provider_effect_observed=False,
                    provider_operation_owned=False,
                    cleanup_settled=True,
                ),
            )
            backup_attempt = attempt.new_attempt()
            backup = await _prepare(store, fallback, backup_attempt)
            if predecessor != "abandoned":
                await store.mark_model_completion_stage_dispatched("session", stage=backup.stage)
            premature_attempt = new_model_step_identity().new_attempt()
            premature = ModelFailoverStageAdmission(
                invocation_context=initial.invocation_context,
                candidate_profiles=initial.candidate_profiles,
                expected=fallback.successor,
                successor=ModelFailoverProgress.model_validate(
                    {
                        **fallback.successor.payload(),
                        "logical_step_id": premature_attempt.model_step_id,
                        "stage_id": "premature-stage",
                        "generation": 3,
                        "dispatch_ordinal": 0,
                        "attempts_used": 1,
                        "candidate_attempt": 1,
                    }
                ),
                transition="next_step",
                source_preparation_digest=backup.stage.preparation_digest,
            )
            for terminal_recorded in (False, True) if predecessor == "completed" else (False,):
                if terminal_recorded:
                    await _complete_stage(store, backup.stage, backup_attempt)
                before = await store.load_checkpoint("session")
                # The existing active-stage guard distinguishes in-flight
                # work from terminal material that has not been published.
                with pytest.raises(
                    SessionModelCompletionStageConflict,
                    match="published" if terminal_recorded else "already active",
                ):
                    await _prepare(store, premature, premature_attempt)
                assert await store.load_checkpoint("session") == before
                assert await store.load_model_completion_stage("session", "premature-stage") is None
            if predecessor == "completed":
                await store.promote_model_completion_stage(
                    "session",
                    stage_id=backup.stage.stage_id,
                    expected_run_epoch=backup.stage.source_run_epoch,
                )
            elif predecessor == "abandoned":
                await store.abandon_model_completion_stage(
                    "session",
                    stage_id=backup.stage.stage_id,
                    preparation_digest=backup.stage.preparation_digest,
                    expected_run_epoch=backup.stage.source_run_epoch,
                )
            else:
                await store.publish_interaction_transition(
                    "session",
                    event=Event(
                        type=EventType.INTERACTION_FAILED,
                        session_id="session",
                        interaction_id=fallback.successor.interaction_id,
                    ),
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.FAILED,
                    model_completion_stage_settlement=model_completion_stage_settlement_request(
                        backup.stage,
                        interaction_id=fallback.successor.interaction_id,
                        disposition=ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN,
                        reason_code="provider_stream_failed",
                        execution_profile_fingerprint=fallback.successor.execution_profile_fingerprint,
                        settlement_run_epoch=backup.stage.source_run_epoch,
                        settled_reservation_ids=backup.stage.reservation_ids,
                    ),
                )
            # A terminal disposition or no-dispatch receipt does not authorize
            # a fresh logical request inside the interrupted interaction.
            if predecessor != "completed":
                checkpoint = await store.load_checkpoint("session")
                with pytest.raises((SessionModelCompletionStageConflict, SessionStatusConflict)):
                    await _prepare(store, premature, premature_attempt)
                assert await store.load_checkpoint("session") == checkpoint
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = _StageSQLiteStore(tmp_path / "continued-route.sqlite")
                restored = await store.load_model_completion_stage("session", backup.stage.stage_id)
                if predecessor == "abandoned":
                    assert restored is None
                else:
                    assert restored is not None
                    assert restored.state == (
                        "completed" if predecessor == "completed" else "in_flight"
                    )
                    assert restored.preparation_digest == backup.stage.preparation_digest
                restored_checkpoint = await store.load_checkpoint("session")
                assert restored_checkpoint is not None
                assert (
                    restored_checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY]
                    == fallback.successor.payload()
                )
            await interrupt_and_release_test_invocation(store, "session")
            app = CayuApp(session_store=store, enable_logging=False)
            session = await admit_test_invocation(
                store,
                "session",
                interaction_started_event=runtime_interaction_started_event(
                    app, session_id="session", interaction_id="next-interaction", agent_name="agent"
                ),
            )
            checkpoint = await store.load_checkpoint("session")
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert active is not None
            old_context = initial.invocation_context
            context = _authenticated_invocation_context(
                active_profile=active,
                binding=replace(
                    old_context.binding,
                    interaction_id=active.interaction_id,
                    run_epoch=session.run_epoch,
                ),
                registered_agent=old_context.registered_agent,
                registered_provider=old_context.registered_provider,
                registered_environment=None,
                validated_profile=active.profile,
                runtime_hooks=(),
                loop_policies=(),
                request_loop_policies=(),
                budget_policy=None,
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            )
            next_attempt = new_model_step_identity().new_attempt()
            successor = ModelFailoverProgress.model_validate(
                {
                    **fallback.successor.payload(),
                    "interaction_id": active.interaction_id,
                    "source_run_epoch": session.run_epoch,
                    "source_transcript_cursor": await store.load_transcript_cursor("session"),
                    "logical_step_id": next_attempt.model_step_id,
                    "stage_id": "next-interaction-stage",
                    "generation": 3,
                    "dispatch_ordinal": 0,
                    "attempts_used": 1,
                    "candidate_attempt": 1,
                }
            )
            continuation = ModelFailoverStageAdmission(
                invocation_context=context,
                candidate_profiles=initial.candidate_profiles,
                expected=fallback.successor,
                successor=successor,
                transition="next_step",
                source_preparation_digest=backup.stage.preparation_digest,
            )
            before = await store.load_checkpoint("session")
            before_events = await store.load_events("session")
            with pytest.raises(SessionModelCompletionStageConflict):
                await _prepare(
                    store,
                    replace(continuation, source_preparation_digest="f" * 64),
                    next_attempt,
                )
            assert await store.load_checkpoint("session") == before
            assert await store.load_events("session") == before_events
            assert await store.load_active_model_completion_stage("session") is None
            continued = await _prepare(store, continuation, next_attempt)
            assert continued.dispatch_authorized
            assert continued.stage.intent["provider_name"] == "backup"
            assert continued.stage.intent["requested_model"] == "large"
            assert session.provider_name == "primary" and session.model == "small"
            replay = await _prepare(store, continuation, next_attempt)
            assert replay.replayed and not replay.dispatch_authorized
            await store.mark_model_completion_stage_dispatched("session", stage=continued.stage)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_stage_preparation_dispatch_fallback_and_exact_replay(tmp_path, backend):
    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "route.sqlite")
        )
        try:
            admission, attempt = await _initial_admission(store)
            first = await _prepare(store, admission, attempt)
            assert first.dispatch_authorized and not first.replayed
            assert len(first.prepared_events) == 1
            assert first.prepared_events[0].type is EventType.MODEL_FAILOVER_SELECTED
            assert first.prepared_events[0].payload["provider"] == "primary"
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == admission.successor.payload()
            await store.mark_model_completion_stage_dispatched("session", stage=first.stage)
            next_progress = ModelFailoverProgress.model_validate(
                {
                    **admission.successor.payload(),
                    "generation": 2,
                    "candidate_index": 1,
                    "stage_id": "backup-stage",
                    "dispatch_ordinal": 1,
                    "attempts_used": 2,
                    "projection_cursor": admission.successor.source_transcript_cursor,
                    "request_fingerprint": "e" * 64,
                }
            )
            provider_failure = ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )
            fallback = replace(
                admission,
                expected=admission.successor,
                successor=next_progress,
                transition="fallback",
                source_preparation_digest=first.stage.preparation_digest,
                failure=provider_failure,
                retry=retry_decision(
                    policy=RetryPolicy(max_attempts=1),
                    attempt=1,
                    error="unavailable",
                    status_code=503,
                    retryable=True,
                ),
                observation=FailoverObservation(
                    provider_name="primary",
                    caller_cancelled=False,
                    completion_observed=False,
                    provider_effect_observed=False,
                    provider_operation_owned=False,
                    cleanup_settled=True,
                ),
            )
            # Preparation retains detached eligibility, not an exception still
            # owned and mutable by provider code after the failure was observed.
            assert fallback.observation is not None
            with pytest.raises(ValueError, match="does not authorize"):
                replace(
                    fallback,
                    observation=replace(fallback.observation, provider_name="another-execution"),
                )
            provider_failure.status_code = 401
            provider_failure.retryable = False
            assert fallback.failure is not provider_failure
            second_attempt = attempt.new_attempt()
            second = await _prepare(store, fallback, second_attempt)
            assert second.dispatch_authorized
            assert len(second.prepared_events) == 1
            assert second.prepared_events[0].payload["provider"] == "backup"
            assert second.prepared_events[0].payload["previous_provider"] == "primary"
            assert second.prepared_events[0].payload["status_code"] == 503
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == next_progress.payload()
            settlement = await store.load_model_completion_stage_settlement(
                "session", first.stage.stage_id
            )
            assert settlement is not None
            assert settlement.disposition is ModelCompletionStageDisposition.SUPERSEDED
            assert settlement.superseding_stage_id == second.stage.stage_id
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.mark_model_completion_stage_dispatched("session", stage=first.stage)
            replay = await _prepare(store, fallback, second_attempt)
            assert replay.replayed and not replay.dispatch_authorized
            assert replay.stage == second.stage
            assert not replay.prepared_events
            selections = [
                event
                for event in await store.load_events("session")
                if event.type is EventType.MODEL_FAILOVER_SELECTED
            ]
            assert [event.model_dump(mode="json") for event in selections] == [
                first.prepared_events[0].model_dump(mode="json"),
                second.prepared_events[0].model_dump(mode="json"),
            ]
            await store.mark_model_completion_stage_dispatched("session", stage=second.stage)
            await store.checkpoint("session", {"application": True})
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == next_progress.payload()
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_raw_stage_cannot_bypass_route_ownership(tmp_path, backend):
    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "route.sqlite")
        )
        try:
            admission, attempt = await _initial_admission(store)
            prepared = await _prepare(store, admission, attempt)
            for supplied in (
                _request(admission, attempt),
                _request(admission, attempt).model_copy(update={"intent": prepared.stage.intent}),
                _request(admission, attempt).model_copy(
                    update={"stage_id": "unauthorized-next", "dispatch_ordinal": 1}
                ),
            ):
                with pytest.raises((ValueError, SessionModelCompletionStageConflict)):
                    await store.prepare_model_completion_stage(
                        "session",
                        request=supplied,
                        expected_statuses={SessionStatus.RUNNING},
                        expected_run_epoch=admission.successor.source_run_epoch,
                        expected_transcript_cursor=admission.successor.source_transcript_cursor,
                    )
            active = await store.load_active_model_completion_stage("session")
            assert active is not None and active.stage == prepared.stage
            assert (
                await store.load_model_completion_stage_settlement(
                    "session", prepared.stage.stage_id
                )
                is None
            )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_committed_preparation_acknowledgement_loss_replays_without_dispatch(tmp_path, backend):
    async def scenario():
        store = (
            _CommitThenRaiseMemory()
            if backend == "memory"
            else _CommitThenRaiseSQLite(tmp_path / "ack.sqlite")
        )
        try:
            admission, attempt = await _initial_admission(store)
            with pytest.raises(OSError, match="acknowledgement"):
                await _prepare(store, admission, attempt)
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == admission.successor.payload()
            active = await store.load_active_model_completion_stage("session")
            assert active is not None
            replay = await _prepare(store, admission, attempt)
            assert replay.stage == active.stage
            selections = [
                event
                for event in await store.load_events("session")
                if event.type is EventType.MODEL_FAILOVER_SELECTED
            ]
            assert len(selections) == 1
            assert replay.replayed and not replay.dispatch_authorized
            assert (
                await store.load_model_completion_stage_dispatch("session", active.stage.stage_id)
                is None
            )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_reprepare_requires_positive_abandonment_and_retains_route_budget(tmp_path, backend):
    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "reprepare.sqlite")
        )
        try:
            initial, attempt = await _initial_admission(store)
            prepared = await _prepare(store, initial, attempt)
            successor = ModelFailoverProgress.model_validate(
                {
                    **initial.successor.payload(),
                    "generation": 2,
                    "stage_id": "reprepared-stage",
                    "dispatch_ordinal": 1,
                }
            )
            admission = replace(
                initial,
                expected=initial.successor,
                successor=successor,
                transition="reprepare",
                source_preparation_digest=prepared.stage.preparation_digest,
            )
            later_attempt = attempt.new_attempt()
            with pytest.raises(SessionModelCompletionStageConflict):
                await _prepare(store, admission, later_attempt)
            assert (
                await store.load_model_completion_stage_abandonment(
                    "session", prepared.stage.stage_id
                )
                is None
            )
            result = await store.abandon_model_completion_stage(
                "session",
                stage_id=prepared.stage.stage_id,
                preparation_digest=prepared.stage.preparation_digest,
                expected_run_epoch=initial.successor.source_run_epoch,
            )
            assert not result.replayed
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = _StageSQLiteStore(tmp_path / "reprepare.sqlite")
            assert (
                await store.load_model_completion_stage("session", prepared.stage.stage_id) is None
            )
            abandonment = await store.load_model_completion_stage_abandonment(
                "session", prepared.stage.stage_id
            )
            assert abandonment == result.abandonment
            assert abandonment is not None
            assert abandonment.preparation_digest == admission.source_preparation_digest
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == initial.successor.payload()
            resumed = await _prepare(store, admission, later_attempt)
            assert resumed.dispatch_authorized
            assert not resumed.prepared_events
            checkpoint = await store.load_checkpoint("session")
            assert checkpoint is not None
            assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == successor.payload()
            assert successor.attempts_used == initial.successor.attempts_used == 1
            assert successor.candidate_index == initial.successor.candidate_index
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.mark_model_completion_stage_dispatched("session", stage=prepared.stage)
            await store.mark_model_completion_stage_dispatched("session", stage=resumed.stage)
            replay = await _prepare(store, admission, later_attempt)
            assert replay.replayed and not replay.dispatch_authorized
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())

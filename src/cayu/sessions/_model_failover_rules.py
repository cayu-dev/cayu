"""Shared model-failover predecessor, admission and checkpoint publication rules."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from cayu._validation import copy_durable_json_value
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    execution_profile_from_session_metadata,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
)
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverProgress,
    ModelFailoverSelection,
    copy_model_failover_state,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ActiveModelCompletionStage,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._model_stage_rules import model_completion_stage_settlement_request
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)
from cayu.sessions.records import Session, SessionStatus


def _model_failover_selection_event(
    prepared: model_stage_rules._PreparedModelCompletionStage,
    *,
    session: Session,
    prepared_at: datetime,
) -> Event | None:
    admission = prepared.failover_admission
    if admission is None or admission.transition not in {"initial", "fallback"}:
        return None
    from cayu.execution_profiles import (
        event_with_execution_profile_authority,
    )

    progress = admission.successor
    target = progress.plan.candidates[progress.candidate_index]
    previous = admission.expected
    previous_target = (
        None if previous is None else previous.plan.candidates[previous.candidate_index]
    )
    eligibility = admission.intent_payload().get("eligibility", {})
    event = Event(
        id="evt_failover_" + prepared.request_digest,
        type=EventType.MODEL_FAILOVER_SELECTED,
        timestamp=prepared_at,
        session_id=session.id,
        interaction_id=progress.interaction_id,
        agent_name=session.agent_name,
        environment_name=session.environment_name,
        payload={
            "schema_version": 1,
            "route_id": progress.route_id,
            "route_generation": progress.generation,
            "stage_id": progress.stage_id,
            "model_step_id": progress.logical_step_id,
            "model_attempt_id": prepared.request.intent["model_attempt_id"],
            "provider": target.provider_name,
            "model": target.model,
            "configured_provider": progress.plan.candidates[0].provider_name,
            "configured_model": progress.plan.candidates[0].model,
            "candidate_index": progress.candidate_index,
            "candidate_count": len(progress.plan.candidates),
            "attempts_used": progress.attempts_used,
            "max_total_attempts": progress.plan.max_total_attempts,
            "previous_provider": None if previous_target is None else previous_target.provider_name,
            "previous_model": None if previous_target is None else previous_target.model,
            "previous_stage_id": (
                previous.stage_id if isinstance(previous, ModelFailoverProgress) else None
            ),
            "reason": admission.transition,
            "status_code": eligibility.get("status_code"),
        },
    )
    event = event_with_runtime_generated_id(event)
    event = event_with_runtime_payload_authority(event, "model_step_id", "model_attempt_id")
    return event_with_runtime_envelope_authority(
        event_with_execution_profile_authority(event, admission.invocation_context.profile),
        "session_id",
        "interaction_id",
    )


def _validate_model_failover_selection_replay(expected: Event, observed: Event | None) -> None:
    if observed is None or publication_contracts._runtime_publication_event_digest(
        observed
    ) != publication_contracts._runtime_publication_event_digest(expected):
        raise SessionModelCompletionStageConflict(
            "Model failover preparation lost its exact durable selection event."
        )


def _model_failover_predecessor_storage_keys(
    prepared: model_stage_rules._PreparedModelCompletionStage,
) -> tuple[str, ...]:
    admission = prepared.failover_admission
    if admission is None or not isinstance(admission.expected, ModelFailoverProgress):
        return ()
    return _model_failover_progress_storage_keys(admission.expected)


def _model_failover_progress_storage_keys(expected: ModelFailoverProgress) -> tuple[str, ...]:
    from cayu.runtime.provider_operations import provider_operation_resolution_storage_key

    _, _, preparation_key, terminal_key = (
        model_stage_rules._model_completion_stage_storage_identity(
            expected.session_id, expected.stage_id
        )
    )
    return (
        preparation_key,
        terminal_key,
        model_stage_rules._model_completion_stage_winner_storage_key(expected.logical_step_id),
        publication_contracts._runtime_publication_storage_key(expected.logical_step_id),
        model_stage_rules._model_completion_stage_abandonment_storage_key(expected.stage_id),
        model_stage_rules._model_completion_stage_dispatch_storage_key(expected.stage_id),
        model_stage_rules._model_completion_stage_settlement_storage_key(expected.stage_id),
        provider_operation_resolution_storage_key(expected.stage_id),
    )


def _require_settled_model_failover_predecessor(
    expected: ModelFailoverProgress,
    *,
    session: Session,
    records: Mapping[str, dict[str, Any]],
    active_present: bool,
    target_run_epoch: int,
    new_interaction: bool,
    source_preparation_digest: str | None = None,
) -> None:
    """Prove terminal ownership from the native transaction's exact read set."""

    if active_present:
        raise SessionModelCompletionStageConflict("The previous model stage still owns execution.")
    (
        preparation_key,
        terminal_key,
        winner_key,
        receipt_key,
        abandonment_key,
        dispatch_key,
        settlement_key,
        resolution_key,
    ) = _model_failover_progress_storage_keys(expected)
    if new_interaction and records.get(abandonment_key) is not None:
        if any(
            records.get(key) is not None
            for key in (
                preparation_key,
                terminal_key,
                winner_key,
                receipt_key,
                dispatch_key,
                settlement_key,
                resolution_key,
            )
        ):
            raise SessionModelCompletionStageConflict(
                "Abandoned model stage has competing evidence."
            )
        abandonment = model_stage_rules._reconstruct_model_completion_stage_abandonment(
            records[abandonment_key],
            session_id=session.id,
            stage_id=expected.stage_id,
            storage_key=abandonment_key,
        )
        if (
            (
                source_preparation_digest is not None
                and abandonment.preparation_digest != source_preparation_digest
            )
            or abandonment.logical_step_id != expected.logical_step_id
            or abandonment.dispatch_ordinal != expected.dispatch_ordinal
            or abandonment.source_run_epoch != expected.source_run_epoch
            or abandonment.source_transcript_cursor != expected.source_transcript_cursor
            or abandonment.purpose != "assistant-turn"
            or abandonment.source_run_epoch >= target_run_epoch
        ):
            raise SessionModelCompletionStageConflict("Model abandonment authority changed.")
        return
    source = model_stage_rules._reconstruct_model_completion_stage(
        records.get(preparation_key),
        records.get(terminal_key),
        session_id=session.id,
        stage_id=expected.stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
    )
    capsule = None if source is None else source.intent.get(MODEL_FAILOVER_CHECKPOINT_KEY)
    if (
        source is None
        or (
            source_preparation_digest is not None
            and source.preparation_digest != source_preparation_digest
        )
        or type(capsule) is not dict
        or ModelFailoverProgress.model_validate(capsule.get("successor")) != expected
    ):
        raise SessionModelCompletionStageConflict("Model failover lost its exact source stage.")
    receipt_record = records.get(receipt_key)
    settlement_record = records.get(settlement_key)
    if new_interaction and source.state == "in_flight" and settlement_record is not None:
        settlement = model_stage_rules._reconstruct_model_completion_stage_settlement(
            settlement_record,
            session_id=session.id,
            stage_id=expected.stage_id,
            storage_key=settlement_key,
        )
        if (
            receipt_record is not None
            or records.get(winner_key) is not None
            or settlement.disposition is ModelCompletionStageDisposition.SUPERSEDED
            or settlement.settlement_run_epoch >= target_run_epoch
        ):
            raise SessionModelCompletionStageConflict(
                "Model predecessor is not terminally settled."
            )
        expected_settlement = model_completion_stage_settlement_request(
            source,
            interaction_id=expected.interaction_id,
            disposition=settlement.disposition,
            reason_code=settlement.reason_code,
            execution_profile_fingerprint=expected.execution_profile_fingerprint,
            settlement_run_epoch=settlement.settlement_run_epoch,
            settled_reservation_ids=source.reservation_ids,
        )
        observed = ModelCompletionStageSettlementRequest.model_validate(
            settlement.model_dump(
                mode="python", include=set(ModelCompletionStageSettlementRequest.model_fields)
            )
        )
        if observed != expected_settlement:
            raise SessionModelCompletionStageConflict("Model settlement conflicts with its stage.")
        return
    if source.state != "completed" or receipt_record is None:
        raise SessionModelCompletionStageConflict("Model predecessor is not durably published.")
    receipt = publication_rules._reconstruct_runtime_publication_receipt(
        receipt_record,
        storage_key=receipt_key,
        session_id=session.id,
        publication_id=expected.logical_step_id,
    )
    model_stage_rules._validate_model_completion_stage_winner(
        records.get(winner_key), stage=source, receipt=receipt, winner_storage_key=winner_key
    )


def _model_failover_admission_storage_keys(
    checkpoint: dict[str, Any] | None,
    candidate_profile: ExecutionProfileIdentity | None,
) -> tuple[str, ...]:
    if candidate_profile is None:
        return ()
    raw = None if checkpoint is None else checkpoint.get(MODEL_FAILOVER_CHECKPOINT_KEY)
    if raw is None and candidate_profile.model_failover is None:
        return ()
    previous = None if raw is None else copy_model_failover_state(raw)
    return (
        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
        *(
            _model_failover_progress_storage_keys(previous)
            if isinstance(previous, ModelFailoverProgress)
            else ()
        ),
    )


def _model_failover_checkpoint_after_profile_admission(
    *,
    source_session: Session,
    admitted_session: Session,
    source_checkpoint: dict[str, Any] | None,
    admitted_checkpoint: dict[str, Any] | None,
    candidate_profile: ExecutionProfileIdentity | None,
    records: Mapping[str, dict[str, Any]],
    transcript_cursor: int,
    supports_model_failover: bool,
) -> dict[str, Any] | None:
    """Rebind selection inside validated native admission, before its receipt.

    This is not a callback privilege or a dispatch authorization. All operation
    records and both checkpoint snapshots belong to the same native transaction.
    """
    if candidate_profile is None:
        return admitted_checkpoint
    active = active_invocation_execution_profile_from_checkpoint(source_checkpoint)
    source_profile = (
        execution_profile_from_session_metadata(source_session.metadata)
        if active is None
        else active.profile
    )
    if source_profile.fingerprint == candidate_profile.fingerprint:
        return admitted_checkpoint
    raw = (
        None if source_checkpoint is None else source_checkpoint.get(MODEL_FAILOVER_CHECKPOINT_KEY)
    )
    if raw is None and candidate_profile.model_failover is None:
        return admitted_checkpoint
    if (
        not supports_model_failover
        or not checkpoint_preservation._INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.get()
    ):
        raise SessionRunFenced("Model routing changes require atomic invocation admission.")
    admitted_active = active_invocation_execution_profile_from_checkpoint(admitted_checkpoint)
    if (
        admitted_session.id != source_session.id
        or admitted_session.instance_id != source_session.instance_id
        or admitted_session.run_epoch != source_session.run_epoch + 1
        or admitted_session.status is not SessionStatus.RUNNING
        or admitted_active is None
        or admitted_active.profile != candidate_profile
        or admitted_active.session_id != source_session.id
        or admitted_active.run_epoch != admitted_session.run_epoch
        or (
            active is not None
            and (
                not active_invocation_execution_profile_is_released(
                    active, session_id=source_session.id, run_epoch=source_session.run_epoch
                )
                or active.interaction_id == admitted_active.interaction_id
            )
        )
    ):
        raise SessionRunFenced("Model routing changes require a released invocation boundary.")
    from cayu.sessions._execution_profile_checkpoint import model_failover_progress_for_session

    previous = model_failover_progress_for_session(
        session=source_session,
        execution_profile=source_profile,
        checkpoint=source_checkpoint,
    )
    active_present = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY) is not None
    if isinstance(previous, ModelFailoverProgress):
        _require_settled_model_failover_predecessor(
            previous,
            session=source_session,
            records=records,
            active_present=active_present,
            target_run_epoch=admitted_session.run_epoch,
            new_interaction=True,
        )
    elif active_present:
        raise SessionModelCompletionStageConflict("Profile adoption cannot clear an active stage.")
    if previous is not None and previous.source_transcript_cursor > transcript_cursor:
        raise SessionRunFenced("Model selection exceeds the stored transcript.")
    assert admitted_checkpoint is not None
    updated = copy_durable_json_value(admitted_checkpoint, "checkpoint")
    binding = candidate_profile.model_failover
    if binding is None:
        updated.pop(MODEL_FAILOVER_CHECKPOINT_KEY, None)
        return updated
    same_targets = previous is not None and tuple(
        (item.provider_name, item.model) for item in previous.plan.candidates
    ) == tuple((item.provider_name, item.model) for item in binding.plan.candidates)
    index = previous.candidate_index if same_targets and previous is not None else 0
    projection_cursor = 0 if previous is None else previous.projection_cursor
    if previous is not None and (
        previous.plan.candidates[previous.candidate_index].provider_name,
        previous.plan.candidates[previous.candidate_index].model,
    ) != (binding.plan.candidates[index].provider_name, binding.plan.candidates[index].model):
        projection_cursor = transcript_cursor
    selection = ModelFailoverSelection(
        session_id=admitted_session.id,
        session_instance_id=admitted_session.instance_id,
        execution_profile_fingerprint=candidate_profile.fingerprint,
        plan=binding.plan,
        candidate_index=index,
        origin_id=session_authority_rules._canonical_runtime_publication_digest(
            {
                "source": None if previous is None else previous.model_dump(mode="json"),
                "source_profile": source_profile.fingerprint,
                "target_profile": candidate_profile.fingerprint,
                "session_instance_id": admitted_session.instance_id,
                "interaction_id": admitted_active.interaction_id,
                "run_epoch": admitted_session.run_epoch,
                "transcript_cursor": transcript_cursor,
            }
        ),
        source_run_epoch=admitted_session.run_epoch,
        source_transcript_cursor=transcript_cursor,
        projection_cursor=projection_cursor,
    )
    updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    updated[MODEL_FAILOVER_CHECKPOINT_KEY] = selection.model_dump(mode="json")
    return updated


def _model_failover_preparation_checkpoint(
    prepared: model_stage_rules._PreparedModelCompletionStage,
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    current_transcript_cursor: int,
    active: ActiveModelCompletionStage | None,
    records: Mapping[str, dict[str, Any]],
    replayed: bool,
) -> dict[str, Any] | None:
    """Validate a route against the same snapshot as stage preparation.

    None means no route mutation, not permission to delete the checkpoint.
    The returned checkpoint must commit with preparation and supersession.
    """

    admission = prepared.failover_admission
    if admission is None:
        if (
            prepared.request.purpose == "assistant-turn"
            and checkpoint is not None
            and MODEL_FAILOVER_CHECKPOINT_KEY in checkpoint
        ):
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session.id)
            if decoded is not None and MODEL_FAILOVER_CHECKPOINT_KEY in decoded:
                raise SessionModelCompletionStageConflict(
                    "An ordinary model preparation cannot bypass its failover route."
                )
        return None
    from cayu.runtime._model_failover_stage import model_failover_checkpoint_after_preparation

    if prepared.request.intent.get(MODEL_FAILOVER_CHECKPOINT_KEY) != admission.intent_payload():
        raise SessionModelCompletionStageConflict(
            "Model failover admission changed after preparation."
        )
    updated = model_failover_checkpoint_after_preparation(
        admission,
        session=session,
        checkpoint=checkpoint,
        transcript_cursor=current_transcript_cursor,
        replayed=replayed,
    )
    if replayed:
        return None
    expected = admission.expected
    if expected is None or isinstance(expected, ModelFailoverSelection):
        if active is not None:
            raise SessionModelCompletionStageConflict(
                "A new failover route cannot supersede an existing active stage."
            )
        return updated
    (
        preparation_key,
        terminal_key,
        winner_key,
        receipt_key,
        abandonment_key,
        dispatch_key,
        settlement_key,
        resolution_key,
    ) = _model_failover_predecessor_storage_keys(prepared)
    new_interaction = (
        admission.transition == "next_step"
        and admission.successor.interaction_id != expected.interaction_id
        and admission.successor.source_run_epoch > expected.source_run_epoch
    )
    if admission.transition == "next_step":
        _require_settled_model_failover_predecessor(
            expected,
            session=session,
            records=records,
            active_present=active is not None,
            target_run_epoch=admission.successor.source_run_epoch,
            new_interaction=new_interaction,
            source_preparation_digest=admission.source_preparation_digest,
        )
        return updated
    if admission.transition == "reprepare":
        if active is not None or any(
            records.get(key) is not None
            for key in (
                preparation_key,
                terminal_key,
                winner_key,
                receipt_key,
                dispatch_key,
                settlement_key,
                resolution_key,
            )
        ):
            raise SessionModelCompletionStageConflict(
                "Model failover re-preparation requires an abandoned, undispatched predecessor."
            )
        abandonment_record = records.get(abandonment_key)
        if abandonment_record is None:
            raise SessionModelCompletionStageConflict(
                "Model failover lost its abandonment receipt."
            )
        abandonment = model_stage_rules._reconstruct_model_completion_stage_abandonment(
            abandonment_record,
            session_id=session.id,
            stage_id=expected.stage_id,
            storage_key=abandonment_key,
        )
        if (
            abandonment.preparation_digest != admission.source_preparation_digest
            or abandonment.logical_step_id != expected.logical_step_id
            or abandonment.dispatch_ordinal != expected.dispatch_ordinal
            or abandonment.source_run_epoch != expected.source_run_epoch
            or abandonment.source_transcript_cursor != expected.source_transcript_cursor
            or abandonment.purpose != "assistant-turn"
        ):
            raise SessionModelCompletionStageConflict(
                "Model failover abandonment authority changed."
            )
        return updated
    source = model_stage_rules._reconstruct_model_completion_stage(
        records.get(preparation_key),
        records.get(terminal_key),
        session_id=session.id,
        stage_id=expected.stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
    )
    capsule = None if source is None else source.intent.get(MODEL_FAILOVER_CHECKPOINT_KEY)
    if (
        source is None
        or source.preparation_digest != admission.source_preparation_digest
        or type(capsule) is not dict
        or ModelFailoverProgress.model_validate(capsule.get("successor")) != expected
    ):
        raise SessionModelCompletionStageConflict("Model failover lost its exact source stage.")
    if (
        admission.transition == "retry"
        and active is None
        and source.state == "in_flight"
        and records.get(resolution_key) is not None
    ):
        from cayu.runtime.provider_operations import validate_provider_operation_retry_preparation

        validate_provider_operation_retry_preparation(
            session_id=session.id,
            checkpoint=checkpoint,
            source=source,
            resolution_record=records[resolution_key],
            target_dispatch_ordinal=admission.successor.dispatch_ordinal,
            execution_profile_fingerprint=admission.successor.execution_profile_fingerprint,
            target_run_epoch=admission.successor.source_run_epoch,
        )
    elif active is None or active.stage != source or source.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "A failover retry requires its exact active predecessor."
        )
    return updated

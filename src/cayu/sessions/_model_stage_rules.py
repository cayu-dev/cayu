"""Model-stage preparation, dispatch, settlement and durable replay rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.execution_units import ModelAttemptIdentity
from cayu.messages import Message
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _model_stage as model_stage_contracts
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions import recovery as session_recovery_rules
from cayu.sessions._durable_operation_ownership import DurableOperationOwnership
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
)
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE,
    MODEL_COMPLETION_STAGE_DISPATCH_OPERATION_KEY_PREFIX,
    MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE,
    MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE,
    MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
    MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE,
    MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE,
    MODEL_COMPLETION_WINNER_RECORD_TYPE,
    ActiveModelCompletionStage,
    ModelCompletionStage,
    ModelCompletionStageAbandonment,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageRecoveryFence,
    ModelCompletionStageRequest,
    ModelCompletionStageSettlement,
    SessionModelCompletionStageConflict,
    SessionModelCompletionStageIncomplete,
)
from cayu.sessions._model_stage_publication import _validate_model_completion_stage_publication
from cayu.sessions._operation_records import (
    MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX,
    ModelCompletionStageRelease,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
    RuntimePublicationResult,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.runtime._model_failover_stage import ModelFailoverStageAdmission


@dataclass(frozen=True)
class _PreparedModelCompletionStage:
    session_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    abandonment_storage_key: str
    winner_storage_key: str
    publication_storage_key: str
    request: ModelCompletionStageRequest
    request_digest: str
    expected_statuses: frozenset[SessionStatus]
    expected_run_epoch: int
    expected_transcript_cursor: int
    failover_admission: ModelFailoverStageAdmission | None = None


@dataclass(frozen=True)
class _PreparedModelCompletionStageTerminal:
    session_id: str
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    settlement_storage_key: str
    publication: RuntimePublicationRequest
    publication_material_digest: str
    recovery_fence: ModelCompletionStageRecoveryFence | None = None


@dataclass(frozen=True)
class _PreparedModelCompletionStageAbandonment:
    session_id: str
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    abandonment_storage_key: str
    preparation_digest: str
    expected_run_epoch: int
    stage_source_run_epoch: int


@dataclass(frozen=True)
class _ModelCompletionStagePromotionContext:
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    active_storage_key: str
    winner_storage_key: str
    completion_digest: str


def _model_completion_stage_storage_identity(
    session_id: str,
    stage_id: str,
) -> tuple[str, str, str, str]:
    session_id = require_clean_nonblank(session_id, "session_id")
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    identity_hash = sha256(stage_id.encode("utf-8")).hexdigest()
    return (
        session_id,
        stage_id,
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "prepared:" + identity_hash,
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "completed:" + identity_hash,
    )


def _model_completion_stage_winner_storage_key(logical_step_id: str) -> str:
    logical_step_id = require_clean_nonblank(logical_step_id, "logical_step_id")
    if len(logical_step_id) > 256:
        raise ValueError("logical_step_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
        + "winner:"
        + sha256(logical_step_id.encode("utf-8")).hexdigest()
    )


def _model_completion_stage_abandonment_storage_key(stage_id: str) -> str:
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
        + "abandoned:"
        + sha256(stage_id.encode("utf-8")).hexdigest()
    )


def _model_completion_stage_settlement_storage_key(stage_id: str) -> str:
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
        + "settled:"
        + sha256(stage_id.encode("utf-8")).hexdigest()
    )


def _model_completion_stage_dispatch_storage_key(stage_id: str) -> str:
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_DISPATCH_OPERATION_KEY_PREFIX
        + sha256(stage_id.encode("utf-8")).hexdigest()
    )


def _prepare_model_completion_stage(
    session_id: str,
    request: ModelCompletionStageRequest,
    *,
    expected_statuses: set[SessionStatus],
    expected_run_epoch: int,
    expected_transcript_cursor: int,
    failover_admission: ModelFailoverStageAdmission | None = None,
) -> _PreparedModelCompletionStage:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(request) is not ModelCompletionStageRequest:
        raise TypeError("Model completion staging requires a ModelCompletionStageRequest.")
    try:
        copied_request = ModelCompletionStageRequest(
            stage_id=request.stage_id,
            logical_step_id=request.logical_step_id,
            dispatch_ordinal=request.dispatch_ordinal,
            purpose=request.purpose,
            intent=request.intent,
            reservation_ids=request.reservation_ids,
        )
    except AttributeError as exc:
        raise ValueError("Model completion stage request is malformed.") from exc
    if MODEL_FAILOVER_CHECKPOINT_KEY in copied_request.intent:
        raise ValueError("Public model-stage intents cannot supply model failover authority.")
    if failover_admission is not None:
        from dataclasses import replace

        from cayu.runtime._model_failover_stage import ModelFailoverStageAdmission

        if type(failover_admission) is not ModelFailoverStageAdmission:
            raise TypeError("Model failover requires authenticated stage admission.")
        failover_admission = replace(failover_admission)
        progress = failover_admission.successor
        candidate = progress.plan.candidates[progress.candidate_index]
        recovery_context = copied_request.intent.get("recovery_context")
        if (
            copied_request.purpose != "assistant-turn"
            or progress.session_id != session_id
            or progress.stage_id != copied_request.stage_id
            or progress.logical_step_id != copied_request.logical_step_id
            or progress.dispatch_ordinal != copied_request.dispatch_ordinal
            or progress.source_run_epoch != expected_run_epoch
            or progress.source_transcript_cursor != expected_transcript_cursor
            or copied_request.intent.get("request_fingerprint") != progress.request_fingerprint
            or copied_request.intent.get("provider_name") != candidate.provider_name
            or copied_request.intent.get("requested_model") != candidate.model
            or copied_request.intent.get("model_step_id") != progress.logical_step_id
            or copied_request.intent.get("interaction_id") != progress.interaction_id
            or type(recovery_context) is not dict
            or recovery_context.get("execution_profile_fingerprint")
            != progress.execution_profile_fingerprint
        ):
            raise ValueError("Model failover conflicts with its exact prepared request.")
        ModelAttemptIdentity.model_validate(
            {
                "model_step_id": copied_request.intent.get("model_step_id"),
                "model_attempt_id": copied_request.intent.get("model_attempt_id"),
            }
        )
        copied_request = ModelCompletionStageRequest(
            stage_id=copied_request.stage_id,
            logical_step_id=copied_request.logical_step_id,
            dispatch_ordinal=copied_request.dispatch_ordinal,
            purpose=copied_request.purpose,
            reservation_ids=copied_request.reservation_ids,
            intent={
                **copied_request.intent,
                MODEL_FAILOVER_CHECKPOINT_KEY: failover_admission.intent_payload(),
            },
        )
    allowed_statuses = frozenset(
        session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
    )
    expected_run_epoch = session_authority_rules._validate_required_runtime_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    expected_transcript_cursor = session_authority_rules._validate_required_runtime_fence(
        expected_transcript_cursor,
        "expected_transcript_cursor",
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        copied_request.stage_id,
    )
    winner_key = _model_completion_stage_winner_storage_key(copied_request.logical_step_id)
    request_digest = session_authority_rules._canonical_runtime_publication_digest(
        {
            "session_id": session_id,
            "stage_id": copied_request.stage_id,
            "logical_step_id": copied_request.logical_step_id,
            "dispatch_ordinal": copied_request.dispatch_ordinal,
            "purpose": copied_request.purpose,
            "intent": copied_request.intent,
            "reservation_ids": list(copied_request.reservation_ids),
            "expected_statuses": sorted(str(status) for status in allowed_statuses),
            "expected_run_epoch": expected_run_epoch,
            "expected_transcript_cursor": expected_transcript_cursor,
        }
    )
    return _PreparedModelCompletionStage(
        session_id=session_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        abandonment_storage_key=_model_completion_stage_abandonment_storage_key(
            copied_request.stage_id
        ),
        winner_storage_key=winner_key,
        publication_storage_key=publication_contracts._runtime_publication_storage_key(
            copied_request.logical_step_id
        ),
        request=copied_request,
        request_digest=request_digest,
        expected_statuses=allowed_statuses,
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
        failover_admission=failover_admission,
    )


def _prepare_model_completion_stage_abandonment(
    session_id: str,
    *,
    stage_id: str,
    preparation_digest: str,
    expected_run_epoch: int,
    stage_source_run_epoch: int | None,
) -> _PreparedModelCompletionStageAbandonment:
    session_id, stage_id, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id, stage_id
    )
    try:
        session_authority_rules._require_raw_sha256_digest(preparation_digest)
    except ValueError as exc:
        raise ValueError("preparation_digest must be a lowercase SHA-256 digest.") from exc
    expected_run_epoch = session_authority_rules._validate_required_runtime_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    if stage_source_run_epoch is None:
        stage_source_run_epoch = expected_run_epoch
    stage_source_run_epoch = session_authority_rules._validate_required_runtime_fence(
        stage_source_run_epoch,
        "stage_source_run_epoch",
    )
    if stage_source_run_epoch > expected_run_epoch:
        raise ValueError("stage_source_run_epoch cannot exceed expected_run_epoch.")
    return _PreparedModelCompletionStageAbandonment(
        session_id=session_id,
        stage_id=stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        abandonment_storage_key=_model_completion_stage_abandonment_storage_key(stage_id),
        preparation_digest=preparation_digest,
        expected_run_epoch=expected_run_epoch,
        stage_source_run_epoch=stage_source_run_epoch,
    )


def _prepare_model_completion_stage_terminal(
    session_id: str,
    *,
    stage_id: str,
    publication: RuntimePublicationRequest,
    recovery_fence: ModelCompletionStageRecoveryFence | None = None,
) -> _PreparedModelCompletionStageTerminal:
    session_id, stage_id, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id, stage_id
    )
    if type(publication) is not RuntimePublicationRequest:
        raise TypeError("publication must be a RuntimePublicationRequest.")
    try:
        copied_publication = RuntimePublicationRequest(
            publication_id=publication.publication_id,
            kind=publication.kind,
            interaction_id=publication.interaction_id,
            intent=publication.intent,
            mutation=publication.mutation,
            transcript_messages=publication.transcript_messages,
            events=publication.events,
            operation_record_mutations=publication.operation_record_mutations,
            referenced_events=publication.referenced_events,
        )
    except AttributeError as exc:
        raise ValueError("Model completion publication is malformed.") from exc
    publication_payload = copied_publication.model_dump(mode="json")
    return _PreparedModelCompletionStageTerminal(
        session_id=session_id,
        stage_id=stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        settlement_storage_key=_model_completion_stage_settlement_storage_key(stage_id),
        publication=copied_publication,
        publication_material_digest=session_authority_rules._canonical_runtime_publication_digest(
            publication_payload
        ),
        recovery_fence=recovery_fence,
    )


def _validate_model_completion_stage_recovery_fence(
    fence: ModelCompletionStageRecoveryFence,
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    stage: ModelCompletionStage,
    active_record: dict[str, Any] | None,
    dispatch_record: dict[str, Any] | None,
    now: datetime,
) -> None:
    """Admit a new synthetic terminal inside the store's existing transaction."""
    if (
        session.instance_id != fence.expected_session_instance_id
        or session.run_epoch != fence.expected_run_epoch
        or active_invocation_execution_profile_from_checkpoint(checkpoint)
        != fence.expected_active_profile
        or session_recovery_rules._active_unexpired_incomplete_recovery_claim_id(
            checkpoint, now=now
        )
        != fence.recovery_claim_id
    ):
        raise SessionRunFenced("Auxiliary recovery lost its invocation or recovery claim.")
    require_terminalization_plan_owner(checkpoint, fence.recovery_plan_ownership, now)
    if active_record is None:
        raise SessionModelCompletionStageConflict("Auxiliary recovery lost its active stage.")
    active = _reconstruct_active_model_completion_stage_record(active_record, session_id=session.id)
    if (
        stage.purpose != "auxiliary-inference"
        or stage.preparation_digest != fence.preparation_digest
        or active.stage_id != stage.stage_id
        or active.preparation_digest != stage.preparation_digest
    ):
        raise SessionModelCompletionStageConflict("Auxiliary recovery stage authority changed.")
    if dispatch_record is None:
        raise SessionModelCompletionStageConflict("Auxiliary recovery requires a dispatch receipt.")
    dispatch = _reconstruct_model_completion_stage_dispatch(
        dispatch_record,
        session_id=session.id,
        stage_id=stage.stage_id,
        storage_key=_model_completion_stage_dispatch_storage_key(stage.stage_id),
    )
    _validate_model_completion_stage_dispatch(dispatch, stage)


def _model_completion_stage_preparation_record(
    prepared: _PreparedModelCompletionStage,
    *,
    source_session: Session,
    prepared_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "stage_id": prepared.request.stage_id,
        "logical_step_id": prepared.request.logical_step_id,
        "dispatch_ordinal": prepared.request.dispatch_ordinal,
        "purpose": prepared.request.purpose,
        "intent": prepared.request.intent,
        "reservation_ids": list(prepared.request.reservation_ids),
        "request_digest": prepared.request_digest,
        "source_status": str(source_session.status),
        "source_run_epoch": source_session.run_epoch,
        "source_transcript_cursor": prepared.expected_transcript_cursor,
        "prepared_at": prepared_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_preparation")


def _model_completion_stage_terminal_record(
    prepared: _PreparedModelCompletionStageTerminal,
    *,
    stage: ModelCompletionStage,
    completed_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "stage_id": prepared.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "publication": prepared.publication.model_dump(mode="json"),
        "publication_material_digest": prepared.publication_material_digest,
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_terminal")


def _active_model_completion_stage_record(
    stage: ModelCompletionStage,
    *,
    activated_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "activated_at": activated_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "active_model_completion_stage")


def _validate_model_completion_stage_release(
    *,
    session: Session,
    active_record: dict[str, Any] | None,
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    release: ModelCompletionStageRelease,
) -> model_stage_contracts._ActiveModelCompletionStageRecord:
    """Validate exact in-flight stage ownership before a disposition clears it."""

    if type(release) is not ModelCompletionStageRelease:
        raise TypeError("model_completion_stage_release is invalid.")
    if active_record is None:
        raise SessionModelCompletionStageConflict("The model-completion stage is no longer active.")
    marker = _reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session.id,
    )
    if (
        marker.stage_id != release.stage_id
        or marker.preparation_digest != release.preparation_digest
    ):
        raise SessionModelCompletionStageConflict(
            "The active model-completion stage changed before external disposition."
        )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session.id,
        release.stage_id,
    )
    stage = _reconstruct_model_completion_stage(
        preparation_record,
        terminal_record,
        session_id=session.id,
        stage_id=release.stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
    )
    if (
        stage is None
        or stage.state != "in_flight"
        or stage.preparation_digest != release.preparation_digest
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion stage became terminal before external disposition."
        )
    return marker


def _model_completion_stage_abandonment_record(
    stage: ModelCompletionStage,
    *,
    active: ActiveModelCompletionStage,
    abandoned_at: datetime,
) -> dict[str, Any]:
    if stage.state != "in_flight" or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active in-flight model-completion stage can be abandoned."
        )
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_request_digest": stage.preparation_request_digest,
        "preparation_digest": stage.preparation_digest,
        "active_marker_digest": active.marker_digest,
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "abandoned_at": abandoned_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_abandonment")


def _model_completion_stage_provider_effect_id(stage: ModelCompletionStage) -> str:
    provider_start = stage.intent.get("provider_operation_start")
    if type(provider_start) is dict:
        idempotency_key = provider_start.get("idempotency_key")
        if type(idempotency_key) is str:
            return require_clean_nonblank(
                idempotency_key,
                "intent.provider_operation_start.idempotency_key",
            )
    request_fingerprint = stage.intent.get("request_fingerprint")
    if type(request_fingerprint) is not str:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage has no provider effect identity."
        )
    try:
        session_authority_rules._require_raw_sha256_digest(request_fingerprint)
    except ValueError as exc:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage request fingerprint is malformed."
        ) from exc
    return f"request:{request_fingerprint}"


def _model_completion_stage_execution_profile_fingerprint(
    stage: ModelCompletionStage,
) -> str | None:
    recovery_context = stage.intent.get("recovery_context")
    if recovery_context is None:
        return None
    if type(recovery_context) is not dict:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage recovery context is malformed."
        )
    fingerprint = recovery_context.get("execution_profile_fingerprint")
    if fingerprint is None:
        return None
    if type(fingerprint) is not str:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage execution profile identity is malformed."
        )
    try:
        session_authority_rules._require_raw_sha256_digest(fingerprint)
    except ValueError as exc:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage execution profile identity is malformed."
        ) from exc
    return fingerprint


def _model_completion_stage_dispatch_record(
    stage: ModelCompletionStage,
    *,
    dispatched_at: datetime,
) -> dict[str, Any]:
    if type(stage) is not ModelCompletionStage or stage.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "Only an exact in-flight model-completion stage can be dispatched."
        )
    model_attempt_id = stage.intent.get("model_attempt_id")
    interaction_id = stage.intent.get("interaction_id")
    request_fingerprint = stage.intent.get("request_fingerprint")
    if not all(
        type(value) is str for value in (model_attempt_id, interaction_id, request_fingerprint)
    ):
        raise SessionModelCompletionStageConflict(
            "Model-completion dispatch lost its exact interaction, attempt, or request identity."
        )
    payload = {
        "record_type": MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "model_attempt_id": model_attempt_id,
        "interaction_id": interaction_id,
        "request_fingerprint": request_fingerprint,
        "provider_effect_id": _model_completion_stage_provider_effect_id(stage),
        "execution_profile_fingerprint": (
            _model_completion_stage_execution_profile_fingerprint(stage)
        ),
        "preparation_digest": stage.preparation_digest,
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "dispatched_at": dispatched_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_dispatch")


def _reconstruct_model_completion_stage_dispatch(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> ModelCompletionStageDispatch:
    try:
        dispatch = ModelCompletionStageDispatch.model_validate(record)
        if (
            dispatch.record_type != MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE
            or dispatch.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION
            or dispatch.session_id != session_id
            or dispatch.stage_id != stage_id
            or storage_key != _model_completion_stage_dispatch_storage_key(stage_id)
        ):
            raise ValueError
        digest_payload = dispatch.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != dispatch.record_digest
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion dispatch record is malformed."
        ) from exc
    return dispatch


def _validate_model_completion_stage_dispatch(
    dispatch: ModelCompletionStageDispatch,
    stage: ModelCompletionStage,
) -> None:
    expected = _reconstruct_model_completion_stage_dispatch(
        _model_completion_stage_dispatch_record(
            stage,
            dispatched_at=dispatch.dispatched_at,
        ),
        session_id=stage.session_id,
        stage_id=stage.stage_id,
        storage_key=_model_completion_stage_dispatch_storage_key(stage.stage_id),
    )
    if dispatch != expected:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion dispatch record conflicts with its stage."
        )


def _validate_model_completion_stage_for_dispatch(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    current_transcript_cursor: int,
    stage: ModelCompletionStage,
    active_record: dict[str, Any] | None,
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    settlement_record: dict[str, Any] | None,
) -> None:
    """Validate the complete dispatch fence from one backend-atomic snapshot."""

    invocation_scope._assert_session_run_epoch(session.id, session)
    if (
        session.id != stage.session_id
        or session.status is not stage.source_status
        or session.run_epoch != stage.source_run_epoch
    ):
        raise SessionRunFenced("Model-completion dispatch lost its exact session epoch.")
    if current_transcript_cursor != stage.source_transcript_cursor:
        raise ValueError(
            "Session source transcript cursor is stale: expected "
            f"{stage.source_transcript_cursor}, current {current_transcript_cursor}."
        )
    active = _reconstruct_active_model_completion_stage(
        active_record,
        preparation_record,
        terminal_record,
        session_id=session.id,
    )
    if active is None or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active model-completion stage can cross the dispatch fence."
        )
    from cayu.runtime._model_failover_stage import validate_model_failover_dispatch

    validate_model_failover_dispatch(session=session, checkpoint=checkpoint, stage=stage)
    settlement_storage_key = _model_completion_stage_settlement_storage_key(stage.stage_id)
    _reject_settled_model_completion_stage(
        settlement_record,
        session_id=session.id,
        stage_id=stage.stage_id,
        settlement_storage_key=settlement_storage_key,
    )


def model_completion_stage_settlement_request(
    stage: ModelCompletionStage,
    *,
    interaction_id: str,
    disposition: ModelCompletionStageDisposition,
    reason_code: str,
    execution_profile_fingerprint: str | None,
    settlement_run_epoch: int,
    settled_reservation_ids: tuple[str, ...],
    superseding_stage_id: str | None = None,
) -> ModelCompletionStageSettlementRequest:
    """Bind one requested disposition to every durable model-effect identity."""

    if type(stage) is not ModelCompletionStage:
        raise TypeError("stage must be a ModelCompletionStage.")
    if stage.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "Only an in-flight model-completion stage can be settled."
        )
    model_attempt_id = stage.intent.get("model_attempt_id")
    request_fingerprint = stage.intent.get("request_fingerprint")
    staged_interaction_id = stage.intent.get("interaction_id")
    if type(model_attempt_id) is not str or type(request_fingerprint) is not str:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage lost its exact attempt or request identity."
        )
    if staged_interaction_id != interaction_id:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage belongs to a different interaction."
        )
    staged_profile = _model_completion_stage_execution_profile_fingerprint(stage)
    if staged_profile != execution_profile_fingerprint:
        raise SessionModelCompletionStageConflict(
            "Model-completion stage belongs to a different execution profile."
        )
    settlement_run_epoch = session_authority_rules._validate_required_runtime_fence(
        settlement_run_epoch,
        "settlement_run_epoch",
    )
    if settlement_run_epoch < stage.source_run_epoch:
        raise SessionRunFenced(
            "Model-completion settlement run epoch cannot precede its source epoch: "
            f"source {stage.source_run_epoch}, settlement {settlement_run_epoch}."
        )
    return ModelCompletionStageSettlementRequest(
        stage_id=stage.stage_id,
        logical_step_id=stage.logical_step_id,
        model_attempt_id=model_attempt_id,
        interaction_id=interaction_id,
        request_fingerprint=request_fingerprint,
        provider_effect_id=_model_completion_stage_provider_effect_id(stage),
        execution_profile_fingerprint=execution_profile_fingerprint,
        preparation_digest=stage.preparation_digest,
        settlement_run_epoch=settlement_run_epoch,
        settled_reservation_ids=settled_reservation_ids,
        disposition=disposition,
        reason_code=reason_code,
        superseding_stage_id=superseding_stage_id,
    )


def _model_completion_retry_settlement_request(
    active: ActiveModelCompletionStage | None,
    prepared: _PreparedModelCompletionStage,
) -> ModelCompletionStageSettlementRequest | None:
    """Create exact supersession evidence for current-contract retry stages.

    Older direct store callers could use opaque intents without an interaction,
    attempt, or request identity. Preserve those stored contracts, while every
    runtime-authored stage now carries enough identity to settle durably.
    """

    if active is None:
        return None
    stage = active.stage
    if not all(
        type(stage.intent.get(field)) is str
        for field in ("interaction_id", "model_attempt_id", "request_fingerprint")
    ):
        return None
    return model_completion_stage_settlement_request(
        stage,
        interaction_id=stage.intent["interaction_id"],
        disposition=ModelCompletionStageDisposition.SUPERSEDED,
        reason_code="retry_superseded",
        execution_profile_fingerprint=(
            _model_completion_stage_execution_profile_fingerprint(stage)
        ),
        settlement_run_epoch=prepared.expected_run_epoch,
        settled_reservation_ids=stage.reservation_ids,
        superseding_stage_id=prepared.request.stage_id,
    )


def _model_completion_stage_settlement_record(
    stage: ModelCompletionStage,
    *,
    request: ModelCompletionStageSettlementRequest,
    settled_at: datetime,
) -> dict[str, Any]:
    expected = model_completion_stage_settlement_request(
        stage,
        interaction_id=request.interaction_id,
        disposition=request.disposition,
        reason_code=request.reason_code,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        settlement_run_epoch=request.settlement_run_epoch,
        settled_reservation_ids=request.settled_reservation_ids,
        superseding_stage_id=request.superseding_stage_id,
    )
    if expected != request:
        raise SessionModelCompletionStageConflict(
            "Model-completion settlement authority conflicts with its active stage."
        )
    payload = {
        "record_type": MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        **request.model_dump(mode="json"),
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "settled_at": settled_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_settlement")


def _reconstruct_model_completion_stage_settlement(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> ModelCompletionStageSettlement:
    try:
        settlement = ModelCompletionStageSettlement.model_validate(record)
        if (
            settlement.record_type != MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE
            or settlement.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION
            or settlement.session_id != session_id
            or settlement.stage_id != stage_id
            or storage_key != _model_completion_stage_settlement_storage_key(stage_id)
        ):
            raise ValueError
        digest_payload = settlement.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != settlement.record_digest
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion settlement record is malformed."
        ) from exc
    return settlement


def _validate_model_completion_stage_for_settlement(
    *,
    session: Session,
    stage: ModelCompletionStage,
    active: ActiveModelCompletionStage | None,
    request: ModelCompletionStageSettlementRequest,
    settlement_record: dict[str, Any] | None,
    winner_exists: bool,
    receipt_exists: bool,
) -> None:
    if stage.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "A completed model-completion stage cannot receive a failure disposition."
        )
    if active is None or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active model-completion stage can be settled."
        )
    if winner_exists or receipt_exists:
        raise SessionModelCompletionStageConflict(
            "A published model-completion stage cannot receive a failure disposition."
        )
    expected = model_completion_stage_settlement_request(
        stage,
        interaction_id=request.interaction_id,
        disposition=request.disposition,
        reason_code=request.reason_code,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        settlement_run_epoch=request.settlement_run_epoch,
        settled_reservation_ids=request.settled_reservation_ids,
        superseding_stage_id=request.superseding_stage_id,
    )
    if expected != request:
        raise SessionModelCompletionStageConflict(
            "Model-completion settlement authority conflicts with its active stage."
        )
    if request.settled_reservation_ids != stage.reservation_ids:
        raise SessionModelCompletionStageConflict(
            "Model-completion settlement does not cover every linked budget reservation."
        )
    if session.run_epoch != request.settlement_run_epoch:
        raise SessionRunFenced(
            "Model-completion settlement run epoch is stale: expected "
            f"{request.settlement_run_epoch}, current {session.run_epoch}."
        )
    if settlement_record is not None:
        prior = _reconstruct_model_completion_stage_settlement(
            settlement_record,
            session_id=session.id,
            stage_id=stage.stage_id,
            storage_key=_model_completion_stage_settlement_storage_key(stage.stage_id),
        )
        raise SessionModelCompletionStageConflict(
            "The active model-completion stage already has terminal settlement: "
            f"{prior.disposition.value}."
        )


def _model_completion_stage_winner_record(
    stage: ModelCompletionStage,
    *,
    receipt: RuntimePublicationReceipt,
) -> dict[str, Any]:
    if (
        stage.state != "completed"
        or stage.completion_digest is None
        or stage.publication_material_digest is None
    ):
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_WINNER_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "completion_digest": stage.completion_digest,
        "publication_material_digest": stage.publication_material_digest,
        "receipt_request_digest": receipt.request_digest,
        "publication_digest": receipt.publication_digest,
        "published_at": receipt.published_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = session_authority_rules._canonical_runtime_publication_digest(
        payload
    )
    return copy_durable_json_object(payload, "model_completion_stage_winner")


def _reconstruct_active_model_completion_stage_record(
    record: dict[str, Any],
    *,
    session_id: str,
    storage_key: str = MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
) -> model_stage_contracts._ActiveModelCompletionStageRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(model_stage_contracts._ActiveModelCompletionStageRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "source_status",
            "activated_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        for field in (
            "schema_version",
            "dispatch_ordinal",
            "source_run_epoch",
            "source_transcript_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        session_authority_rules._require_raw_sha256_digest(record["preparation_digest"])
        session_authority_rules._require_raw_sha256_digest(record["record_digest"])
        reconstructed = model_stage_contracts._ActiveModelCompletionStageRecord.model_validate(
            record
        )
        if reconstructed.record_type != MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id:
            raise ValueError
        if storage_key != MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY:
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != reconstructed.record_digest
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable active model-completion marker is malformed."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_abandonment(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> ModelCompletionStageAbandonment:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(
            model_stage_contracts._ModelCompletionStageAbandonmentRecord.model_fields
        ):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_request_digest",
            "preparation_digest",
            "active_marker_digest",
            "source_status",
            "abandoned_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        for field in (
            "schema_version",
            "dispatch_ordinal",
            "source_run_epoch",
            "source_transcript_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        for field in (
            "preparation_request_digest",
            "preparation_digest",
            "active_marker_digest",
            "record_digest",
        ):
            session_authority_rules._require_raw_sha256_digest(record[field])
        reconstructed = model_stage_contracts._ModelCompletionStageAbandonmentRecord.model_validate(
            record
        )
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        if storage_key != _model_completion_stage_abandonment_storage_key(stage_id):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != reconstructed.record_digest
        ):
            raise ValueError
        return ModelCompletionStageAbandonment(
            session_id=reconstructed.session_id,
            stage_id=reconstructed.stage_id,
            logical_step_id=reconstructed.logical_step_id,
            dispatch_ordinal=reconstructed.dispatch_ordinal,
            purpose=reconstructed.purpose,
            preparation_request_digest=reconstructed.preparation_request_digest,
            preparation_digest=reconstructed.preparation_digest,
            active_marker_digest=reconstructed.active_marker_digest,
            source_status=reconstructed.source_status,
            source_run_epoch=reconstructed.source_run_epoch,
            source_transcript_cursor=reconstructed.source_transcript_cursor,
            abandoned_at=reconstructed.abandoned_at,
            abandonment_digest=reconstructed.record_digest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion abandonment is malformed or conflicts with its key."
        ) from exc


def _reconstruct_model_completion_stage_winner_record(
    record: dict[str, Any],
    *,
    session_id: str,
    logical_step_id: str,
    storage_key: str,
) -> model_stage_contracts._ModelCompletionStageWinnerRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(model_stage_contracts._ModelCompletionStageWinnerRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "completion_digest",
            "publication_material_digest",
            "receipt_request_digest",
            "publication_digest",
            "published_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["dispatch_ordinal"]) is not int:
            raise TypeError
        for field in (
            "preparation_digest",
            "completion_digest",
            "publication_material_digest",
            "receipt_request_digest",
            "publication_digest",
            "record_digest",
        ):
            session_authority_rules._require_raw_sha256_digest(record[field])
        reconstructed = model_stage_contracts._ModelCompletionStageWinnerRecord.model_validate(
            record
        )
        if reconstructed.record_type != MODEL_COMPLETION_WINNER_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if (
            reconstructed.session_id != session_id
            or reconstructed.logical_step_id != logical_step_id
            or storage_key != _model_completion_stage_winner_storage_key(logical_step_id)
        ):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != reconstructed.record_digest
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion winner is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_preparation(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> model_stage_contracts._ModelCompletionStagePreparationRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(
            model_stage_contracts._ModelCompletionStagePreparationRecord.model_fields
        ):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "request_digest",
            "source_status",
            "prepared_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["dispatch_ordinal"]) is not int:
            raise TypeError
        if type(record["intent"]) is not dict or type(record["reservation_ids"]) is not list:
            raise TypeError
        if type(record["source_run_epoch"]) is not int:
            raise TypeError
        if type(record["source_transcript_cursor"]) is not int:
            raise TypeError
        session_authority_rules._require_raw_sha256_digest(record["request_digest"])
        session_authority_rules._require_raw_sha256_digest(record["record_digest"])
        reconstructed = model_stage_contracts._ModelCompletionStagePreparationRecord.model_validate(
            record
        )
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        _, _, expected_key, _ = _model_completion_stage_storage_identity(session_id, stage_id)
        if storage_key != expected_key:
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != reconstructed.record_digest
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion preparation is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_terminal(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> model_stage_contracts._ModelCompletionStageTerminalRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(
            model_stage_contracts._ModelCompletionStageTerminalRecord.model_fields
        ):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "publication_material_digest",
            "completed_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if (
            type(record["schema_version"]) is not int
            or type(record["dispatch_ordinal"]) is not int
            or type(record["publication"]) is not dict
        ):
            raise TypeError
        session_authority_rules._require_raw_sha256_digest(record["preparation_digest"])
        session_authority_rules._require_raw_sha256_digest(record["publication_material_digest"])
        session_authority_rules._require_raw_sha256_digest(record["record_digest"])
        publication_record = record["publication"]
        required_publication_fields = set(RuntimePublicationRequest.model_fields) - {
            "interaction_id"
        }
        if not required_publication_fields.issubset(publication_record) or (
            set(publication_record) - set(RuntimePublicationRequest.model_fields)
        ):
            raise ValueError
        raw_transcript = publication_record["transcript_messages"]
        raw_events = publication_record["events"]
        if type(raw_transcript) is not list or type(raw_events) is not list:
            raise TypeError
        publication = RuntimePublicationRequest(
            publication_id=publication_record["publication_id"],
            kind=publication_record["kind"],
            interaction_id=publication_record.get("interaction_id"),
            intent=publication_record["intent"],
            mutation=publication_record["mutation"],
            transcript_messages=tuple(
                Message.model_validate(message) for message in raw_transcript
            ),
            events=tuple(Event.model_validate(event) for event in raw_events),
            operation_record_mutations=publication_record["operation_record_mutations"],
            referenced_events=publication_record["referenced_events"],
        )
        reconstructed = model_stage_contracts._ModelCompletionStageTerminalRecord.model_validate(
            {**record, "publication": publication}
        )
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        _, _, _, expected_key = _model_completion_stage_storage_identity(session_id, stage_id)
        if storage_key != expected_key:
            raise ValueError
        publication_payload = reconstructed.publication.model_dump(mode="json")
        if (
            session_authority_rules._canonical_runtime_publication_digest(publication_payload)
            != reconstructed.publication_material_digest
        ):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != reconstructed.record_digest
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable terminal model completion is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage(
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    *,
    session_id: str,
    stage_id: str,
    preparation_storage_key: str,
    terminal_storage_key: str,
) -> ModelCompletionStage | None:
    if preparation_record is None:
        if terminal_record is not None:
            raise SessionModelCompletionStageConflict(
                "A terminal model completion exists without its immutable preparation."
            )
        return None
    preparation = _reconstruct_model_completion_stage_preparation(
        preparation_record,
        session_id=session_id,
        stage_id=stage_id,
        storage_key=preparation_storage_key,
    )
    terminal = None
    if terminal_record is not None:
        terminal = _reconstruct_model_completion_stage_terminal(
            terminal_record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=terminal_storage_key,
        )
        if terminal.preparation_digest != preparation.record_digest:
            raise SessionModelCompletionStageConflict(
                "The terminal model completion does not match its immutable preparation."
            )
        if (
            terminal.logical_step_id != preparation.logical_step_id
            or terminal.dispatch_ordinal != preparation.dispatch_ordinal
            or terminal.purpose != preparation.purpose
        ):
            raise SessionModelCompletionStageConflict(
                "The terminal model completion identity conflicts with its preparation."
            )
        if not session_authority_rules._runtime_publication_json_equal(
            terminal.publication.intent, preparation.intent
        ):
            raise SessionModelCompletionStageConflict(
                "The terminal model completion intent conflicts with its preparation."
            )
        try:
            _validate_model_completion_stage_publication(
                terminal.publication,
                session_id=session_id,
                stage=preparation,
            )
        except (TypeError, ValueError) as exc:
            raise SessionModelCompletionStageConflict(
                "The terminal model completion publication conflicts with its preparation."
            ) from exc

    try:
        return ModelCompletionStage(
            session_id=session_id,
            stage_id=stage_id,
            logical_step_id=preparation.logical_step_id,
            dispatch_ordinal=preparation.dispatch_ordinal,
            purpose=preparation.purpose,
            state="in_flight" if terminal is None else "completed",
            intent=preparation.intent,
            reservation_ids=preparation.reservation_ids,
            preparation_request_digest=preparation.request_digest,
            preparation_digest=preparation.record_digest,
            source_status=preparation.source_status,
            source_run_epoch=preparation.source_run_epoch,
            source_transcript_cursor=preparation.source_transcript_cursor,
            prepared_at=preparation.prepared_at,
            publication=None if terminal is None else terminal.publication,
            publication_material_digest=(
                None if terminal is None else terminal.publication_material_digest
            ),
            completion_digest=None if terminal is None else terminal.record_digest,
            completed_at=None if terminal is None else terminal.completed_at,
        )
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage is malformed."
        ) from exc


def _reconstruct_active_model_completion_stage(
    active_record: dict[str, Any] | None,
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    *,
    session_id: str,
) -> ActiveModelCompletionStage | None:
    if active_record is None:
        if preparation_record is not None or terminal_record is not None:
            raise SessionModelCompletionStageConflict(
                "Active model-completion material exists without its marker."
            )
        return None
    marker = _reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session_id,
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        marker.stage_id,
    )
    stage = _reconstruct_model_completion_stage(
        preparation_record,
        terminal_record,
        session_id=session_id,
        stage_id=marker.stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
    )
    if stage is None:
        raise SessionModelCompletionStageConflict(
            "The active model-completion marker references a missing stage."
        )
    if (
        stage.logical_step_id != marker.logical_step_id
        or stage.dispatch_ordinal != marker.dispatch_ordinal
        or stage.purpose != marker.purpose
        or stage.preparation_digest != marker.preparation_digest
        or stage.source_status != marker.source_status
        or stage.source_run_epoch != marker.source_run_epoch
        or stage.source_transcript_cursor != marker.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The active model-completion marker conflicts with its stage."
        )
    try:
        return ActiveModelCompletionStage(
            stage=stage,
            activated_at=marker.activated_at,
            marker_digest=marker.record_digest,
        )
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The active model-completion snapshot is malformed."
        ) from exc


def _reconstruct_active_model_completion_stage_from_records(
    records: Mapping[str, dict[str, Any]],
    *,
    session_id: str,
) -> ActiveModelCompletionStage | None:
    active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
    if active_record is None:
        return None
    marker = _reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session_id,
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        marker.stage_id,
    )
    return _reconstruct_active_model_completion_stage(
        active_record,
        records.get(preparation_key),
        records.get(terminal_key),
        session_id=session_id,
    )


def _validate_model_completion_active_marker_for_preparation(
    active: ActiveModelCompletionStage | None,
    prepared: _PreparedModelCompletionStage,
    *,
    source_status: SessionStatus,
) -> None:
    if active is None:
        return
    current = active.stage
    request = prepared.request
    if current.state == "completed":
        raise SessionModelCompletionStageConflict(
            "A terminal model-completion stage must be published before another dispatch."
        )
    if (
        current.logical_step_id != request.logical_step_id
        or current.purpose != request.purpose
        or current.source_status != source_status
        or current.source_run_epoch != prepared.expected_run_epoch
        or current.source_transcript_cursor != prepared.expected_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "A different logical model completion is already active for the session."
        )
    if request.dispatch_ordinal <= current.dispatch_ordinal:
        raise SessionModelCompletionStageConflict(
            "A model-completion retry must advance dispatch_ordinal monotonically."
        )


def _validate_model_completion_preparation_replay_state(
    stage: ModelCompletionStage,
    *,
    active: ActiveModelCompletionStage | None,
    winner_exists: bool,
    receipt_exists: bool,
) -> None:
    if winner_exists or receipt_exists:
        raise SessionModelCompletionStageConflict(
            "A prepared model-completion stage already has durable publication state."
        )
    if active is None:
        raise SessionModelCompletionStageConflict(
            "A prepared model-completion stage no longer has an active marker."
        )
    active_stage = active.stage
    if (
        active_stage.stage_id != stage.stage_id
        or active_stage.logical_step_id != stage.logical_step_id
        or active_stage.dispatch_ordinal != stage.dispatch_ordinal
        or active_stage.purpose != stage.purpose
        or active_stage.preparation_digest != stage.preparation_digest
        or active_stage.source_status != stage.source_status
        or active_stage.source_run_epoch != stage.source_run_epoch
        or active_stage.source_transcript_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The prepared model-completion stage is no longer the exact active dispatch."
        )


def _validate_model_completion_stage_repreparation(
    abandonment_record: dict[str, Any] | None,
    prepared: _PreparedModelCompletionStage,
    *,
    source_status: SessionStatus,
) -> None:
    if abandonment_record is None:
        return
    abandonment = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.request.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    request = prepared.request
    if (
        abandonment.logical_step_id != request.logical_step_id
        or abandonment.dispatch_ordinal != request.dispatch_ordinal
        or abandonment.purpose != request.purpose
        or abandonment.preparation_request_digest != prepared.request_digest
        or abandonment.source_status != source_status
        or abandonment.source_run_epoch != prepared.expected_run_epoch
        or abandonment.source_transcript_cursor != prepared.expected_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "An abandoned model-completion stage id can only be reused for its exact "
            "request and durable source boundary."
        )


def _replay_model_completion_stage_abandonment(
    prepared: _PreparedModelCompletionStageAbandonment,
    abandonment_record: dict[str, Any] | None,
) -> ModelCompletionStageAbandonmentResult:
    if abandonment_record is None:
        raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
    abandonment = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    if (
        abandonment.preparation_digest != prepared.preparation_digest
        or abandonment.source_run_epoch != prepared.stage_source_run_epoch
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion abandonment conflicts with the durable tombstone."
        )
    return ModelCompletionStageAbandonmentResult(
        abandonment=abandonment,
        replayed=True,
    )


def _validate_model_completion_stage_for_abandonment(
    *,
    session: Session,
    stage: ModelCompletionStage,
    active: ActiveModelCompletionStage | None,
    prepared: _PreparedModelCompletionStageAbandonment,
    abandonment_record: dict[str, Any] | None,
    dispatch_exists: bool,
    winner_exists: bool,
    receipt_exists: bool,
) -> None:
    if stage.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "A terminal model-completion stage cannot be abandoned."
        )
    if stage.preparation_digest != prepared.preparation_digest:
        raise SessionModelCompletionStageConflict(
            "The model-completion abandonment preparation digest is stale."
        )
    if stage.source_run_epoch != prepared.stage_source_run_epoch:
        raise SessionRunFenced(
            "Model-completion stage run epoch is stale: expected source "
            f"{prepared.stage_source_run_epoch}, prepared {stage.source_run_epoch}."
        )
    invocation_scope._assert_session_run_epoch(prepared.session_id, session)
    if session.run_epoch != prepared.expected_run_epoch:
        raise SessionRunFenced(
            "Session source run epoch is stale: expected "
            f"{prepared.expected_run_epoch}, current {session.run_epoch}."
        )
    if active is None or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active model-completion stage can be abandoned."
        )
    if dispatch_exists:
        raise SessionModelCompletionStageConflict(
            "A dispatched model-completion stage cannot be abandoned."
        )
    if winner_exists or receipt_exists:
        raise SessionModelCompletionStageConflict(
            "A model-completion stage with durable publication state cannot be abandoned."
        )
    if abandonment_record is None:
        return
    prior = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    if (
        prior.logical_step_id != stage.logical_step_id
        or prior.dispatch_ordinal != stage.dispatch_ordinal
        or prior.purpose != stage.purpose
        or prior.preparation_request_digest != stage.preparation_request_digest
        or prior.source_status != stage.source_status
        or prior.source_run_epoch != stage.source_run_epoch
        or prior.source_transcript_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The active model-completion stage conflicts with its prior abandonment."
        )
    if prior.preparation_digest == stage.preparation_digest:
        raise SessionModelCompletionStageConflict(
            "An abandoned model-completion preparation was unexpectedly restored."
        )


def _validate_model_completion_promotion_replay_active_marker(
    active: ActiveModelCompletionStage | None,
    stage: ModelCompletionStage,
) -> None:
    if active is not None and active.stage.logical_step_id == stage.logical_step_id:
        raise SessionModelCompletionStageConflict(
            "A published model-completion winner still has an active marker."
        )


def _model_completion_terminal_advances_last_activity(
    active_record: dict[str, Any] | None,
    *,
    stage: ModelCompletionStage,
    current_run_epoch: int,
) -> bool:
    if active_record is None:
        return False
    try:
        marker = _reconstruct_active_model_completion_stage_record(
            active_record,
            session_id=stage.session_id,
        )
    except SessionModelCompletionStageConflict:
        # Terminal provider evidence is append-only even when unrelated active
        # state is corrupt. A malformed marker can never refresh run liveness.
        return False
    return (
        stage.source_run_epoch == current_run_epoch
        and marker.stage_id == stage.stage_id
        and marker.logical_step_id == stage.logical_step_id
        and marker.dispatch_ordinal == stage.dispatch_ordinal
        and marker.purpose == stage.purpose
        and marker.preparation_digest == stage.preparation_digest
        and marker.source_status == stage.source_status
        and marker.source_run_epoch == stage.source_run_epoch
        and marker.source_transcript_cursor == stage.source_transcript_cursor
    )


def _validate_model_completion_active_marker_for_promotion(
    active: ActiveModelCompletionStage | None,
    stage: ModelCompletionStage,
) -> None:
    if active is None:
        raise SessionModelCompletionStageConflict(
            "The model-completion stage is no longer active and cannot be promoted."
        )
    if (
        active.stage.stage_id != stage.stage_id
        or active.stage.logical_step_id != stage.logical_step_id
        or active.stage.dispatch_ordinal != stage.dispatch_ordinal
        or active.stage.purpose != stage.purpose
        or active.stage.preparation_digest != stage.preparation_digest
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion stage was superseded before promotion."
        )


def _validate_model_completion_stage_winner(
    winner_record: dict[str, Any] | None,
    *,
    stage: ModelCompletionStage,
    receipt: RuntimePublicationReceipt,
    winner_storage_key: str,
) -> model_stage_contracts._ModelCompletionStageWinnerRecord:
    if winner_record is None:
        raise SessionModelCompletionStageConflict(
            "A model-completion receipt exists without its durable winner claim."
        )
    winner = _reconstruct_model_completion_stage_winner_record(
        winner_record,
        session_id=stage.session_id,
        logical_step_id=stage.logical_step_id,
        storage_key=winner_storage_key,
    )
    if (
        winner.stage_id != stage.stage_id
        or winner.dispatch_ordinal != stage.dispatch_ordinal
        or winner.purpose != stage.purpose
        or winner.preparation_digest != stage.preparation_digest
        or winner.completion_digest != stage.completion_digest
        or winner.publication_material_digest != stage.publication_material_digest
        or winner.receipt_request_digest != receipt.request_digest
        or winner.publication_digest != receipt.publication_digest
        or winner.published_at != receipt.published_at
    ):
        raise SessionModelCompletionStageConflict(
            "The runtime publication receipt belongs to a different model-completion winner."
        )
    return winner


def _validate_model_completion_stage_preparation_replay(
    stage: ModelCompletionStage,
    prepared: _PreparedModelCompletionStage,
) -> None:
    if stage.preparation_request_digest != prepared.request_digest:
        raise SessionModelCompletionStageConflict(
            "The model-completion stage id was already prepared for a different request."
        )


def _validate_model_completion_stage_terminal_replay(
    stage: ModelCompletionStage,
    prepared: _PreparedModelCompletionStageTerminal,
) -> None:
    if (
        stage.state != "completed"
        or stage.publication_material_digest != prepared.publication_material_digest
        or (
            prepared.recovery_fence is not None
            and stage.preparation_digest != prepared.recovery_fence.preparation_digest
        )
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion stage already has different terminal material."
        )


def _reject_settled_model_completion_stage(
    settlement_record: dict[str, Any] | None,
    *,
    session_id: str,
    stage_id: str,
    settlement_storage_key: str,
) -> None:
    if settlement_record is None:
        return
    settlement = _reconstruct_model_completion_stage_settlement(
        settlement_record,
        session_id=session_id,
        stage_id=stage_id,
        storage_key=settlement_storage_key,
    )
    raise SessionModelCompletionStageConflict(
        "The model-completion stage is already terminally settled as "
        f"{settlement.disposition.value}."
    )


def _model_completion_stage_promotion_statuses(stage: ModelCompletionStage) -> set[SessionStatus]:
    statuses = {stage.source_status, SessionStatus.INTERRUPTING, SessionStatus.FAILED}
    if stage.purpose == "auxiliary-inference":
        # Auxiliary publications are accounting only: they cannot reopen the
        # conversation or alter the parent transcript/checkpoint.
        statuses.update({SessionStatus.COMPLETED, SessionStatus.INTERRUPTED})
    return statuses


def _prepare_model_completion_stage_promotion(
    stage: ModelCompletionStage,
    *,
    expected_run_epoch: int,
) -> publication_rules._PreparedRuntimePublication:
    if stage.state != "completed" or stage.publication is None:
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    return publication_rules._prepare_runtime_publication(
        stage.session_id,
        stage.publication,
        expected_statuses=_model_completion_stage_promotion_statuses(stage),
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=(
            None if stage.purpose == "auxiliary-inference" else stage.source_transcript_cursor
        ),
    )


def _replay_promoted_model_completion_stage(
    *,
    session: Session,
    stage: ModelCompletionStage,
    receipt_record: dict[str, Any],
    winner_record: dict[str, Any] | None,
) -> RuntimePublicationResult:
    if stage.state != "completed" or stage.publication is None:
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    storage_key = publication_contracts._runtime_publication_storage_key(stage.logical_step_id)
    receipt = publication_rules._reconstruct_runtime_publication_receipt(
        receipt_record,
        storage_key=storage_key,
        session_id=stage.session_id,
        publication_id=stage.logical_step_id,
    )
    if receipt.source_status not in _model_completion_stage_promotion_statuses(stage) or (
        stage.purpose != "auxiliary-inference"
        and receipt.transcript_start_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The runtime publication receipt conflicts with its model-completion stage."
        )
    _validate_model_completion_stage_winner(
        winner_record,
        stage=stage,
        receipt=receipt,
        winner_storage_key=_model_completion_stage_winner_storage_key(stage.logical_step_id),
    )
    prepared = publication_rules._prepare_runtime_publication(
        stage.session_id,
        stage.publication,
        expected_statuses={stage.source_status},
        expected_run_epoch=receipt.source_run_epoch,
        expected_transcript_cursor=(
            None if stage.purpose == "auxiliary-inference" else stage.source_transcript_cursor
        ),
    )
    receipt = publication_rules._reconstruct_runtime_publication_receipt(
        receipt_record,
        storage_key=prepared.storage_key,
        session_id=stage.session_id,
        publication_id=stage.logical_step_id,
        request_digest=prepared.request_digest,
    )
    publication_rules._validate_runtime_publication_replay_receipt(receipt, prepared)
    return RuntimePublicationResult(
        session=session.model_copy(deep=True),
        receipt=receipt,
        replayed=True,
    )


def require_terminalization_plan_owner(
    checkpoint: dict[str, Any] | None,
    expected: DurableOperationOwnership | None,
    now: datetime,
) -> None:
    marker = None if checkpoint is None else checkpoint.get("recovery_plan_execution")
    if marker is None:
        if expected is not None:
            raise SessionRunFenced("Model terminalization lost its plan owner.")
        return
    if type(marker) is not dict or expected is None:
        raise SessionRunFenced("Model terminalization is owned by another recovery plan.")
    current = DurableOperationOwnership.model_validate(marker.get("ownership"))
    if (
        current.claim_id != expected.claim_id
        or current.generation != expected.generation
        or current.owner_id != expected.owner_id
        or current.operation_id != expected.operation_id
        or current.state.value != "active"
        or current.lease_expires_at is None
        or current.lease_expires_at <= now
    ):
        raise SessionRunFenced("Model terminalization recovery plan expired or changed.")

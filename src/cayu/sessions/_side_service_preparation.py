"""Plan two native receiving preparations before either is applied.

Only SessionStore's native transaction owns this pair. CollaborationStore permit
registration remains a later, separate durable handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_durable_json_object
from cayu.collaboration._preparation import prepare_contract
from cayu.sessions._session_continuation import (
    CONTINUATION_NAMESPACE_KEY,
    ContinuationConflict,
    continuation_operation_key,
)
from cayu.sessions._session_continuation_scope import service_publication_scope
from cayu.sessions._temporary_continuation import (
    TemporaryServicePreparation,
    TemporaryServiceRecord,
    temporary_service_key,
)
from cayu.sessions._temporary_service_target import TemporaryServiceTarget, target_service_key
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from datetime import datetime

    from cayu.sessions.base import SessionOperationPublication
    from cayu.sessions.records import Session


@dataclass(frozen=True)
class SidePreparationSnapshot:
    session: Session
    checkpoint: dict[str, Any] | None
    records: dict[str, dict[str, Any]]


def prepare_selection(value: TemporaryServicePreparation) -> TemporaryServicePreparation:
    prepared = prepare_contract(TemporaryServicePreparation, value, redactor=SecretRedactor())
    intent = prepared.dispatch.intent
    if intent.mode != "side_session" or intent.target.object_id == intent.ticket.session_id:
        raise ContinuationConflict("Joint preparation requires two distinct existing sessions.")
    return prepared


def preparation_record_keys(prepared: TemporaryServicePreparation) -> dict[str, tuple[str, ...]]:
    intent = prepared.dispatch.intent
    return {
        intent.ticket.session_id: (
            continuation_operation_key(intent.ticket),
            temporary_service_key(intent.operation),
        ),
        intent.target.object_id: (target_service_key(intent.operation),),
    }


def plan_preparation(
    prepared: TemporaryServicePreparation,
    snapshots: dict[str, SidePreparationSnapshot],
    now: datetime,
) -> dict[str, SessionOperationPublication]:
    from cayu.sessions._checkpoint_preservation import (
        _checkpoint_transform_result_preserving_completion_result_event_publications,
        _copy_checkpoint_for_transform,
        _invocation_lifecycle_authority_read_scope,
    )
    from cayu.sessions._session_continuation_store import ROOT_KEY
    from cayu.sessions._temporary_continuation_store import publish_service_record
    from cayu.sessions._temporary_service_target import publish_target_record
    from cayu.sessions.base import (
        SessionOperationPublication,
        _validate_session_operation_record_keys,
    )

    prepared = prepare_selection(prepared)
    intent = prepared.dispatch.intent
    source = snapshots[intent.ticket.session_id]
    target = snapshots[intent.target.object_id]
    source_key = continuation_operation_key(intent.ticket)
    child_key = temporary_service_key(intent.operation)
    target_key = target_service_key(intent.operation)
    child = TemporaryServiceRecord(admission=prepared, state="prepared")
    receiving = TemporaryServiceTarget(service=child)
    prior_source = source.records.get(child_key)
    prior_target = target.records.get(target_key)
    if (prior_source is None) != (prior_target is None):
        raise ContinuationConflict("Joint preparation has incomplete native ownership evidence.")
    if prior_source is not None and (
        prior_source != child.model_dump(mode="json")
        or prior_target != receiving.model_dump(mode="json")
    ):
        raise ContinuationConflict("Joint preparation conflicts with an existing decision.")
    result = {}
    for snapshot, parent_key, operation_key, transform in (
        (
            source,
            source_key,
            child_key,
            lambda checkpoint: publish_service_record(
                source.session,
                checkpoint,
                source.records.get(source_key),
                None if prior_source is None else child,
                child,
                now,
            ),
        ),
        (
            target,
            target_key,
            CONTINUATION_NAMESPACE_KEY,
            lambda checkpoint: publish_target_record(
                target.session,
                checkpoint,
                prior_target,
                None if prior_target is None else receiving,
                receiving,
                now,
            ),
        ),
    ):
        with (
            service_publication_scope(parent_key, operation_key),
            _invocation_lifecycle_authority_read_scope(),
        ):
            publication = transform(
                _copy_checkpoint_for_transform(snapshot.checkpoint, session_id=snapshot.session.id)
            )
            checkpoint = (
                _checkpoint_transform_result_preserving_completion_result_event_publications(
                    snapshot.checkpoint, publication.checkpoint, session_id=snapshot.session.id
                )
            )
            records = copy_durable_json_object(publication.operation_records, "operation_records")
            _validate_session_operation_record_keys(records)
            if {k: v for k, v in checkpoint.items() if k != ROOT_KEY} != {
                k: v for k, v in (snapshot.checkpoint or {}).items() if k != ROOT_KEY
            }:
                raise ContinuationConflict("Joint preparation changed another checkpoint owner.")
            result[snapshot.session.id] = SessionOperationPublication(
                checkpoint=checkpoint, operation_records=records
            )
    # Both index/record comparisons were checked above, but exact replay must not
    # refresh either session or turn a retry into a new preparation.
    return {} if prior_source is not None else result

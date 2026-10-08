"""Schema-aware checkpoint callbacks shared by the runtime store adapter.

Compose the existing decoder and preservation rules within the caller's native
transaction. Runtime dispatch and store orchestration keep their own boundaries.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from cayu.sessions._checkpoint_preservation import (
    _copy_checkpoint_for_transform,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions.base import (
    CheckpointTransform,
    SessionOperationPublication,
    StoreTimeCheckpointTransform,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)
from cayu.sessions.records import Session


def _preserve_checkpoint(
    _session: Session,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return checkpoint


def _versioned_checkpoint_transform(
    session_id: str,
    checkpoint_transform: CheckpointTransform,
    *,
    stamp_noop: bool = False,
    stamp_empty: bool = False,
    preserve_completion_result_publications: bool = False,
    preserve_session_exports: bool = True,
    preserve_session_continuations: bool = True,
) -> CheckpointTransform:
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")

    @wraps(checkpoint_transform)
    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        decoded = None
        callback_checkpoint = None
        transformed = None
        try:
            if session.id != session_id:
                raise RuntimeError("Checkpoint transform received another session's authority.")
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            callback_checkpoint = _copy_checkpoint_for_transform(
                decoded,
                session_id=session_id,
                decoded=True,
            )
            transformed = checkpoint_transform(session, callback_checkpoint)
            if transformed is None:
                if stamp_empty:
                    transformed = {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
                elif stamp_noop and checkpoint is not None:
                    transformed = decoded
                else:
                    return None
            result = decode_runtime_checkpoint(transformed, session_id=session_id)
            return _replace_checkpoint_preserving_completion_result_event_publications(
                decoded,
                {} if result is None else result,
                preserve_completion_result_publications=(preserve_completion_result_publications),
                preserve_session_exports=preserve_session_exports,
                preserve_session_continuations=preserve_session_continuations,
                session_id=session_id,
                decoded_replacement=True,
            )
        except BaseException:
            checkpoint = None
            if decoded is not None:
                decoded.clear()
            if callback_checkpoint is not None:
                callback_checkpoint.clear()
            if transformed is not None:
                transformed.clear()
            raise

    return transform


def _optional_versioned_checkpoint_transform(
    session_id: str,
    checkpoint_transform: CheckpointTransform | None,
    *,
    preserve_session_exports: bool = True,
    preserve_session_continuations: bool = True,
) -> CheckpointTransform | None:
    if checkpoint_transform is None:
        return None
    return _versioned_checkpoint_transform(
        session_id,
        checkpoint_transform,
        preserve_session_exports=preserve_session_exports,
        preserve_session_continuations=preserve_session_continuations,
    )


def _versioned_store_time_checkpoint_transform(
    session_id: str,
    checkpoint_transform: StoreTimeCheckpointTransform,
    *,
    stamp_empty: bool = False,
    preserve_completion_result_publications: bool = False,
) -> StoreTimeCheckpointTransform:
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")

    @wraps(checkpoint_transform)
    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
        store_now: Any,
    ) -> dict[str, Any] | None:
        def apply_store_time(
            callback_session: Session,
            callback_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return checkpoint_transform(
                callback_session,
                callback_checkpoint,
                store_now,
            )

        return _versioned_checkpoint_transform(
            session_id,
            apply_store_time,
            stamp_empty=stamp_empty,
            preserve_completion_result_publications=(preserve_completion_result_publications),
        )(session, checkpoint)

    return transform


def _versioned_operation_transform(
    session_id: str,
    operation_transform: Any,
) -> Any:
    if operation_transform is None:
        raise TypeError("operation_transform is required.")

    @wraps(operation_transform)
    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
        operation_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        decoded = None
        callback_checkpoint = None
        publication = None
        versioned = None
        try:
            if session.id != session_id:
                raise RuntimeError("Session operation received another session's authority.")
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            callback_checkpoint = _copy_checkpoint_for_transform(
                decoded,
                session_id=session_id,
                decoded=True,
            )
            publication = operation_transform(session, callback_checkpoint, operation_record)
            if type(publication) is not SessionOperationPublication:
                raise TypeError(
                    "Session operation transform must return a SessionOperationPublication."
                )
            versioned = decode_runtime_checkpoint(
                publication.checkpoint,
                session_id=session_id,
            )
            if versioned is None:
                raise TypeError("Session operation checkpoint must be an object.")
            versioned = _replace_checkpoint_preserving_completion_result_event_publications(
                decoded,
                versioned,
                session_id=session_id,
                decoded_replacement=True,
            )
            return SessionOperationPublication(
                checkpoint=versioned,
                operation_records=publication.operation_records,
                model_completion_stage_release=(publication.model_completion_stage_release),
            )
        except BaseException:
            checkpoint = None
            if decoded is not None:
                decoded.clear()
            if callback_checkpoint is not None:
                callback_checkpoint.clear()
            if publication is not None:
                publication.checkpoint.clear()
            if versioned is not None:
                versioned.clear()
            raise

    return transform


def _versioned_store_time_operation_transform(
    session_id: str,
    operation_transform: Any,
) -> Any:
    if operation_transform is None:
        raise TypeError("operation_transform is required.")

    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
        operation_record: dict[str, Any] | None,
        store_now: Any,
    ) -> SessionOperationPublication:
        def apply_store_time(
            callback_session: Session,
            callback_checkpoint: dict[str, Any] | None,
            callback_operation_record: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            return operation_transform(
                callback_session,
                callback_checkpoint,
                callback_operation_record,
                store_now,
            )

        return _versioned_operation_transform(session_id, apply_store_time)(
            session,
            checkpoint,
            operation_record,
        )

    return transform

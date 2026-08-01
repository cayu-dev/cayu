"""Runtime-owned checkpoint schema boundary over opaque session stores."""

from __future__ import annotations

from functools import wraps
from typing import Any, cast

from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
    validate_runtime_checkpoint_root_projection,
)
from cayu.runtime.sessions import (
    CheckpointRootFieldGuard,
    CheckpointTransform,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    Session,
    SessionOperationPublication,
    SessionStore,
    runtime_publication_checkpoint_value_digest,
)

_ROOT_CHECKPOINT_GUARD = CheckpointRootFieldGuard(
    key=CHECKPOINT_SCHEMA_VERSION_KEY,
    validate=validate_runtime_checkpoint_root_projection,
)


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
) -> CheckpointTransform:
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")

    @wraps(checkpoint_transform)
    def transform(
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        decoded = None
        transformed = None
        try:
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            transformed = checkpoint_transform(session, decoded)
            if transformed is None:
                return decoded if stamp_noop and checkpoint is not None else None
            return decode_runtime_checkpoint(transformed, session_id=session_id)
        except BaseException:
            checkpoint = None
            if decoded is not None:
                decoded.clear()
            if transformed is not None:
                transformed.clear()
            raise

    return transform


def _optional_versioned_checkpoint_transform(
    session_id: str,
    checkpoint_transform: CheckpointTransform | None,
) -> CheckpointTransform | None:
    if checkpoint_transform is None:
        return None
    return _versioned_checkpoint_transform(session_id, checkpoint_transform)


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
        publication = None
        versioned = None
        try:
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            publication = operation_transform(session, decoded, operation_record)
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
            return SessionOperationPublication(
                checkpoint=versioned,
                operation_records=publication.operation_records,
            )
        except BaseException:
            checkpoint = None
            if decoded is not None:
                decoded.clear()
            if publication is not None:
                publication.checkpoint.clear()
            if versioned is not None:
                versioned.clear()
            raise

    return transform


class _RuntimeCheckpointSessionStore:
    """Intercept checkpoint reads and writes without teaching stores their schema."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await self._store.load_checkpoint(session_id)
        try:
            return decode_runtime_checkpoint(checkpoint, session_id=session_id)
        except BaseException:
            checkpoint = None
            raise

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        try:
            checkpoint = decode_runtime_checkpoint(state, session_id=session_id)
        except BaseException:
            state = {}
            raise
        if checkpoint is None:
            raise TypeError("Checkpoint state must be an object.")
        await self._store.checkpoint(session_id, checkpoint)

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        await self._store.transform_checkpoint(
            session_id,
            _versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                stamp_noop=True,
            ),
        )

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        **kwargs: Any,
    ) -> Session:
        return await self._store.transition_status_and_checkpoint(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Any],
        checkpoint_transform: CheckpointTransform,
        **kwargs: Any,
    ) -> None:
        await self._store.append_transcript_messages_and_transform_checkpoint(
            session_id,
            messages,
            _versioned_checkpoint_transform(session_id, checkpoint_transform),
            **kwargs,
        )

    async def create_fork(
        self,
        *,
        source_session_id: str,
        checkpoint_transform: CheckpointTransform | None,
        **kwargs: Any,
    ) -> Session:
        return await self._store.create_fork(
            source_session_id=source_session_id,
            checkpoint_transform=_optional_versioned_checkpoint_transform(
                source_session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        checkpoint_transform: CheckpointTransform | None,
        **kwargs: Any,
    ) -> Session:
        return await self._store.create_fork_with_transcript_validation(
            source_session_id=source_session_id,
            checkpoint_transform=_optional_versioned_checkpoint_transform(
                source_session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        **kwargs: Any,
    ) -> Session:
        return await self._store.fence_run_and_transform_checkpoint(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        **kwargs: Any,
    ) -> Session:
        return await self._store.publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        operation_transform: Any,
        **kwargs: Any,
    ) -> Session:
        return await self._store.publish_session_operation(
            session_id,
            operation_transform=_versioned_operation_transform(
                session_id,
                operation_transform,
            ),
            **kwargs,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        operation_transform: Any,
        **kwargs: Any,
    ) -> Session:
        return await self._store.publish_session_operation_guarded(
            session_id,
            operation_transform=_versioned_operation_transform(
                session_id,
                operation_transform,
            ),
            **kwargs,
        )

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        return await self._store.load_session_operation(
            session_id,
            idempotency_key,
            checkpoint_root_guard=_ROOT_CHECKPOINT_GUARD,
        )

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Any],
        replacement_messages: list[Any],
        **kwargs: Any,
    ) -> None:
        checkpoint_transform = kwargs.pop("checkpoint_transform", None)
        if checkpoint_transform is None:
            checkpoint_transform = _preserve_checkpoint
        await self._store.replace_initial_transcript_messages(
            session_id,
            expected_messages,
            replacement_messages,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
            ),
            **kwargs,
        )

    async def _ensure_checkpoint_version(
        self,
        session_id: str,
    ) -> None:
        raw_checkpoint = await self._store.load_checkpoint(session_id)
        try:
            decode_runtime_checkpoint(
                raw_checkpoint,
                session_id=session_id,
            )
        except BaseException:
            raw_checkpoint = None
            raise
        if (
            raw_checkpoint is not None
            and raw_checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY)
            == CURRENT_CHECKPOINT_SCHEMA_VERSION
        ):
            return

        def stamp(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if checkpoint is None:
                return {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                }
            return checkpoint

        await self.transform_checkpoint(session_id, stamp)

    def _versioned_publication_request(
        self,
        request: RuntimePublicationRequest,
    ) -> RuntimePublicationRequest:
        if any(
            operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
            for operation in request.mutation.operations
        ):
            raise ValueError(
                "Runtime publication callers cannot mutate the root checkpoint schema version."
            )
        schema_operation = RuntimePublicationCheckpointOperation(
            key=CHECKPOINT_SCHEMA_VERSION_KEY,
            expected_value_digest=runtime_publication_checkpoint_value_digest(
                CURRENT_CHECKPOINT_SCHEMA_VERSION,
            ),
            action="set",
            value=CURRENT_CHECKPOINT_SCHEMA_VERSION,
        )
        return RuntimePublicationRequest(
            publication_id=request.publication_id,
            kind=request.kind,
            interaction_id=request.interaction_id,
            intent=request.intent,
            mutation=RuntimePublicationMutation(
                operations=(*request.mutation.operations, schema_operation),
            ),
            transcript_messages=request.transcript_messages,
            events=request.events,
            referenced_events=request.referenced_events,
        )

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        **kwargs: Any,
    ) -> Any:
        await self._ensure_checkpoint_version(session_id)
        return await self._store.publish_runtime_publication(
            session_id,
            request=self._versioned_publication_request(request),
            **kwargs,
        )

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> Any:
        await self._ensure_checkpoint_version(session_id)
        return await self._store.complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=self._versioned_publication_request(publication),
        )

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ) -> Any:
        await self._ensure_checkpoint_version(session_id)
        return await self._store.promote_model_completion_stage(
            session_id,
            stage_id=stage_id,
            expected_run_epoch=expected_run_epoch,
        )

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        return await self._store.load_interruption_cascade_marker(
            session_id,
            checkpoint_root_guard=_ROOT_CHECKPOINT_GUARD,
        )

    async def query_pending_actions(self, query: Any = None) -> Any:
        return await self._store.query_pending_actions(
            query,
            checkpoint_root_guard=_ROOT_CHECKPOINT_GUARD,
        )

    async def inspect_summary(self, session_id: str) -> Any:
        return await self._store.inspect_summary(
            session_id,
            checkpoint_root_guard=_ROOT_CHECKPOINT_GUARD,
        )


def runtime_checkpoint_session_store(store: SessionStore) -> SessionStore:
    """Return a runtime-only schema adapter while preserving the public raw store."""

    if isinstance(store, _RuntimeCheckpointSessionStore):
        return cast("SessionStore", store)
    return cast("SessionStore", _RuntimeCheckpointSessionStore(store))

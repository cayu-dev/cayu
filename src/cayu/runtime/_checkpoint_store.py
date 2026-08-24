"""Runtime-owned checkpoint schema boundary over opaque session stores."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any, cast

from cayu._validation import copy_durable_json_object
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
    validate_runtime_checkpoint_root_projection,
)
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.sessions import (
    CheckpointRootFieldGuard,
    CheckpointTransform,
    ProfiledSessionForkResult,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    Session,
    SessionInvocationAdmission,
    SessionOperationPublication,
    SessionStore,
    _apply_runtime_publication_checkpoint_mutation,
    _replace_checkpoint_preserving_completion_result_event_publications,
    _runtime_publication_checkpoint_codec_scope,
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
    stamp_empty: bool = False,
    preserve_completion_result_publications: bool = False,
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
                if stamp_empty:
                    return {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
                return decoded if stamp_noop and checkpoint is not None else None
            result = decode_runtime_checkpoint(transformed, session_id=session_id)
            if preserve_completion_result_publications:
                return _replace_checkpoint_preserving_completion_result_event_publications(
                    checkpoint,
                    {} if result is None else result,
                )
            return result
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
            versioned = _replace_checkpoint_preserving_completion_result_event_publications(
                checkpoint,
                versioned,
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

    @property
    def supports_owned_off_thread_session_commit_guards(self) -> bool:
        """Preserve the wrapped store's guarded-mutation capability exactly."""

        return self._supports_owned_off_thread_session_commit_guard_protocol()

    def _supports_owned_off_thread_session_commit_guard_protocol(self) -> bool:
        checker = getattr(
            self._store,
            "_supports_owned_off_thread_session_commit_guard_protocol",
            None,
        )
        return callable(checker) and checker() is True

    def _supports_completion_result_event_publication_reservation_protocol(self) -> bool:
        checker = getattr(
            self._store,
            "_supports_completion_result_event_publication_reservation_protocol",
            None,
        )
        return callable(checker) and checker() is True

    async def create(
        self,
        request: Any,
        *,
        identity: Any,
        interaction_started_event: Any = None,
        interaction_source_messages: Any = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        if checkpoint_transform is None:
            return await self._store.create(
                request,
                identity=identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=interaction_source_messages,
            )

        def transform_initial_checkpoint(
            session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return _versioned_checkpoint_transform(
                session.id,
                checkpoint_transform,
            )(session, checkpoint)

        return await self._store.create(
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=transform_initial_checkpoint,
        )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await self._store.load_checkpoint(session_id)
        try:
            return decode_runtime_checkpoint(checkpoint, session_id=session_id)
        except BaseException:
            checkpoint = None
            raise

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        checkpoint = None
        try:
            checkpoint = decode_runtime_checkpoint(state, session_id=session_id)
            if checkpoint is None:
                raise TypeError("Checkpoint state must be an object.")
            encoded_checkpoint = checkpoint

            def replace_checkpoint(
                _session: Session,
                current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                decode_runtime_checkpoint(current, session_id=session_id)
                return _replace_checkpoint_preserving_completion_result_event_publications(
                    current,
                    encoded_checkpoint,
                )

            await self._store.transform_checkpoint(session_id, replace_checkpoint)
        except BaseException:
            state = {}
            if checkpoint is not None:
                checkpoint.clear()
            raise

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
                preserve_completion_result_publications=True,
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
                preserve_completion_result_publications=True,
            ),
            **kwargs,
        )

    async def admit_execution_profile_resume(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        execution_profile: ExecutionProfileIdentity,
        **kwargs: Any,
    ) -> Session:
        return await self._store.admit_execution_profile_resume(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            execution_profile=execution_profile,
            **kwargs,
        )

    async def admit_session_invocation(
        self,
        session_id: str,
        *,
        admission: SessionInvocationAdmission,
    ) -> Session:
        return await self._store.admit_session_invocation(
            session_id,
            admission=replace(
                admission,
                checkpoint_transform=_versioned_checkpoint_transform(
                    session_id,
                    admission.checkpoint_transform,
                    stamp_empty=True,
                    preserve_completion_result_publications=True,
                ),
            ),
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
            _versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
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

    async def create_profiled_fork(
        self,
        *,
        source_session_id: str,
        checkpoint_transform: CheckpointTransform | None,
        **kwargs: Any,
    ) -> ProfiledSessionForkResult:
        def decode_profile_authority(
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return decode_runtime_checkpoint(
                checkpoint,
                session_id=source_session_id,
            )

        return await self._store.create_profiled_fork(
            source_session_id=source_session_id,
            checkpoint_transform=_optional_versioned_checkpoint_transform(
                source_session_id,
                checkpoint_transform,
            ),
            checkpoint_authority_decoder=decode_profile_authority,
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
                preserve_completion_result_publications=True,
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
                preserve_completion_result_publications=True,
            ),
            **kwargs,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Any],
    ) -> Session:
        return await self._store._publish_completion_result_event_publication(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
            ),
            events=events,
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
                preserve_completion_result_publications=True,
            ),
            **kwargs,
        )

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
        if any(
            operation.key == ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
            for operation in request.mutation.operations
        ):
            raise ValueError(
                "Runtime publication callers cannot mutate active invocation "
                "execution-profile authority."
            )
        if any(
            operation.key == COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY
            for operation in request.mutation.operations
        ):
            raise ValueError(
                "Runtime publication callers cannot mutate completion-result event "
                "publication authority."
            )
        schema_operation = RuntimePublicationCheckpointOperation(
            key=CHECKPOINT_SCHEMA_VERSION_KEY,
            expected_value_digest=runtime_publication_checkpoint_value_digest(
                CURRENT_CHECKPOINT_SCHEMA_VERSION
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

    def _decode_publication_checkpoint(
        self,
        session: Session,
        raw_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        decoded = decode_runtime_checkpoint(raw_checkpoint, session_id=session.id)
        if decoded is not None:
            return decoded
        # Publication requests are expressed against the current logical
        # schema. Treat a missing root as that empty logical root so their
        # schema fence can be evaluated before the current writer commits it.
        return {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    def _encode_publication_checkpoint(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return decode_runtime_checkpoint(checkpoint, session_id=session.id)

    def _apply_publication_checkpoint_mutation(
        self,
        session: Session,
        raw_checkpoint: dict[str, Any] | None,
        mutation: RuntimePublicationMutation,
    ) -> dict[str, Any] | None:
        """Apply a staged mutation in its writer schema, then upcast the result."""

        writer_version = CURRENT_CHECKPOINT_SCHEMA_VERSION
        schema_operations = [
            operation
            for operation in mutation.operations
            if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
        ]
        if len(schema_operations) == 1 and type(schema_operations[0].value) is int:
            writer_version = schema_operations[0].value
        elif not schema_operations:
            pointer_operation = next(
                (
                    operation
                    for operation in mutation.operations
                    if operation.key == "last_model_step_publication"
                    and operation.action == "set"
                    and type(operation.value) is dict
                ),
                None,
            )
            if pointer_operation is not None:
                pointer_version = pointer_operation.value.get("schema_version")
                if type(pointer_version) is int:
                    writer_version = pointer_version

        writer_checkpoint = raw_checkpoint
        preserved_active_profile: dict[str, Any] | None = None
        if writer_version != CURRENT_CHECKPOINT_SCHEMA_VERSION and raw_checkpoint is not None:
            current_checkpoint = decode_runtime_checkpoint(
                raw_checkpoint,
                session_id=session.id,
            )
            if (
                current_checkpoint is not None
                and ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY in current_checkpoint
            ):
                if any(
                    operation.key == ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
                    for operation in mutation.operations
                ):
                    raise ValueError(
                        "An older runtime publication cannot mutate active invocation "
                        "execution-profile authority."
                    )
                writer_checkpoint = copy_durable_json_object(
                    current_checkpoint,
                    "checkpoint",
                )
                preserved_active_profile = copy_durable_json_object(
                    writer_checkpoint.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY),
                    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                )

        source_checkpoint = runtime_checkpoint_writer_view(
            writer_checkpoint,
            writer_version=writer_version,
            session_id=session.id,
        )

        target_checkpoint = _apply_runtime_publication_checkpoint_mutation(
            mutation,
            source_checkpoint,
        )
        decoded_target = decode_runtime_checkpoint(target_checkpoint, session_id=session.id)
        if preserved_active_profile is not None:
            if decoded_target is None:
                raise ValueError(
                    "An older runtime publication cannot remove active invocation "
                    "execution-profile authority."
                )
            decoded_target[ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY] = (
                preserved_active_profile
            )
        return decoded_target

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        **kwargs: Any,
    ) -> Any:
        versioned_request = self._versioned_publication_request(request)
        with _runtime_publication_checkpoint_codec_scope(
            decode=self._decode_publication_checkpoint,
            encode=self._encode_publication_checkpoint,
            apply_mutation=self._apply_publication_checkpoint_mutation,
        ):
            return await self._store.publish_runtime_publication(
                session_id,
                request=versioned_request,
                **kwargs,
            )

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> Any:
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
        with _runtime_publication_checkpoint_codec_scope(
            decode=self._decode_publication_checkpoint,
            encode=self._encode_publication_checkpoint,
            apply_mutation=self._apply_publication_checkpoint_mutation,
        ):
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


def runtime_checkpoint_session_store(
    store: SessionStore,
) -> SessionStore:
    """Return a runtime-only schema adapter while preserving the public raw store."""

    if isinstance(store, _RuntimeCheckpointSessionStore):
        return cast("SessionStore", store)
    return cast(
        "SessionStore",
        _RuntimeCheckpointSessionStore(store),
    )

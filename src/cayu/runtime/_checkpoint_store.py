"""Runtime-owned checkpoint schema boundary over opaque session stores."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any, cast, overload

if TYPE_CHECKING:
    from cayu.runtime._invocation_lifecycle import (
        AdmitInvocationCommand,
        CreateInvocationCommand,
        InvocationLifecycleCommand,
        InvocationMutationResult,
        InvocationReleaseResult,
        RebindInvocationCommand,
        RejectInvocationCommand,
        ReleaseInvocationCommand,
        SettleInvocationCommand,
    )
    from cayu.runtime.execution_profiles import ExecutionProfileRejectionResult
    from cayu.runtime.sessions import InteractionTransitionResult

from cayu._validation import copy_durable_json_object
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
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
    SessionOperationInitializer,
    SessionOperationPublication,
    SessionStore,
    StoreTimeCheckpointTransform,
    _apply_runtime_publication_checkpoint_mutation,
    _copy_checkpoint_for_transform,
    _invocation_lifecycle_authority_read_scope,
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
        callback_checkpoint = None
        transformed = None
        try:
            if session.id != session_id:
                raise RuntimeError("Checkpoint transform received another session's authority.")
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            callback_checkpoint = _copy_checkpoint_for_transform(
                decoded,
                session_id=session_id,
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
                session_id=session_id,
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
) -> CheckpointTransform | None:
    if checkpoint_transform is None:
        return None
    return _versioned_checkpoint_transform(session_id, checkpoint_transform)


def _versioned_store_time_checkpoint_transform(
    session_id: str,
    checkpoint_transform: StoreTimeCheckpointTransform,
    *,
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
                checkpoint,
                versioned,
                session_id=session_id,
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

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: CreateInvocationCommand | AdmitInvocationCommand | RebindInvocationCommand,
    ) -> InvocationMutationResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: RejectInvocationCommand,
    ) -> ExecutionProfileRejectionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: SettleInvocationCommand,
    ) -> InteractionTransitionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: ReleaseInvocationCommand,
    ) -> InvocationReleaseResult: ...

    async def apply_invocation_lifecycle_command(
        self,
        command: object,
    ) -> object:
        """Apply lifecycle commands without bypassing the root checkpoint codec."""

        checker = getattr(
            self._store,
            "_supports_invocation_lifecycle_command_protocol",
            None,
        )
        if not callable(checker) or checker() is not True:
            raise NotImplementedError(
                "This SessionStore does not support invocation lifecycle command version 1."
            )
        from cayu.runtime._invocation_lifecycle import apply_invocation_lifecycle_command

        return await apply_invocation_lifecycle_command(
            cast("SessionStore", self),
            cast("InvocationLifecycleCommand", command),
        )

    async def create(
        self,
        request: Any,
        *,
        identity: Any,
        interaction_started_event: Any = None,
        interaction_source_messages: Any = None,
        checkpoint_transform: CheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        versioned_result_transform = None
        if result_checkpoint_transform is not None:

            def transform_result_checkpoint(
                session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                return _versioned_checkpoint_transform(
                    session.id,
                    result_checkpoint_transform,
                )(session, checkpoint)

            versioned_result_transform = transform_result_checkpoint
        if checkpoint_transform is None:
            if operation_initializer is None:
                if versioned_result_transform is None:
                    return await self._store.create(
                        request,
                        identity=identity,
                        interaction_started_event=interaction_started_event,
                        interaction_source_messages=interaction_source_messages,
                    )
                return await self._store.create(
                    request,
                    identity=identity,
                    interaction_started_event=interaction_started_event,
                    interaction_source_messages=interaction_source_messages,
                    result_checkpoint_transform=versioned_result_transform,
                )
            if versioned_result_transform is None:
                return await self._store.create(
                    request,
                    identity=identity,
                    interaction_started_event=interaction_started_event,
                    interaction_source_messages=interaction_source_messages,
                    operation_initializer=operation_initializer,
                )
            return await self._store.create(
                request,
                identity=identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=interaction_source_messages,
                result_checkpoint_transform=versioned_result_transform,
                operation_initializer=operation_initializer,
            )

        def transform_initial_checkpoint(
            session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return _versioned_checkpoint_transform(
                session.id,
                checkpoint_transform,
            )(session, checkpoint)

        if operation_initializer is None:
            if versioned_result_transform is None:
                return await self._store.create(
                    request,
                    identity=identity,
                    interaction_started_event=interaction_started_event,
                    interaction_source_messages=interaction_source_messages,
                    checkpoint_transform=transform_initial_checkpoint,
                )
            return await self._store.create(
                request,
                identity=identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=interaction_source_messages,
                checkpoint_transform=transform_initial_checkpoint,
                result_checkpoint_transform=versioned_result_transform,
            )
        if versioned_result_transform is None:
            return await self._store.create(
                request,
                identity=identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=interaction_source_messages,
                checkpoint_transform=transform_initial_checkpoint,
                operation_initializer=operation_initializer,
            )
        return await self._store.create(
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=transform_initial_checkpoint,
            result_checkpoint_transform=versioned_result_transform,
            operation_initializer=operation_initializer,
        )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await self._store.load_checkpoint(session_id)
        try:
            return decode_runtime_checkpoint(checkpoint, session_id=session_id)
        except BaseException:
            checkpoint = None
            raise

    async def load_session_checkpoint_snapshot(
        self,
        session_id: str,
    ) -> tuple[Session, dict[str, Any] | None]:
        """Atomically read runtime authority, writing only an actual schema migration."""

        checker = getattr(
            self._store,
            "_supports_invocation_lifecycle_command_protocol",
            None,
        )
        if not callable(checker) or checker() is not True:
            raise NotImplementedError(
                "This SessionStore does not support invocation lifecycle command version 1."
            )

        snapshot: tuple[Session, dict[str, Any] | None] | None = None

        def capture_snapshot(
            session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal snapshot
            if session.id != session_id:
                raise RuntimeError("Checkpoint snapshot received another session's authority.")
            decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
            snapshot = (
                session.model_copy(deep=True),
                (None if decoded is None else copy_durable_json_object(decoded, "checkpoint")),
            )
            if decoded == checkpoint:
                return None
            return decoded

        with _invocation_lifecycle_authority_read_scope():
            await self._store.transform_checkpoint(session_id, capture_snapshot)
        if snapshot is None:
            raise RuntimeError("Runtime checkpoint snapshot was not produced.")
        return snapshot

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
                    session_id=session_id,
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

    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        await self._store.transform_checkpoint_with_store_time(
            session_id,
            _versioned_store_time_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
        )

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        **kwargs: Any,
    ) -> Session:
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        versioned_checkpoint_transform = (
            None
            if checkpoint_transform is None
            else _versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            )
        )
        versioned_store_time_transform = (
            None
            if store_time_checkpoint_transform is None
            else _versioned_store_time_checkpoint_transform(
                session_id,
                store_time_checkpoint_transform,
                preserve_completion_result_publications=True,
            )
        )
        return await self._store.transition_status_and_checkpoint(
            session_id,
            checkpoint_transform=versioned_checkpoint_transform,
            store_time_checkpoint_transform=versioned_store_time_transform,
            result_checkpoint_transform=_optional_versioned_checkpoint_transform(
                session_id,
                result_checkpoint_transform,
            ),
            **kwargs,
        )

    async def admit_execution_profile_resume(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
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
            result_checkpoint_transform=_optional_versioned_checkpoint_transform(
                session_id,
                result_checkpoint_transform,
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
                result_checkpoint_transform=_optional_versioned_checkpoint_transform(
                    session_id,
                    admission.result_checkpoint_transform,
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
        if kwargs.get("operation_initializer") is None:
            kwargs.pop("operation_initializer", None)
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
        if kwargs.get("operation_initializer") is None:
            kwargs.pop("operation_initializer", None)
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
        if kwargs.get("operation_initializer") is None:
            kwargs.pop("operation_initializer", None)

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
        result_checkpoint_transform: CheckpointTransform | None = None,
        **kwargs: Any,
    ) -> Session:
        return await self._store.fence_run_and_transform_checkpoint(
            session_id,
            checkpoint_transform=_versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            result_checkpoint_transform=_optional_versioned_checkpoint_transform(
                session_id,
                result_checkpoint_transform,
            ),
            **kwargs,
        )

    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        **kwargs: Any,
    ) -> Session | None:
        return await self._store.reserve_stalled_run_recovery(
            session_id,
            checkpoint_transform=_versioned_store_time_checkpoint_transform(
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

    async def publish_checkpoint_and_events_with_store_time(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        **kwargs: Any,
    ) -> Session:
        return await self._store.publish_checkpoint_and_events_with_store_time(
            session_id,
            checkpoint_transform=_versioned_store_time_checkpoint_transform(
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
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Any],
    ) -> Session:
        return await self._store._publish_completion_result_event_publication(
            session_id,
            checkpoint_transform=_versioned_store_time_checkpoint_transform(
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

    async def publish_session_operation_guarded_with_store_time(
        self,
        session_id: str,
        *,
        operation_transform: Any,
        **kwargs: Any,
    ) -> Session:
        return await self._store.publish_session_operation_guarded_with_store_time(
            session_id,
            operation_transform=_versioned_store_time_operation_transform(
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
        schema_operations = tuple(
            operation
            for operation in request.mutation.operations
            if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
        )
        if schema_operations:
            if request.kind != "workspace-observation":
                raise ValueError(
                    "Only workspace-observation publications may carry a root checkpoint "
                    "schema stamp."
                )
            if len(schema_operations) != 1:
                raise ValueError("Runtime publication carries duplicate schema operations.")
            schema_operation = schema_operations[0]
            supported_schema_digests = {
                runtime_publication_checkpoint_value_digest(version)
                for version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
            }
            if (
                schema_operation.action != "set"
                or schema_operation.value != CURRENT_CHECKPOINT_SCHEMA_VERSION
                or schema_operation.expected_value_digest not in supported_schema_digests | {None}
            ):
                raise ValueError(
                    "Runtime publication carries an invalid root checkpoint schema stamp."
                )
        if any(
            operation.key
            in {
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            }
            for operation in request.mutation.operations
        ):
            raise ValueError(
                "Runtime publication callers cannot mutate active invocation lifecycle authority."
            )
        if any(
            operation.key == COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY
            for operation in request.mutation.operations
        ):
            raise ValueError(
                "Runtime publication callers cannot mutate completion-result event "
                "publication authority."
            )
        if schema_operations:
            return request
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
            operation_record_mutations=request.operation_record_mutations,
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
        preserved_lifecycle_receipt: dict[str, Any] | None = None
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
            if (
                current_checkpoint is not None
                and INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in current_checkpoint
            ):
                if any(
                    operation.key == INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
                    for operation in mutation.operations
                ):
                    raise ValueError(
                        "An older runtime publication cannot mutate invocation lifecycle "
                        "receipt authority."
                    )
                if writer_checkpoint is raw_checkpoint:
                    writer_checkpoint = copy_durable_json_object(
                        current_checkpoint,
                        "checkpoint",
                    )
                if writer_checkpoint is None:
                    raise RuntimeError(
                        "A versioned checkpoint writer lost current lifecycle authority."
                    )
                preserved_lifecycle_receipt = copy_durable_json_object(
                    writer_checkpoint.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY),
                    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                )

        source_checkpoint = runtime_checkpoint_writer_view(
            writer_checkpoint,
            writer_version=writer_version,
            session_id=session.id,
        )

        applied_mutation = mutation
        if raw_checkpoint is None:
            applied_operations = tuple(
                (
                    RuntimePublicationCheckpointOperation(
                        key=operation.key,
                        expected_value_digest=runtime_publication_checkpoint_value_digest(
                            CURRENT_CHECKPOINT_SCHEMA_VERSION
                        ),
                        action=operation.action,
                        value=operation.value,
                    )
                    if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
                    and operation.expected_value_digest is None
                    and operation.action == "set"
                    and operation.value == CURRENT_CHECKPOINT_SCHEMA_VERSION
                    else operation
                )
                for operation in mutation.operations
            )
            applied_mutation = RuntimePublicationMutation(operations=applied_operations)

        target_checkpoint = _apply_runtime_publication_checkpoint_mutation(
            applied_mutation,
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
        if preserved_lifecycle_receipt is not None:
            if decoded_target is None:
                raise ValueError(
                    "An older runtime publication cannot remove invocation lifecycle "
                    "receipt authority."
                )
            decoded_target[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = (
                preserved_lifecycle_receipt
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


async def load_runtime_session_checkpoint_snapshot(
    store: SessionStore,
    session_id: str,
) -> tuple[Session, dict[str, Any] | None]:
    """Load one atomic runtime-only session/checkpoint authority snapshot."""

    if not isinstance(store, _RuntimeCheckpointSessionStore):
        raise TypeError("Runtime checkpoint snapshots require the private store adapter.")
    return await store.load_session_checkpoint_snapshot(session_id)

"""Runtime-owned checkpoint schema boundary over opaque session stores."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast, overload

from cayu.sessions import _checkpoint_publication as checkpoint_publication
from cayu.sessions import _checkpoint_transforms as checkpoint_transforms

if TYPE_CHECKING:
    from cayu.execution_profiles import (
        ExecutionProfileRejectionResult,
    )
    from cayu.sessions._invocation_lifecycle import (
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
    from cayu.sessions.base import InteractionTransitionResult

from cayu._validation import copy_durable_json_object
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_read_scope,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions.base import (
    CheckpointRootFieldGuard,
    CheckpointTransform,
    ModelCompletionStageRecoveryFence,
    RuntimePublicationRequest,
    SessionInvocationAdmission,
    SessionOperationInitializer,
    SessionStore,
    StoreTimeCheckpointTransform,
    _runtime_publication_checkpoint_codec_scope,
    _session_continuation_methods_owned,
    _session_export_methods_owned,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    _DecodedRuntimeCheckpoint,
    decode_runtime_checkpoint,
    validate_runtime_checkpoint_root_projection,
)
from cayu.sessions.forks import (
    ProfiledSessionForkResult,
)
from cayu.sessions.records import Session

_ROOT_CHECKPOINT_GUARD = CheckpointRootFieldGuard(
    key=CHECKPOINT_SCHEMA_VERSION_KEY,
    validate=validate_runtime_checkpoint_root_projection,
)


class _RuntimeCheckpointSessionStore:
    """Intercept checkpoint reads and writes without teaching stores their schema."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    @property
    def session_export_version(self) -> int:
        return 1 if self._supports_session_export_protocol() else 0

    def _supports_session_export_protocol(self) -> bool:
        checker = getattr(self._store, "_supports_session_export_protocol", None)
        return _session_export_methods_owned(self) and callable(checker) and checker() is True

    @property
    def session_continuation_version(self) -> int:
        return 1 if self._supports_session_continuation_protocol() else 0

    def _supports_session_continuation_protocol(self) -> bool:
        checker = getattr(self._store, "_supports_session_continuation_protocol", None)
        return _session_continuation_methods_owned(self) and callable(checker) and checker() is True

    async def append_tool_effect_conflict(self, request: object) -> Any:
        """Evidence-only append has no checkpoint to decode, project, or stamp."""
        return await self._store.append_tool_effect_conflict(request)

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

    def _supports_terminal_interaction_publication_protocol(self) -> bool:
        checker = getattr(
            self._store,
            "_supports_terminal_interaction_publication_protocol",
            None,
        )
        return callable(checker) and checker() is True

    async def _admit_native_producer(self, registration, command):
        checker = getattr(self._store, "_supports_producer_attachment_protocol", None)
        if not callable(checker) or checker() is not True:
            raise NotImplementedError("Native producer admission is not qualified.")
        from cayu.runtime._producer_output_store import admit_native_producer

        return await admit_native_producer(self, registration, command)

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
                return checkpoint_transforms._versioned_checkpoint_transform(
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
            return checkpoint_transforms._versioned_checkpoint_transform(
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

    async def _load_decoded_runtime_checkpoint(self, session_id: str) -> _DecodedRuntimeCheckpoint:
        load_checkpoint = self.load_checkpoint
        if (
            getattr(load_checkpoint, "__func__", None)
            is _RuntimeCheckpointSessionStore.load_checkpoint
        ):
            load_checkpoint = self._store.load_checkpoint
        # An override can enforce its own read policy. Preserve it even when
        # that requires admitting its result again before transferring it.
        checkpoint = await load_checkpoint(session_id)
        try:
            return _DecodedRuntimeCheckpoint(checkpoint, session_id=session_id)
        finally:
            checkpoint = None

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
            unchanged = decoded == checkpoint
            snapshot = (
                session.model_copy(deep=True),
                (
                    decoded
                    if unchanged or decoded is None
                    else copy_durable_json_object(decoded, "checkpoint")
                ),
            )
            if unchanged:
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
            checkpoint_transforms._versioned_checkpoint_transform(
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
            checkpoint_transforms._versioned_store_time_checkpoint_transform(
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
            else checkpoint_transforms._versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            )
        )
        versioned_store_time_transform = (
            None
            if store_time_checkpoint_transform is None
            else checkpoint_transforms._versioned_store_time_checkpoint_transform(
                session_id,
                store_time_checkpoint_transform,
                preserve_completion_result_publications=True,
            )
        )
        return await self._store.transition_status_and_checkpoint(
            session_id,
            checkpoint_transform=versioned_checkpoint_transform,
            store_time_checkpoint_transform=versioned_store_time_transform,
            result_checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
                session_id,
                result_checkpoint_transform,
            ),
            **kwargs,
        )

    async def admit_execution_profile_resume(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        execution_profile: ExecutionProfileIdentity,
        **kwargs: Any,
    ) -> Session:
        return await self._store.admit_execution_profile_resume(
            session_id,
            checkpoint_transform=None
            if checkpoint_transform is None
            else checkpoint_transforms._versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            store_time_checkpoint_transform=None
            if store_time_checkpoint_transform is None
            else checkpoint_transforms._versioned_store_time_checkpoint_transform(
                session_id,
                store_time_checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            result_checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
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
                checkpoint_transform=None
                if admission.checkpoint_transform is None
                else checkpoint_transforms._versioned_checkpoint_transform(
                    session_id,
                    admission.checkpoint_transform,
                    stamp_empty=True,
                    preserve_completion_result_publications=True,
                ),
                store_time_checkpoint_transform=None
                if admission.store_time_checkpoint_transform is None
                else checkpoint_transforms._versioned_store_time_checkpoint_transform(
                    session_id,
                    admission.store_time_checkpoint_transform,
                    stamp_empty=True,
                    preserve_completion_result_publications=True,
                ),
                result_checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
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
            checkpoint_transforms._versioned_checkpoint_transform(
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
            checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
                source_session_id,
                checkpoint_transform,
                preserve_session_exports=False,
                preserve_session_continuations=False,
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
            checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
                source_session_id,
                checkpoint_transform,
                preserve_session_exports=False,
                preserve_session_continuations=False,
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

        versioned_transform = checkpoint_transforms._optional_versioned_checkpoint_transform(
            source_session_id,
            checkpoint_transform,
            preserve_session_exports=False,
            preserve_session_continuations=False,
        )

        def validate_fork_source(
            session: Session, checkpoint: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            if versioned_transform is None:
                return None
            # This internal owner compares the exact source checkpoint, which
            # includes model selection and effective invocation authority.
            # Generic callbacks retain their restricted view. Read permission
            # cannot replace private roots, and the store's fork transform
            # removes all source-owned routing authority from the child.
            with _invocation_lifecycle_authority_read_scope():
                return versioned_transform(session, checkpoint)

        return await self._store.create_profiled_fork(
            source_session_id=source_session_id,
            checkpoint_transform=None if versioned_transform is None else validate_fork_source,
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
            checkpoint_transform=checkpoint_transforms._versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            result_checkpoint_transform=checkpoint_transforms._optional_versioned_checkpoint_transform(
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
            checkpoint_transform=checkpoint_transforms._versioned_store_time_checkpoint_transform(
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
            checkpoint_transform=checkpoint_transforms._versioned_checkpoint_transform(
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
            checkpoint_transform=checkpoint_transforms._versioned_store_time_checkpoint_transform(
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
            checkpoint_transform=checkpoint_transforms._versioned_store_time_checkpoint_transform(
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
            operation_transform=checkpoint_transforms._versioned_operation_transform(
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
            operation_transform=checkpoint_transforms._versioned_operation_transform(
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
            operation_transform=checkpoint_transforms._versioned_store_time_operation_transform(
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
            checkpoint_transform = checkpoint_transforms._preserve_checkpoint
        await self._store.replace_initial_transcript_messages(
            session_id,
            expected_messages,
            replacement_messages,
            checkpoint_transform=checkpoint_transforms._versioned_checkpoint_transform(
                session_id,
                checkpoint_transform,
                preserve_completion_result_publications=True,
            ),
            **kwargs,
        )

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        **kwargs: Any,
    ) -> Any:
        versioned_request = checkpoint_publication._versioned_publication_request(request)
        with _runtime_publication_checkpoint_codec_scope(
            decode=checkpoint_publication._decode_publication_checkpoint,
            encode=checkpoint_publication._encode_publication_checkpoint,
            apply_mutation=checkpoint_publication._apply_publication_checkpoint_mutation,
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
            publication=checkpoint_publication._versioned_publication_request(publication),
        )

    async def complete_recovered_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
        recovery_fence: ModelCompletionStageRecoveryFence,
    ) -> Any:
        return await self._store.complete_recovered_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=checkpoint_publication._versioned_publication_request(publication),
            recovery_fence=recovery_fence,
        )

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ) -> Any:
        with _runtime_publication_checkpoint_codec_scope(
            decode=checkpoint_publication._decode_publication_checkpoint,
            encode=checkpoint_publication._encode_publication_checkpoint,
            apply_mutation=checkpoint_publication._apply_publication_checkpoint_mutation,
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

"""Publication schema composition within the caller's native transaction.

Compose canonical publication records, checkpoint decoding and writer projection.
The task-local codec scope is captured by publication preparation; callers retain
runtime dispatch and the complete atomic store operation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    _apply_runtime_publication_checkpoint_mutation,
    _apply_runtime_publication_checkpoint_operations,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_CONFIGURATION_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
)
from cayu.sessions.records import CheckpointTransform, Session


def _versioned_publication_request(
    request: RuntimePublicationRequest,
) -> RuntimePublicationRequest:
    if request.kind == "auxiliary-inference":
        if request.mutation.operations:
            raise ValueError("Auxiliary publications cannot mutate the parent checkpoint.")
        # Auxiliary accounting has no checkpoint ownership. Even an otherwise
        # harmless schema stamp changes its exact no-mutation publication.
        return request
    schema_operations = tuple(
        operation
        for operation in request.mutation.operations
        if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
    )
    if schema_operations:
        if request.kind != "workspace-observation":
            raise ValueError(
                "Only workspace-observation publications may carry a root checkpoint schema stamp."
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
            raise ValueError("Runtime publication carries an invalid root checkpoint schema stamp.")
    if any(
        operation.key
        in {
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            INVOCATION_CONFIGURATION_CHECKPOINT_KEY,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
            SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        }
        for operation in request.mutation.operations
    ):
        raise ValueError(
            "Runtime publication callers cannot mutate invocation lifecycle authority."
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
        argument_continuity=request.argument_continuity,
    )


def _decode_publication_checkpoint(
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
    session: Session,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return decode_runtime_checkpoint(checkpoint, session_id=session.id)


def _publication_writer_version(mutation: RuntimePublicationMutation) -> int:
    """Return the checkpoint schema a staged publication's mutation was written in."""

    schema_operations = [
        operation
        for operation in mutation.operations
        if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
    ]
    if len(schema_operations) == 1 and type(schema_operations[0].value) is int:
        return schema_operations[0].value
    if not schema_operations:
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
                return pointer_version
    return CURRENT_CHECKPOINT_SCHEMA_VERSION


def _mutation_for_root(
    mutation: RuntimePublicationMutation,
    *,
    root_present: bool,
) -> RuntimePublicationMutation:
    """Fence a current schema stamp on an absent root against its empty logical root."""

    if root_present:
        return mutation
    return RuntimePublicationMutation(
        operations=tuple(
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
    )


def _apply_publication_checkpoint_mutation(
    session: Session,
    raw_checkpoint: dict[str, Any] | None,
    mutation: RuntimePublicationMutation,
) -> dict[str, Any] | None:
    """Apply a staged mutation in its writer schema, then upcast the result."""

    writer_version = _publication_writer_version(mutation)
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

    target_checkpoint = _apply_runtime_publication_checkpoint_mutation(
        _mutation_for_root(mutation, root_present=raw_checkpoint is not None),
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
                "An older runtime publication cannot remove invocation lifecycle receipt authority."
            )
        decoded_target[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = preserved_lifecycle_receipt
    return decoded_target


def _encode_publication_checkpoint_mutation(
    session: Session,
    raw_checkpoint: dict[str, Any] | None,
    decoded_checkpoint: dict[str, Any],
    mutation: RuntimePublicationMutation,
) -> dict[str, Any] | None:
    """Return the stored target, reusing the publication's one decoded checkpoint.

    ``decoded_checkpoint`` is ``_decode_publication_checkpoint(session,
    raw_checkpoint)`` from the same transaction. The publication owns it and
    reads none of it after this call, so a current-schema mutation is applied to
    it in place. The result is then decoded once: that checks the whole new
    document, and it is already in the stored form, so it needs no encoding.
    The result equals ``_encode_publication_checkpoint`` of
    ``_apply_publication_checkpoint_mutation`` and is a fresh validated copy.
    """

    if _publication_writer_version(mutation) != CURRENT_CHECKPOINT_SCHEMA_VERSION:
        # An older writer's projection pops and restores private roots, so its
        # result is encoded (and checked whole) after they are back.
        return _encode_publication_checkpoint(
            session,
            _apply_publication_checkpoint_mutation(session, raw_checkpoint, mutation),
        )
    _apply_runtime_publication_checkpoint_operations(
        _mutation_for_root(mutation, root_present=raw_checkpoint is not None),
        decoded_checkpoint,
    )
    return decode_runtime_checkpoint(decoded_checkpoint, session_id=session.id)


def _validate_checkpoint_codec_pair(
    checkpoint_decode: CheckpointTransform | None,
    checkpoint_encode: CheckpointTransform | None,
) -> None:
    if (checkpoint_decode is None) != (checkpoint_encode is None):
        raise ValueError("Checkpoint decode and encode transforms must be supplied together.")
    if checkpoint_decode is not None and not callable(checkpoint_decode):
        raise TypeError("checkpoint_decode must be callable.")
    if checkpoint_encode is not None and not callable(checkpoint_encode):
        raise TypeError("checkpoint_encode must be callable.")


@dataclass(frozen=True)
class _RuntimePublicationCheckpointCodec:
    """Task-scoped checkpoint projection captured by a prepared publication."""

    decode: CheckpointTransform
    encode: CheckpointTransform
    apply_mutation: Callable[
        [Session, dict[str, Any] | None, RuntimePublicationMutation],
        dict[str, Any] | None,
    ]
    encode_mutation: (
        Callable[
            [Session, dict[str, Any] | None, dict[str, Any], RuntimePublicationMutation],
            dict[str, Any] | None,
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        _validate_checkpoint_codec_pair(self.decode, self.encode)
        if not callable(self.apply_mutation):
            raise TypeError("Checkpoint mutation projection must be callable.")
        if self.encode_mutation is not None and not callable(self.encode_mutation):
            raise TypeError("Checkpoint mutation encoding must be callable.")

    def stored_target(
        self,
        session: Session,
        stored_checkpoint: dict[str, Any] | None,
        current_checkpoint: dict[str, Any] | None,
        mutation: RuntimePublicationMutation,
    ) -> dict[str, Any] | None:
        """Return a fresh, validated stored checkpoint after ``mutation``.

        ``current_checkpoint`` is this codec's ``decode`` of ``stored_checkpoint``
        in the caller's transaction, and the caller hands it over: it reads none
        of it afterwards. The result is private to the caller, so the row writer
        need not copy it again.
        """

        if self.encode_mutation is not None:
            if current_checkpoint is None:
                raise TypeError("A decoded publication checkpoint is required.")
            return self.encode_mutation(session, stored_checkpoint, current_checkpoint, mutation)
        encoded = self.encode(session, self.apply_mutation(session, stored_checkpoint, mutation))
        return None if encoded is None else copy_durable_json_object(encoded, "checkpoint")


_RUNTIME_PUBLICATION_CHECKPOINT_CODEC: ContextVar[_RuntimePublicationCheckpointCodec | None] = (
    ContextVar("cayu_runtime_publication_checkpoint_codec", default=None)
)


@contextmanager
def _runtime_publication_checkpoint_codec_scope(
    *,
    decode: CheckpointTransform,
    encode: CheckpointTransform,
    apply_mutation: Callable[
        [Session, dict[str, Any] | None, RuntimePublicationMutation],
        dict[str, Any] | None,
    ],
    encode_mutation: (
        Callable[
            [Session, dict[str, Any] | None, dict[str, Any], RuntimePublicationMutation],
            dict[str, Any] | None,
        ]
        | None
    ) = None,
) -> Iterator[None]:
    """Carry one codec through legacy-compatible public store overrides."""

    codec = _RuntimePublicationCheckpointCodec(
        decode=decode,
        encode=encode,
        apply_mutation=apply_mutation,
        encode_mutation=encode_mutation,
    )
    token = _RUNTIME_PUBLICATION_CHECKPOINT_CODEC.set(codec)
    try:
        yield
    finally:
        _RUNTIME_PUBLICATION_CHECKPOINT_CODEC.reset(token)


def _active_runtime_publication_checkpoint_codec() -> _RuntimePublicationCheckpointCodec | None:
    return _RUNTIME_PUBLICATION_CHECKPOINT_CODEC.get()


INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY = "initial_transcript_pending"


def _initial_transcript_pending_checkpoint(
    session: Session,
    interaction_id: str,
    *,
    checkpoint_transform: CheckpointTransform | None = None,
) -> dict[str, Any]:
    """Create the authority marker held until initial transcript publication."""

    checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY: {
            "version": 1,
            "interaction_id": require_clean_nonblank(interaction_id, "interaction_id"),
        },
    }
    if checkpoint_transform is None:
        return checkpoint
    from cayu.sessions import _checkpoint_preservation as checkpoint_preservation

    transformed = checkpoint_transform(session, checkpoint)
    if type(transformed) is not dict:
        raise TypeError("Initial transcript checkpoint transform must return an object.")
    return (
        checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
            checkpoint,
            copy_durable_json_object(transformed, "checkpoint"),
            session_id=session.id,
        )
    )


def _initial_transcript_pending_interaction_id(
    checkpoint: dict[str, Any] | None,
) -> str | None:
    if checkpoint is None or INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Initial transcript authority marker is invalid.")
    interaction_id = marker.get("interaction_id")
    if type(interaction_id) is not str:
        raise ValueError("Initial transcript authority marker is invalid.")
    return require_clean_nonblank(interaction_id, "interaction_id")


def _checkpoint_after_initial_transcript_publication(
    checkpoint: dict[str, Any] | None,
    *,
    interaction_id: str,
) -> dict[str, Any] | None:
    pending_interaction_id = _initial_transcript_pending_interaction_id(checkpoint)
    if pending_interaction_id is None:
        # Deferred admission is also used after session creation, when there is
        # no environment-derived bootstrap authority to protect.
        return checkpoint
    if pending_interaction_id != interaction_id:
        raise RuntimeError("Initial transcript authority changed before finalization.")
    assert checkpoint is not None
    updated = copy_durable_json_object(checkpoint, "checkpoint")
    updated.pop(INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY)
    return updated or None

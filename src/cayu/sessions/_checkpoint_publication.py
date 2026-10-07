"""Publication schema composition within the caller's native transaction.

Reuse the canonical publication records, checkpoint decoder and writer projection.
Runtime dispatch and the task-local codec scope retain their existing owners.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_durable_json_object
from cayu.sessions.base import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationRequest,
    Session,
    _apply_runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
)


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


def _apply_publication_checkpoint_mutation(
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
                "An older runtime publication cannot remove invocation lifecycle receipt authority."
            )
        decoded_target[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = preserved_lifecycle_receipt
    return decoded_target

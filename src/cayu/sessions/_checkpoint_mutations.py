"""Typed checkpoint mutations and shared compare-and-set validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_durable_json_object, copy_durable_json_value
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY
from cayu.sessions.authority import (
    _canonical_runtime_publication_digest,
    _runtime_publication_json_equal,
)
from cayu.sessions.checkpoints import COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY


class SessionRuntimePublicationConflict(ValueError):
    """A runtime publication identity conflicts with its durable receipt."""


RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS = 128


class RuntimePublicationCheckpointOperation(BaseModel):
    """One independently fenced top-level checkpoint mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(max_length=256)
    expected_value_digest: str | None
    action: Literal["set", "delete"]
    value: Any = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = require_clean_nonblank(value, "checkpoint operation key")
        if value == COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY:
            raise ValueError(
                "Runtime publications cannot mutate completion-result event publication authority."
            )
        if value == MODEL_FAILOVER_CHECKPOINT_KEY:
            raise ValueError("Only model-stage preparation may mutate model failover authority.")
        return value

    @field_validator("expected_value_digest")
    @classmethod
    def validate_expected_value_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("expected_value_digest must be a lowercase SHA-256 digest.")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def copy_value(cls, value: Any) -> Any:
        return copy_durable_json_value(value, "checkpoint operation value")

    @model_validator(mode="after")
    def validate_action(self) -> RuntimePublicationCheckpointOperation:
        if self.action == "delete":
            if self.expected_value_digest is None:
                raise ValueError("A delete checkpoint operation requires a present-value digest.")
            if self.value is not None:
                raise ValueError("A delete checkpoint operation cannot carry a value.")
        return self


def _copy_runtime_publication_checkpoint_operation(
    operation: RuntimePublicationCheckpointOperation | dict[str, Any],
) -> RuntimePublicationCheckpointOperation:
    try:
        if type(operation) is RuntimePublicationCheckpointOperation:
            return RuntimePublicationCheckpointOperation(
                key=operation.key,
                expected_value_digest=operation.expected_value_digest,
                action=operation.action,
                value=operation.value,
            )
        if type(operation) is dict:
            return RuntimePublicationCheckpointOperation.model_validate(operation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication checkpoint operation is malformed.") from exc
    raise TypeError(
        "Checkpoint operations must be RuntimePublicationCheckpointOperation values or objects."
    )


class RuntimePublicationMutation(BaseModel):
    """Deterministic checkpoint patch committed with a publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[RuntimePublicationCheckpointOperation, ...] = Field(default_factory=tuple)

    @field_validator("operations", mode="before")
    @classmethod
    def copy_operations(
        cls,
        value,
    ) -> tuple[RuntimePublicationCheckpointOperation, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("operations must be a list or tuple of checkpoint operations.")
        if len(value) > RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS:
            raise ValueError(
                "operations cannot contain more than "
                f"{RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS} checkpoint mutations."
            )
        copied = tuple(
            sorted(
                (_copy_runtime_publication_checkpoint_operation(operation) for operation in value),
                key=lambda operation: operation.key,
            )
        )
        keys = tuple(operation.key for operation in copied)
        if len(set(keys)) != len(keys):
            raise ValueError("operations cannot mutate the same checkpoint key more than once.")
        return copied


def _copy_runtime_publication_mutation_value(
    mutation: RuntimePublicationMutation | dict[str, Any],
) -> RuntimePublicationMutation:
    try:
        if type(mutation) is RuntimePublicationMutation:
            return RuntimePublicationMutation(
                operations=mutation.operations,
            )
        if type(mutation) is dict:
            return RuntimePublicationMutation.model_validate(mutation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication mutation is malformed.") from exc
    raise TypeError("Runtime publication mutation must be a RuntimePublicationMutation or object.")


def runtime_publication_checkpoint_value_digest(value: Any) -> str:
    """Return the canonical digest used to fence one checkpoint value."""

    return _canonical_runtime_publication_digest(value)


def _apply_runtime_publication_checkpoint_mutation(
    mutation: RuntimePublicationMutation,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    from cayu.collaboration._session_export_store import ROOT_KEY

    if checkpoint is None and not mutation.operations:
        return None
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    for operation in mutation.operations:
        if operation.key == ROOT_KEY:
            raise ValueError("Session export authority requires its publication owner.")
        if operation.key == MODEL_FAILOVER_CHECKPOINT_KEY:
            raise ValueError("Only model-stage preparation may mutate model failover authority.")
        present = operation.key in updated
        if operation.expected_value_digest is None:
            matches = not present
        else:
            matches = present and (
                runtime_publication_checkpoint_value_digest(updated[operation.key])
                == operation.expected_value_digest
            )
        if not matches:
            raise SessionRuntimePublicationConflict(
                "A runtime publication checkpoint key changed before publication: "
                f"{operation.key!r}."
            )
        if operation.action == "set":
            updated[operation.key] = copy_durable_json_value(
                operation.value,
                "checkpoint operation value",
            )
        else:
            del updated[operation.key]
    return updated


def apply_runtime_publication_checkpoint_mutation(
    mutation: RuntimePublicationMutation,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Apply a validated runtime publication mutation to a defensive checkpoint copy."""

    return _apply_runtime_publication_checkpoint_mutation(mutation, checkpoint)


def runtime_publication_checkpoint_mutation(
    source_checkpoint: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
) -> RuntimePublicationMutation:
    """Build a deterministic top-level CAS patch without fencing unrelated keys."""

    if checkpoint is None:
        if source_checkpoint is not None:
            raise ValueError(
                "Runtime publication checkpoint mutations cannot delete the whole checkpoint."
            )
        return RuntimePublicationMutation()
    source = (
        {}
        if source_checkpoint is None
        else copy_durable_json_object(source_checkpoint, "source_checkpoint")
    )
    target = copy_durable_json_object(checkpoint, "checkpoint")
    operations: list[RuntimePublicationCheckpointOperation] = []
    for key in sorted(source.keys() | target.keys()):
        source_present = key in source
        target_present = key in target
        if (
            source_present
            and target_present
            and _runtime_publication_json_equal(
                source[key],
                target[key],
            )
        ):
            continue
        expected_value_digest = (
            runtime_publication_checkpoint_value_digest(source[key]) if source_present else None
        )
        if target_present:
            operations.append(
                RuntimePublicationCheckpointOperation(
                    key=key,
                    expected_value_digest=expected_value_digest,
                    action="set",
                    value=target[key],
                )
            )
        else:
            operations.append(
                RuntimePublicationCheckpointOperation(
                    key=key,
                    expected_value_digest=expected_value_digest,
                    action="delete",
                )
            )
    return RuntimePublicationMutation(operations=tuple(operations))

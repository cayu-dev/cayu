"""Protected session-operation record keys, mutations and compare-and-set rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_NOTIFICATION_OPERATION_KEY_PREFIX,
)
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._interaction_receipts import (
    INTERACTION_TRANSITION_OPERATION_KEY_PREFIX,
    INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX,
)

RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX = "__cayu_runtime_publication_v1__:"


ZERO_WORK_INTERRUPTION_OPERATION_KEY = "__cayu_zero_work_interruption_v1__"


class RuntimePublicationOperationRecordMutation(BaseModel):
    """One CAS-protected session-operation record committed with a publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(max_length=512)
    expected_value_digest: str | None
    value: dict[str, Any]

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = require_clean_nonblank(value, "operation record mutation key")
        _reject_reserved_runtime_publication_key(value, "operation record mutation key")
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
    def copy_value(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "operation record mutation value")


_RuntimePublicationOperationRecordMutationValues = (
    list[RuntimePublicationOperationRecordMutation | dict[str, Any]]
    | tuple[RuntimePublicationOperationRecordMutation | dict[str, Any], ...]
)


def _copy_runtime_publication_operation_record_mutation(
    mutation: RuntimePublicationOperationRecordMutation | dict[str, Any],
) -> RuntimePublicationOperationRecordMutation:
    try:
        if type(mutation) is RuntimePublicationOperationRecordMutation:
            return RuntimePublicationOperationRecordMutation(
                key=mutation.key,
                expected_value_digest=mutation.expected_value_digest,
                value=mutation.value,
            )
        if type(mutation) is dict:
            return RuntimePublicationOperationRecordMutation.model_validate(mutation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication operation record mutation is malformed.") from exc
    raise TypeError(
        "Operation record mutations must be "
        "RuntimePublicationOperationRecordMutation values or objects."
    )


MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX = "__cayu_model_completion_stage_v1__:"


def _reject_reserved_runtime_publication_key(
    value: str, field_name: str, *, browser_control_read: bool = False
) -> str:
    from cayu.collaboration._session_export_store import require_operation_key_access
    from cayu.sessions._argument_continuity import require_private_key_access
    from cayu.sessions._browser_control_checkpoint import require_browser_control_operation_owner
    from cayu.sessions._producer_checkpoint import (
        require_operation_key_access as require_producer_key_access,
    )
    from cayu.sessions._session_continuation_scope import (
        require_operation_key_access as require_continuation_key_access,
    )
    from cayu.sessions._tool_call_replay_scope import (
        require_operation_key_access as require_replay_key_access,
    )

    value = require_clean_nonblank(value, field_name)
    require_operation_key_access(value, read=browser_control_read)
    require_continuation_key_access(value, read=browser_control_read)
    require_producer_key_access(value, read=browser_control_read)
    require_private_key_access(value, read=browser_control_read)
    require_replay_key_access(value, read=browser_control_read)
    if not browser_control_read:
        require_browser_control_operation_owner(value)
    if value == ZERO_WORK_INTERRUPTION_OPERATION_KEY:
        raise ValueError(f"{field_name} cannot use the reserved zero-work interruption key.")
    if value.startswith(RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved runtime publication namespace.")
    if value.startswith(INTERACTION_TRANSITION_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved interaction-transition namespace.")
    if value.startswith(INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved terminal-event namespace.")
    if value.startswith(MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved model-completion stage namespace.")
    if value.startswith(CHILD_SESSION_NOTIFICATION_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved child-notification namespace.")
    return value


def _validate_session_operation_record_keys(records: Mapping[str, Any]) -> None:
    from cayu.collaboration._session_export_store import require_operation_record_owner
    from cayu.sessions._browser_control_checkpoint import require_browser_control_operation_owner
    from cayu.sessions._session_continuation_store import (
        require_operation_record_owner as require_continuation_record_owner,
    )

    for key in records:
        _reject_reserved_runtime_publication_key(key, "operation_records key")
        require_browser_control_operation_owner(key, records[key])
        require_operation_record_owner(key, records[key])
        require_continuation_record_owner(key, records[key])


def runtime_publication_operation_record_value_digest(value: object) -> str:
    """Return the CAS digest used for one session-operation record value."""

    return session_authority_rules._canonical_runtime_publication_digest(
        copy_durable_json_object(value, "operation record value")
    )


def _apply_runtime_publication_operation_record_mutations(
    mutations: tuple[RuntimePublicationOperationRecordMutation, ...],
    records: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    from cayu.collaboration._session_export_store import OPERATION_PREFIX

    updated = dict(records)
    for mutation in mutations:
        if mutation.key.startswith(OPERATION_PREFIX):
            raise ValueError("Session export records require their publication owner.")
        current = updated.get(mutation.key)
        matches = (
            current is None
            if mutation.expected_value_digest is None
            else current is not None
            and runtime_publication_operation_record_value_digest(current)
            == mutation.expected_value_digest
        )
        if not matches:
            raise SessionRuntimePublicationConflict(
                "A runtime publication operation record changed before publication: "
                f"{mutation.key!r}."
            )
        updated[mutation.key] = copy_durable_json_object(
            mutation.value,
            "operation record mutation value",
        )
    return updated

"""Persisted execution-profile records and pure session checkpoint rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    copy_session_metadata,
    require_durable_clean_nonblank,
)
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.sessions.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY

_EXECUTION_PROFILE_RECORD_SCHEMA_VERSION = 1
_ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_METADATA_KEY = "cayu:execution_profile"
_EXECUTION_PROFILE_RECORD_TYPE = "cayu.execution-profile"
_ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE = "cayu.active-invocation-execution-profile"


class ActiveInvocationExecutionProfile(BaseModel):
    """Durable profile authority for one active interaction and run epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.active-invocation-execution-profile"] = (
        _ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE
    )
    schema_version: Literal[1] = _ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION
    session_id: str
    interaction_id: str
    run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    profile: ExecutionProfileIdentity

    @field_validator("session_id", "interaction_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)


def execution_profile_session_metadata(
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Return the bounded runtime-owned record stored with a new session."""

    dumped = profile.model_dump(mode="json")
    return {
        "record_type": _EXECUTION_PROFILE_RECORD_TYPE,
        "schema_version": _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION,
        "baseline": dumped,
        "expected": dumped,
    }


def execution_profile_metadata_after_adoption(
    metadata: Mapping[str, Any],
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Advance the expected profile while retaining the immutable baseline."""

    copied = copy_session_metadata(metadata)
    current = copied.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(current) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    # Validate the complete record before retaining its immutable baseline.
    execution_profile_from_session_metadata(copied)
    current["expected"] = profile.model_dump(mode="json")
    copied[EXECUTION_PROFILE_METADATA_KEY] = current
    return copy_session_metadata(copied)


def execution_profile_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ExecutionProfileIdentity:
    """Load the current expected profile, failing closed on absent/malformed state."""

    raw = metadata.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(raw) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    if set(raw) != {"record_type", "schema_version", "baseline", "expected"}:
        raise ValueError("Session execution-profile metadata is malformed.")
    if (
        raw["record_type"] != _EXECUTION_PROFILE_RECORD_TYPE
        or raw["schema_version"] != _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("Session execution-profile metadata version is unsupported.")
    # Revalidate both identities after every backend round trip. The immutable
    # baseline is durable audit authority even though admission returns the
    # current expectation. No raw component material is stored in either value.
    ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["baseline"], "execution_profile.baseline")
    )
    return ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["expected"], "execution_profile.expected")
    )


def execution_profile_baseline_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ExecutionProfileIdentity:
    """Load the immutable creation baseline from one session profile record."""

    raw = metadata.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(raw) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    if set(raw) != {"record_type", "schema_version", "baseline", "expected"}:
        raise ValueError("Session execution-profile metadata is malformed.")
    if (
        raw["record_type"] != _EXECUTION_PROFILE_RECORD_TYPE
        or raw["schema_version"] != _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("Session execution-profile metadata version is unsupported.")
    # Validate the mutable expectation too. A malformed sibling field must not
    # be bypassed merely because a caller asks only for the immutable baseline.
    ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["expected"], "execution_profile.expected")
    )
    return ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["baseline"], "execution_profile.baseline")
    )


def active_invocation_execution_profile_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> ActiveInvocationExecutionProfile | None:
    """Load the active invocation profile, failing closed on malformed authority."""

    if checkpoint is None:
        return None
    raw = checkpoint.get(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
    if raw is None:
        return None
    return ActiveInvocationExecutionProfile.model_validate(
        copy_durable_json_value(raw, "active_invocation_execution_profile")
    )


def active_invocation_execution_profile_matches_session_epoch(
    snapshot: ActiveInvocationExecutionProfile,
    *,
    session_id: str,
    run_epoch: int,
) -> bool:
    """Return whether active authority can continue from this durable epoch."""

    permitted_epochs = {run_epoch}
    if run_epoch > 0:
        # Releasing a completed, paused run fences that epoch by advancing the
        # session once. The invocation snapshot continues to own the open
        # interaction until a continuation atomically rebinds it.
        permitted_epochs.add(run_epoch - 1)
    return snapshot.session_id == session_id and snapshot.run_epoch in permitted_epochs


def active_invocation_execution_profile_is_released(
    snapshot: ActiveInvocationExecutionProfile,
    *,
    session_id: str,
    run_epoch: int,
) -> bool:
    """Whether trailing work released the exact invocation's durable run fence."""

    return (
        snapshot.session_id == session_id and run_epoch > 0 and snapshot.run_epoch == run_epoch - 1
    )


def checkpoint_with_active_invocation_execution_profile(
    checkpoint: Mapping[str, Any] | None,
    *,
    session_id: str,
    interaction_id: str,
    run_epoch: int,
    profile: ExecutionProfileIdentity,
    expected: ActiveInvocationExecutionProfile | None = None,
) -> dict[str, Any]:
    """Bind one profile to an interaction epoch with optional CAS authority."""

    current = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if expected is not None and current != expected:
        raise RuntimeError("Active invocation execution profile changed before it was claimed.")
    snapshot = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=interaction_id,
        run_epoch=run_epoch,
        profile=profile,
    )
    updated = {} if checkpoint is None else copy_durable_json_value(dict(checkpoint), "checkpoint")
    updated[ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY] = snapshot.model_dump(mode="json")
    return updated

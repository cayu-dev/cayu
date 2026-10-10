"""Persisted execution-profile records and pure session checkpoint rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    copy_session_metadata,
    require_durable_clean_nonblank,
)
from cayu.execution_profiles import ExecutionProfileAdmissionBoundary, ExecutionProfileIdentity
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverProgress,
    ModelFailoverSelection,
    copy_model_failover_state,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    decode_runtime_checkpoint,
)

if TYPE_CHECKING:
    from cayu.sessions.records import Session

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


class SessionInvocationExecutionProfile(BaseModel):
    """Public view of the profile bound to a session's latest invocation.

    ``released`` is true once the invocation gave up its run epoch (the session
    paused or finished). An unreleased invocation is still running, or was
    interrupted and is continued by recovery.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    interaction_id: str
    run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    released: StrictBool
    profile: ExecutionProfileIdentity

    @field_validator("profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)


SessionExecutionProfileIssue = Literal[
    "expected_profile_invalid",
    "active_invocation_profile_invalid",
    "active_invocation_epoch_mismatch",
    "session_not_found",
    "checkpoint_incompatible",
    "load_failed",
]

# Checkpoint records whose work continues the active invocation. Runtime
# admits it only on exact reuse of that invocation's profile.
_CONTINUATION_CHECKPOINT_KEYS = (
    "pending_tool_round",
    "pending_tool_approval",
    "pending_user_input",
    "foreground_child_wait",
    "foreground_parent_continuation",
    # An accepted provider-operation resolution is recovered before other work.
    "provider_operation_pending_resolution_disposition",
)


class SessionExecutionProfiles(BaseModel):
    """Redacted profiles a later invocation of one session must match.

    ``boundary`` is where Runtime meets this session next. ``continuation``
    (pending approval, user input, tool round, child wait, model-completion
    stage or provider-operation resolution, or an unreleased invocation) must
    match ``active_invocation.profile`` exactly. ``resume`` (the last invocation
    released with nothing pending) is compared with ``expected``. ``boundary``
    is ``None`` when the profile it needs is missing or the records conflict;
    a conflict is reported in ``issues``. Both profiles contain only component
    classes, strengths, fingerprints and typed authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    boundary: ExecutionProfileAdmissionBoundary | None = None
    expected: ExecutionProfileIdentity | None = None
    active_invocation: SessionInvocationExecutionProfile | None = None
    issues: tuple[SessionExecutionProfileIssue, ...] = ()

    @field_validator("expected", mode="before")
    @classmethod
    def copy_expected(cls, value: object) -> ExecutionProfileIdentity | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)


def session_execution_profiles(
    session: Session,
    checkpoint: Mapping[str, Any] | None,
    *,
    model_completion_pending: bool = False,
) -> SessionExecutionProfiles:
    """Project a session's stored profiles without failing on one bad record.

    ``model_completion_pending`` says whether the store holds an active
    model-completion stage for the session
    (``SessionStore.load_active_model_completion_stage``). That record lives
    outside the checkpoint, and Runtime continues the invocation that owns it.

    A session created before execution profiles has neither value. A malformed
    record, or an active invocation bound to another session or run epoch, is
    reported in ``issues`` rather than raised, so one damaged session cannot hide
    the others in a listing; Runtime still fails closed on it.
    """

    expected: ExecutionProfileIdentity | None = None
    active: SessionInvocationExecutionProfile | None = None
    issues: list[SessionExecutionProfileIssue] = []
    if session.metadata.get(EXECUTION_PROFILE_METADATA_KEY) is not None:
        try:
            expected = execution_profile_from_session_metadata(session.metadata)
        except (TypeError, ValueError):
            issues.append("expected_profile_invalid")
    try:
        snapshot = active_invocation_execution_profile_from_checkpoint(checkpoint)
    except (TypeError, ValueError):
        issues.append("active_invocation_profile_invalid")
    else:
        if snapshot is not None and not active_invocation_execution_profile_matches_session_epoch(
            snapshot, session_id=session.id, run_epoch=session.run_epoch
        ):
            issues.append("active_invocation_epoch_mismatch")
        elif snapshot is not None:
            active = SessionInvocationExecutionProfile(
                interaction_id=snapshot.interaction_id,
                run_epoch=snapshot.run_epoch,
                released=active_invocation_execution_profile_is_released(
                    snapshot, session_id=session.id, run_epoch=session.run_epoch
                ),
                profile=snapshot.profile,
            )
    boundary: ExecutionProfileAdmissionBoundary | None = None
    if not issues:
        continues = (
            model_completion_pending
            or (active is not None and not active.released)
            or any(
                checkpoint is not None and checkpoint.get(key) is not None
                for key in _CONTINUATION_CHECKPOINT_KEYS
            )
        )
        if continues and active is not None:
            boundary = ExecutionProfileAdmissionBoundary.CONTINUATION
        elif not continues and expected is not None:
            boundary = ExecutionProfileAdmissionBoundary.RESUME
    return SessionExecutionProfiles(
        boundary=boundary,
        expected=expected,
        active_invocation=active,
        issues=tuple(issues),
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


def model_failover_progress_for_session(
    *,
    session: Session,
    execution_profile: ExecutionProfileIdentity,
    checkpoint: dict[str, Any] | None,
) -> ModelFailoverProgress | ModelFailoverSelection | None:
    """Resolve stored selection, not permission to resume or repeat a dispatch.

    The caller owns snapshot provenance and invocation admission. The stage
    transaction must still prove the exact predecessor and its settlement.
    """

    if checkpoint is None or MODEL_FAILOVER_CHECKPOINT_KEY not in checkpoint:
        return None
    current = decode_runtime_checkpoint(checkpoint, session_id=session.id)
    if current is None or MODEL_FAILOVER_CHECKPOINT_KEY not in current:
        return None
    progress = copy_model_failover_state(current[MODEL_FAILOVER_CHECKPOINT_KEY])
    binding = execution_profile.model_failover
    if (
        binding is None
        or progress.session_id != session.id
        or progress.session_instance_id != session.instance_id
        or progress.execution_profile_fingerprint != execution_profile.fingerprint
        or progress.plan != binding.plan
        or progress.source_run_epoch > session.run_epoch
        or progress.plan.candidates[0].provider_name != session.provider_name
        or progress.plan.candidates[0].model != session.model
    ):
        raise ValueError("Stored model selection conflicts with its session/profile authority.")
    return progress

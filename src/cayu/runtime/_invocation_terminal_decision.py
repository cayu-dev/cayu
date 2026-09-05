"""Durable single-winner authority for invocation terminal outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.runtime.checkpoints import (
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)

_DECISION_RECORD_TYPE = "cayu.invocation-terminal-decision"
_DECISION_SCHEMA_VERSION = 1


class InvocationTerminalOutcome(StrEnum):
    """The mutually exclusive terminal outcomes governed by the decision."""

    FAILED = "failed"
    INTERRUPTED = "interrupted"


class InvocationTerminalDecision(BaseModel):
    """Content-bound authority installed before terminal side effects begin."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    record_type: Literal["cayu.invocation-terminal-decision"] = _DECISION_RECORD_TYPE
    schema_version: Literal[1] = _DECISION_SCHEMA_VERSION
    decision_id: str
    outcome: InvocationTerminalOutcome
    session_id: str
    session_instance_id: str
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    profile_interaction_id: str
    interaction_id: str
    execution_profile_fingerprint: str
    interaction_event_id: str | None
    predecessor_interaction_event_id: str | None = None
    terminal_event_id: str
    observed_at: datetime
    terminal_payload: dict[str, Any]
    interruption_request_id: str | None = None
    task_id: str | None = None
    runtime_task_failure_id: str | None = None
    task_terminalization_request_sha256: str | None = None
    task_error_payload: dict[str, Any] | None = None
    turn_completed_payload: dict[str, Any] | None = None
    model_recovery_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^model-recovery:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    record_digest: str

    @field_validator(
        "decision_id",
        "session_id",
        "session_instance_id",
        "profile_interaction_id",
        "interaction_id",
        "interaction_event_id",
        "predecessor_interaction_event_id",
        "terminal_event_id",
        "interruption_request_id",
        "task_id",
        "runtime_task_failure_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("execution_profile_fingerprint", "record_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task_terminalization_request_sha256")
    @classmethod
    def validate_optional_digest(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "observed_at")

    @field_validator("terminal_payload", "task_error_payload", "turn_completed_payload")
    @classmethod
    def copy_payloads(cls, value: dict[str, Any] | None, info: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> InvocationTerminalDecision:
        failure_fields = (
            self.task_id,
            self.runtime_task_failure_id,
            self.task_error_payload,
            self.turn_completed_payload,
        )
        if self.outcome is InvocationTerminalOutcome.INTERRUPTED:
            if (
                self.interruption_request_id is None
                or self.model_recovery_id is not None
                or any(value is not None for value in failure_fields)
                or self.task_terminalization_request_sha256 is not None
            ):
                raise ValueError("Interrupted decisions require only interruption authority.")
        elif self.model_recovery_id is not None:
            if (
                self.interruption_request_id is not None
                or any(value is not None for value in failure_fields)
                or self.task_terminalization_request_sha256 is not None
            ):
                raise ValueError("Model failure decisions cannot substitute task authority.")
        elif self.interruption_request_id is not None or any(
            value is None for value in failure_fields
        ):
            raise ValueError("Failed decisions require complete linked-task failure authority.")
        if self.outcome is InvocationTerminalOutcome.INTERRUPTED:
            assert self.interruption_request_id is not None
            expected_interaction_event_id = (
                None
                if self.predecessor_interaction_event_id is not None
                else invocation_terminal_event_id(
                    outcome=self.outcome,
                    session_id=self.session_id,
                    session_instance_id=self.session_instance_id,
                    run_epoch=self.run_epoch,
                    interaction_id=self.interaction_id,
                    source_id=self.interruption_request_id,
                    event_kind="interaction",
                )
            )
            expected_terminal_event_id = invocation_terminal_event_id(
                outcome=self.outcome,
                session_id=self.session_id,
                session_instance_id=self.session_instance_id,
                run_epoch=self.run_epoch,
                interaction_id=self.interaction_id,
                source_id=self.interruption_request_id,
                event_kind="session",
            )
        else:
            source_id = self.model_recovery_id or self.runtime_task_failure_id
            assert source_id is not None
            expected_interaction_event_id = f"{source_id}:interaction_failed"
            expected_terminal_event_id = f"{source_id}:session_failed"
        if self.outcome is InvocationTerminalOutcome.FAILED and (
            self.predecessor_interaction_event_id is not None
        ):
            raise ValueError("Failed decisions cannot reuse a terminal interaction event.")
        if (self.interaction_event_id is None) != (
            self.predecessor_interaction_event_id is not None
        ):
            raise ValueError("Invocation terminal decision interaction evidence is incomplete.")
        if (
            self.interaction_event_id != expected_interaction_event_id
            or self.terminal_event_id != expected_terminal_event_id
        ):
            raise ValueError(
                "Invocation terminal decision event identities conflict with its authority."
            )
        expected_id = _decision_id(
            self.model_dump(mode="json", exclude={"decision_id", "record_digest"})
        )
        if self.decision_id != expected_id:
            raise ValueError("Invocation terminal decision identity is invalid.")
        expected_digest = _decision_digest(self.model_dump(mode="json", exclude={"record_digest"}))
        if self.record_digest != expected_digest:
            raise ValueError("Invocation terminal decision digest is invalid.")
        return self


def _decision_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        canonical_durable_json_bytes(dict(payload), "invocation_terminal_decision")
    ).hexdigest()


def _decision_id(payload: Mapping[str, Any]) -> str:
    return f"invocation-terminal-decision:v1:{_decision_digest(payload)}"


def invocation_terminal_event_id(
    *,
    outcome: InvocationTerminalOutcome,
    session_id: str,
    session_instance_id: str,
    run_epoch: int,
    interaction_id: str,
    source_id: str,
    event_kind: Literal["interaction", "session"],
) -> str:
    """Return a stable event ID for one candidate terminal decision."""

    digest = _decision_digest(
        {
            "schema": "cayu.invocation-terminal-event-id.v1",
            "outcome": outcome.value,
            "session_id": session_id,
            "session_instance_id": session_instance_id,
            "run_epoch": run_epoch,
            "interaction_id": interaction_id,
            "source_id": source_id,
            "event_kind": event_kind,
        }
    )
    return f"invocation-terminal:v1:{digest}:{event_kind}"


def invocation_terminal_decision_matches_active_profile(
    decision: InvocationTerminalDecision,
    *,
    session_id: str,
    session_instance_id: str,
    run_epoch: int,
    interaction_id: str,
    execution_profile_fingerprint: str,
) -> bool:
    """Return whether one decision is bound to the exact active invocation."""

    if type(decision) is not InvocationTerminalDecision:
        raise TypeError("decision must be an InvocationTerminalDecision.")
    return bool(
        decision.session_id == session_id
        and decision.session_instance_id == session_instance_id
        and decision.run_epoch == run_epoch
        and decision.profile_interaction_id == interaction_id
        and decision.execution_profile_fingerprint == execution_profile_fingerprint
    )


def invocation_terminal_decision_matches_recovery_profile(
    decision: InvocationTerminalDecision,
    *,
    session_id: str,
    session_instance_id: str,
    current_run_epoch: int,
    interaction_id: str,
    execution_profile_fingerprint: str,
) -> bool:
    """Match a decision owned by this epoch or its exact recovery successor."""

    if type(decision) is not InvocationTerminalDecision:
        raise TypeError("decision must be an InvocationTerminalDecision.")
    if type(current_run_epoch) is not int or isinstance(current_run_epoch, bool):
        raise TypeError("current_run_epoch must be an integer.")
    return bool(
        decision.session_id == session_id
        and decision.session_instance_id == session_instance_id
        and decision.run_epoch in {current_run_epoch, current_run_epoch - 1}
        and decision.profile_interaction_id == interaction_id
        and decision.execution_profile_fingerprint == execution_profile_fingerprint
    )


def build_invocation_terminal_decision(
    *,
    outcome: InvocationTerminalOutcome,
    session_id: str,
    session_instance_id: str,
    run_epoch: int,
    profile_interaction_id: str,
    interaction_id: str,
    execution_profile_fingerprint: str,
    interaction_event_id: str | None,
    predecessor_interaction_event_id: str | None = None,
    terminal_event_id: str,
    observed_at: datetime,
    terminal_payload: dict[str, Any],
    interruption_request_id: str | None = None,
    task_id: str | None = None,
    runtime_task_failure_id: str | None = None,
    task_terminalization_request_sha256: str | None = None,
    task_error_payload: dict[str, Any] | None = None,
    turn_completed_payload: dict[str, Any] | None = None,
    model_recovery_id: str | None = None,
) -> InvocationTerminalDecision:
    """Build one self-authenticating immutable decision."""

    payload: dict[str, Any] = {
        "record_type": _DECISION_RECORD_TYPE,
        "schema_version": _DECISION_SCHEMA_VERSION,
        "outcome": outcome.value,
        "session_id": session_id,
        "session_instance_id": session_instance_id,
        "run_epoch": run_epoch,
        "profile_interaction_id": profile_interaction_id,
        "interaction_id": interaction_id,
        "execution_profile_fingerprint": execution_profile_fingerprint,
        "interaction_event_id": interaction_event_id,
        "predecessor_interaction_event_id": predecessor_interaction_event_id,
        "terminal_event_id": terminal_event_id,
        "observed_at": normalize_utc_datetime(observed_at, "observed_at")
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "terminal_payload": copy_durable_json_object(terminal_payload, "terminal_payload"),
        "interruption_request_id": interruption_request_id,
        "task_id": task_id,
        "runtime_task_failure_id": runtime_task_failure_id,
        "task_terminalization_request_sha256": task_terminalization_request_sha256,
        "task_error_payload": (
            None
            if task_error_payload is None
            else copy_durable_json_object(task_error_payload, "task_error_payload")
        ),
        "turn_completed_payload": (
            None
            if turn_completed_payload is None
            else copy_durable_json_object(turn_completed_payload, "turn_completed_payload")
        ),
    }
    if model_recovery_id is not None:
        payload["model_recovery_id"] = model_recovery_id
    identity_payload = {
        key: value for key, value in payload.items() if key not in {"decision_id", "record_digest"}
    }
    payload["decision_id"] = _decision_id(identity_payload)
    payload["record_digest"] = _decision_digest(payload)
    return InvocationTerminalDecision.model_validate(payload)


def invocation_terminal_decision_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> InvocationTerminalDecision | None:
    """Load and authenticate the current terminal winner."""

    if checkpoint is None or INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint:
        return None
    try:
        return InvocationTerminalDecision.model_validate(
            checkpoint[INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY]
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stored invocation terminal decision is invalid.") from exc


def settled_invocation_terminal_decision_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> InvocationTerminalDecision | None:
    """Load the immutable winner retained by paired terminal publication."""

    if checkpoint is None or SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY not in checkpoint:
        return None
    try:
        return InvocationTerminalDecision.model_validate(
            checkpoint[SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY]
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stored settled invocation terminal decision is invalid.") from exc


def checkpoint_with_invocation_terminal_decision(
    checkpoint: dict[str, Any] | None,
    decision: InvocationTerminalDecision,
) -> dict[str, Any]:
    """Install one winner or replay the exact previously installed decision."""

    if type(decision) is not InvocationTerminalDecision:
        raise TypeError("decision must be an InvocationTerminalDecision.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    current = invocation_terminal_decision_from_checkpoint(updated)
    if current is not None and current != decision:
        raise RuntimeError("Another invocation terminal outcome already owns this run.")
    updated[INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY] = decision.model_dump(mode="json")
    updated.pop(SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY, None)
    return updated


def checkpoint_after_invocation_terminal_decision(
    checkpoint: dict[str, Any] | None,
    *,
    expected: InvocationTerminalDecision,
) -> dict[str, Any] | None:
    """Clear only the exact winner after its paired publication commits."""

    if checkpoint is None:
        raise RuntimeError("Invocation terminal decision disappeared before settlement.")
    updated = copy_durable_json_object(checkpoint, "checkpoint")
    current = invocation_terminal_decision_from_checkpoint(updated)
    if current != expected:
        raise RuntimeError("Invocation terminal decision changed before settlement.")
    updated.pop(INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY)
    updated[SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY] = expected.model_dump(mode="json")
    return updated

"""Exact durable interaction and invocation-terminal receipt validation."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType, copy_event, event_envelope_authority_is_runtime_generated
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import recovery as session_recovery_rules
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    checkpoint_after_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_recovery_profile,
)
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ActiveModelCompletionStage,
    ModelCompletionStageDispatch,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._operation_records import (
    INTERACTION_TRANSITION_OPERATION_KEY_PREFIX as INTERACTION_TRANSITION_OPERATION_KEY_PREFIX,
)
from cayu.sessions._operation_records import (
    INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX as INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.records import Session, SessionStatus, copy_session


class _InteractionTransitionReceipt(BaseModel):
    """Immutable store-owned evidence for exact lifecycle-transition replay."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    record_type: Literal["cayu.interaction-transition"] = "cayu.interaction-transition"
    schema_version: Literal[1, 2, 3, 4, 5, 6] = 1
    session: Session
    event: Event
    from_statuses: tuple[SessionStatus, ...]
    to_status: SessionStatus
    only_if_no_queued_messages: StrictBool
    model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None
    checkpoint_mutation: dict[str, Any] | None = None
    terminal_event: Event | None = None
    terminal_decision: InvocationTerminalDecision | None = None
    invocation_session_instance_id: str | None = None
    invocation_active_profile: ActiveInvocationExecutionProfile | None = None
    invocation_authority_state: Literal["active", "released"] | None = None
    recovery_claim_id: str | None = None
    status_changed: StrictBool
    record_digest: str

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("checkpoint_mutation", mode="before")
    @classmethod
    def copy_checkpoint_mutation(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        if type(value) is not dict:
            raise TypeError("checkpoint_mutation must be an object or None.")
        return copy_durable_json_object(value, "checkpoint_mutation")

    @field_validator("terminal_event")
    @classmethod
    def copy_terminal_event(cls, value: Event | None) -> Event | None:
        return None if value is None else copy_event(value)

    @field_validator("record_digest")
    @classmethod
    def validate_record_digest(cls, value: str) -> str:
        session_authority_rules._require_raw_sha256_digest(value)
        return value

    @model_validator(mode="after")
    def validate_source_statuses(self) -> _InteractionTransitionReceipt:
        if not self.from_statuses:
            raise ValueError("Interaction-transition receipt requires source statuses.")
        if len(set(self.from_statuses)) != len(self.from_statuses):
            raise ValueError("Interaction-transition receipt source statuses must be unique.")
        if self.session.id != self.event.session_id:
            raise ValueError("Interaction-transition receipt session and event disagree.")
        if self.status_changed:
            if self.session.status is not self.to_status:
                raise ValueError(
                    "Interaction-transition receipt changed status does not match its session."
                )
        elif not self.only_if_no_queued_messages or self.session.status not in self.from_statuses:
            raise ValueError("Interaction-transition receipt unchanged status is inconsistent.")
        carries_invocation_authority = self.invocation_session_instance_id is not None
        if self.schema_version < 6 and (
            self.terminal_event is not None or self.terminal_decision is not None
        ):
            raise ValueError("Interaction-transition receipt version conflicts with terminality.")
        if carries_invocation_authority != (self.invocation_active_profile is not None):
            raise ValueError("Interaction-transition invocation authority is incomplete.")
        if self.schema_version == 1:
            if carries_invocation_authority or self.invocation_authority_state is not None:
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
            if self.recovery_claim_id is not None:
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
        elif self.schema_version == 2:
            if not carries_invocation_authority or self.invocation_authority_state is not None:
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
            if self.recovery_claim_id is not None:
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
        elif self.schema_version == 3:
            if (
                not carries_invocation_authority
                or self.invocation_authority_state is None
                or self.recovery_claim_id is not None
            ):
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
        elif self.schema_version == 4:
            if self.recovery_claim_id is None or (
                carries_invocation_authority != (self.invocation_authority_state is not None)
            ):
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
            require_clean_nonblank(self.recovery_claim_id, "recovery_claim_id")
        elif self.schema_version == 5:
            if (
                self.checkpoint_mutation is None
                or self.recovery_claim_id is not None
                or (not carries_invocation_authority or self.invocation_authority_state != "active")
                or self.terminal_event is not None
                or self.terminal_decision is not None
            ):
                raise ValueError("Interaction-transition receipt version conflicts with mutation.")
        else:
            if (self.terminal_event is None) != (self.terminal_decision is None):
                raise ValueError("Terminal interaction receipt authority is incomplete.")
            if self.terminal_event is None:
                raise ValueError("Version 6 interaction receipt requires terminal evidence.")
            if self.checkpoint_mutation is not None:
                raise ValueError("Terminal interaction receipt conflicts with mutation.")
            if carries_invocation_authority != (self.invocation_authority_state is not None):
                raise ValueError("Interaction-transition receipt version conflicts with authority.")
            if self.recovery_claim_id is not None:
                require_clean_nonblank(self.recovery_claim_id, "recovery_claim_id")
            InteractionTransitionSpec(
                event=self.event,
                from_statuses=self.from_statuses,
                to_status=self.to_status,
                only_if_no_queued_messages=self.only_if_no_queued_messages,
                model_completion_stage_settlement=self.model_completion_stage_settlement,
                terminal_event=self.terminal_event,
                terminal_decision=self.terminal_decision,
            )
        if self.schema_version < 5 and self.checkpoint_mutation is not None:
            raise ValueError("Interaction-transition receipt version conflicts with mutation.")
        if carries_invocation_authority:
            SessionInvocationBinding.validate_session_instance_id(
                self.invocation_session_instance_id
            )
            active = self.invocation_active_profile
            if active is None:  # pragma: no cover - paired-field validation above
                raise AssertionError("Interaction-transition authority disappeared.")
            authority_state = self.invocation_authority_state or "active"
            expected_session_epoch = active.run_epoch + (authority_state == "released")
            if (
                self.session.instance_id != self.invocation_session_instance_id
                or active.session_id != self.session.id
                or expected_session_epoch != self.session.run_epoch
            ):
                raise ValueError("Interaction-transition invocation authority conflicts.")
        return self


class _InvocationTerminalEventReceipt(BaseModel):
    """Immutable binding from one terminal event to its invocation authority."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    record_type: Literal["cayu.invocation-terminal-event"] = "cayu.invocation-terminal-event"
    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    active_profile: ActiveInvocationExecutionProfile
    event: Event
    record_digest: str

    @field_validator("session_id", "session_instance_id")
    @classmethod
    def validate_identifier(cls, value: str, info: Any) -> str:
        if info.field_name == "session_instance_id":
            return SessionInvocationBinding.validate_session_instance_id(value)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("event")
    @classmethod
    def copy_terminal_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("record_digest")
    @classmethod
    def validate_record_digest(cls, value: str) -> str:
        session_authority_rules._require_raw_sha256_digest(value)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> _InvocationTerminalEventReceipt:
        if (
            self.event.session_id != self.session_id
            or self.event.interaction_id not in {None, self.active_profile.interaction_id}
            or self.event.type
            not in {
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }
        ):
            raise ValueError("Invocation terminal-event receipt has invalid event scope.")
        if self.active_profile.session_id != self.session_id:
            raise ValueError("Invocation terminal-event receipt has invalid profile scope.")
        return self


def _interaction_transition_storage_key(event_id: str) -> str:
    event_id = require_clean_nonblank(event_id, "event.id")
    return (
        INTERACTION_TRANSITION_OPERATION_KEY_PREFIX + sha256(event_id.encode("utf-8")).hexdigest()
    )


def _invocation_terminal_event_storage_key(event_id: str) -> str:
    event_id = require_clean_nonblank(event_id, "event.id")
    return (
        INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX
        + sha256(event_id.encode("utf-8")).hexdigest()
    )


def _invocation_terminal_event_receipt_record(
    *,
    session: Session,
    checkpoint: Mapping[str, Any] | None,
    event: Event,
) -> tuple[str, dict[str, Any]] | None:
    if event.type not in {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }:
        return None
    if not event_envelope_authority_is_runtime_generated(
        event,
        field_name="session_id",
        value=session.id,
    ):
        return None
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        active_profile is None
        or active_profile.session_id != session.id
        or active_profile.run_epoch != session.run_epoch
    ):
        return None
    payload = {
        "record_type": "cayu.invocation-terminal-event",
        "schema_version": 1,
        "session_id": session.id,
        "session_instance_id": session.instance_id,
        "active_profile": active_profile.model_dump(mode="json"),
        "event": event.model_dump(mode="json"),
    }
    receipt = _InvocationTerminalEventReceipt(
        session_id=session.id,
        session_instance_id=session.instance_id,
        active_profile=active_profile,
        event=event,
        record_digest=session_authority_rules._canonical_runtime_publication_digest(payload),
    )
    return (
        _invocation_terminal_event_storage_key(event.id),
        receipt.model_dump(mode="json"),
    )


def _load_invocation_terminal_event_receipt(
    record: object,
) -> _InvocationTerminalEventReceipt:
    try:
        receipt = _InvocationTerminalEventReceipt.model_validate(record)
        digest_payload = receipt.model_dump(mode="json", exclude={"record_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != receipt.record_digest
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stored invocation terminal-event receipt is invalid.") from exc
    return receipt


def _validate_interaction_transition_invocation_authority_parameters(
    *,
    expected_session_instance_id: str | None,
    expected_active_invocation_profile: ActiveInvocationExecutionProfile | None,
    expected_invocation_authority_state: object,
) -> Literal["active", "released"]:
    if type(expected_invocation_authority_state) is not str or (
        expected_invocation_authority_state not in {"active", "released"}
    ):
        raise TypeError("expected_invocation_authority_state must be active or released.")
    if (expected_session_instance_id is None) != (expected_active_invocation_profile is None):
        raise TypeError(
            "Interaction settlement authority requires both session incarnation and active profile."
        )
    if (
        expected_active_invocation_profile is None
        and expected_invocation_authority_state != "active"
    ):
        raise TypeError("Released interaction authority requires an invocation profile.")
    return cast('Literal["active", "released"]', expected_invocation_authority_state)


def _validate_interaction_transition_recovery_claim_id(
    expected_recovery_claim_id: str | None,
) -> str | None:
    """Validate the identity used for a store-time recovery-claim fence."""

    if expected_recovery_claim_id is None:
        return None
    return require_clean_nonblank(
        expected_recovery_claim_id,
        "expected_recovery_claim_id",
    )


def _interaction_transition_receipt_record(
    *,
    session: Session,
    event: Event,
    from_statuses: set[SessionStatus],
    to_status: SessionStatus,
    only_if_no_queued_messages: bool,
    model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None,
    checkpoint_mutation: dict[str, Any] | None = None,
    terminal_event: Event | None = None,
    terminal_decision: InvocationTerminalDecision | None = None,
    status_changed: bool,
    invocation_session_instance_id: str | None = None,
    invocation_active_profile: ActiveInvocationExecutionProfile | None = None,
    invocation_authority_state: Literal["active", "released"] = "active",
    recovery_claim_id: str | None = None,
) -> dict[str, Any]:
    if (invocation_session_instance_id is None) != (invocation_active_profile is None):
        raise TypeError("Interaction-transition invocation authority is incomplete.")
    if invocation_active_profile is None and invocation_authority_state != "active":
        raise TypeError("Released interaction authority requires an invocation profile.")
    if terminal_event is not None:
        if terminal_decision is None:
            raise TypeError("Terminal interaction receipt requires its decision.")
        schema_version = 6
    elif checkpoint_mutation is not None:
        schema_version = 5
    elif recovery_claim_id is not None:
        recovery_claim_id = require_clean_nonblank(recovery_claim_id, "recovery_claim_id")
        schema_version = 4
    else:
        schema_version = (
            3
            if invocation_authority_state == "released"
            else (2 if invocation_active_profile is not None else 1)
        )
    ordered_from_statuses = tuple(sorted(from_statuses, key=str))
    payload: dict[str, Any] = {
        "record_type": "cayu.interaction-transition",
        "schema_version": schema_version,
        "session": session.model_dump(mode="json"),
        "event": event.model_dump(mode="json"),
        "from_statuses": [str(status) for status in ordered_from_statuses],
        "to_status": str(to_status),
        "only_if_no_queued_messages": only_if_no_queued_messages,
        "status_changed": status_changed,
    }
    if model_completion_stage_settlement is not None:
        payload["model_completion_stage_settlement"] = model_completion_stage_settlement.model_dump(
            mode="json"
        )
    if checkpoint_mutation is not None:
        payload["checkpoint_mutation"] = copy_durable_json_object(
            checkpoint_mutation,
            "checkpoint_mutation",
        )
    if terminal_event is not None:
        assert terminal_decision is not None
        payload["terminal_event"] = terminal_event.model_dump(mode="json")
        payload["terminal_decision"] = terminal_decision.model_dump(mode="json")
    if invocation_active_profile is not None:
        payload["invocation_session_instance_id"] = invocation_session_instance_id
        payload["invocation_active_profile"] = invocation_active_profile.model_dump(mode="json")
    if schema_version == 3 or (
        schema_version in {4, 5, 6} and invocation_active_profile is not None
    ):
        payload["invocation_authority_state"] = invocation_authority_state
    if recovery_claim_id is not None:
        payload["recovery_claim_id"] = recovery_claim_id
    receipt = _InteractionTransitionReceipt(
        schema_version=schema_version,
        session=session,
        event=event,
        from_statuses=ordered_from_statuses,
        to_status=to_status,
        only_if_no_queued_messages=only_if_no_queued_messages,
        model_completion_stage_settlement=model_completion_stage_settlement,
        checkpoint_mutation=checkpoint_mutation,
        terminal_event=terminal_event,
        terminal_decision=terminal_decision,
        invocation_session_instance_id=invocation_session_instance_id,
        invocation_active_profile=invocation_active_profile,
        invocation_authority_state=(
            invocation_authority_state
            if schema_version == 3
            or (schema_version in {4, 5, 6} and invocation_active_profile is not None)
            else None
        ),
        recovery_claim_id=recovery_claim_id,
        status_changed=status_changed,
        record_digest=session_authority_rules._canonical_runtime_publication_digest(payload),
    )
    return receipt.model_dump(mode="json")


def _load_interaction_transition_receipt(record: object) -> _InteractionTransitionReceipt:
    try:
        receipt = _InteractionTransitionReceipt.model_validate(record)
        digest_payload = receipt.model_dump(mode="json", exclude={"record_digest"})
        if receipt.model_completion_stage_settlement is None:
            # Schema-v1 receipts predate the optional settlement field. Keep
            # their exact digest valid while binding the field whenever set.
            digest_payload.pop("model_completion_stage_settlement", None)
        if receipt.invocation_active_profile is None:
            digest_payload.pop("invocation_session_instance_id", None)
            digest_payload.pop("invocation_active_profile", None)
            digest_payload.pop("invocation_authority_state", None)
        elif receipt.schema_version < 3:
            digest_payload.pop("invocation_authority_state", None)
        if receipt.schema_version < 4 or receipt.recovery_claim_id is None:
            digest_payload.pop("recovery_claim_id", None)
        if receipt.schema_version < 5 or receipt.checkpoint_mutation is None:
            digest_payload.pop("checkpoint_mutation", None)
        if receipt.schema_version < 6:
            digest_payload.pop("terminal_event", None)
            digest_payload.pop("terminal_decision", None)
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != receipt.record_digest
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stored interaction-transition receipt is invalid.") from exc
    return receipt


def _validate_interaction_transition_receipt_authority(
    receipt: _InteractionTransitionReceipt,
    *,
    current_session: Session,
    current_checkpoint: dict[str, Any] | None,
    expected_session_instance_id: str | None,
    expected_active_invocation_profile: ActiveInvocationExecutionProfile | None,
    expected_invocation_authority_state: Literal["active", "released"] = "active",
    expected_recovery_claim_id: str | None = None,
) -> None:
    if (
        current_session.id != receipt.session.id
        or current_session.instance_id != receipt.session.instance_id
    ):
        raise SessionRunFenced(
            "Interaction-transition receipt belongs to another session incarnation."
        )
    expects_authority = expected_active_invocation_profile is not None
    if expects_authority != (expected_session_instance_id is not None):
        raise TypeError("Interaction-transition invocation authority is incomplete.")
    if not expects_authority and expected_invocation_authority_state != "active":
        raise TypeError("Released interaction authority requires an invocation profile.")
    _validate_interaction_transition_receipt_recovery_authority(
        receipt,
        current_checkpoint=current_checkpoint,
        expected_recovery_claim_id=expected_recovery_claim_id,
    )
    receipt_has_authority = receipt.invocation_active_profile is not None
    if expects_authority != receipt_has_authority:
        raise SessionRunFenced(
            "Interaction-transition receipt belongs to another authority contract."
        )
    if expects_authority and (
        receipt.invocation_session_instance_id != expected_session_instance_id
        or receipt.invocation_active_profile != expected_active_invocation_profile
        or (receipt.invocation_authority_state or "active") != expected_invocation_authority_state
    ):
        raise SessionRunFenced(
            "Interaction-transition receipt belongs to another invocation authority."
        )
    if (
        expects_authority
        and expected_invocation_authority_state == "released"
        and current_session.run_epoch < receipt.session.run_epoch
    ):
        raise SessionRunFenced(
            "Interaction-transition receipt is ahead of the current invocation epoch."
        )
    if (
        expects_authority
        and expected_active_invocation_profile is not None
        and expected_invocation_authority_state == "active"
        and current_session.run_epoch != expected_active_invocation_profile.run_epoch
    ):
        # A later exact release makes the settlement receipt permanently
        # replayable.  A bare run-fence transfer does not: it proves that this
        # caller lost mutation authority and must retain the earlier failed
        # attempt as causal evidence.
        from cayu.sessions._invocation_lifecycle import (
            ReleaseInvocationCommand,
            invocation_release_replay_from_state,
        )

        assert expected_session_instance_id is not None
        release = ReleaseInvocationCommand(
            session_id=current_session.id,
            expected_session_instance_id=expected_session_instance_id,
            expected_run_epoch=expected_active_invocation_profile.run_epoch,
            expected_active_profile=expected_active_invocation_profile,
            settlement_transition=_interaction_transition_spec_from_receipt(receipt),
        )
        if (
            invocation_release_replay_from_state(
                current_session,
                current_checkpoint,
                release,
            )
            is None
        ):
            raise SessionRunFenced(
                "Interaction-transition receipt lost its active invocation epoch."
            )


def _validate_interaction_transition_receipt_recovery_authority(
    receipt: _InteractionTransitionReceipt,
    *,
    current_checkpoint: dict[str, Any] | None,
    expected_recovery_claim_id: str | None,
) -> None:
    """Bind mutation replay and readback to the same recovery claimant."""

    if receipt.recovery_claim_id != expected_recovery_claim_id:
        raise SessionRunFenced(
            "Interaction-transition receipt belongs to another recovery authority."
        )
    if expected_recovery_claim_id is not None:
        current_claim = session_recovery_rules._incomplete_recovery_claim_from_checkpoint(
            current_checkpoint
        )
        if current_claim is not None and current_claim[0] != expected_recovery_claim_id:
            raise SessionRunFenced(
                "Interaction-transition receipt was superseded by another recovery authority."
            )


def _interaction_transition_spec_from_receipt(
    receipt: _InteractionTransitionReceipt,
) -> InteractionTransitionSpec:
    return InteractionTransitionSpec(
        event=receipt.event,
        from_statuses=receipt.from_statuses,
        to_status=receipt.to_status,
        only_if_no_queued_messages=receipt.only_if_no_queued_messages,
        model_completion_stage_settlement=(receipt.model_completion_stage_settlement),
        checkpoint_mutation=receipt.checkpoint_mutation,
        terminal_event=receipt.terminal_event,
        terminal_decision=receipt.terminal_decision,
    )


def _validate_interaction_transition_receipt_spec(
    receipt: _InteractionTransitionReceipt,
    transition: InteractionTransitionSpec,
) -> None:
    if _interaction_transition_spec_from_receipt(receipt) != transition:
        raise ValueError("Interaction transition identity was reused with different data.")


def _reconstruct_interaction_transition_receipt(
    record: object,
    *,
    transition: InteractionTransitionSpec,
) -> _InteractionTransitionReceipt:
    receipt = _load_interaction_transition_receipt(record)
    _validate_interaction_transition_receipt_spec(receipt, transition)
    return receipt


def _require_invocation_release_settlement_record(
    record: object,
    *,
    current_session: Session,
    transition: InteractionTransitionSpec,
    expected_session_instance_id: str,
    expected_active_invocation_profile: ActiveInvocationExecutionProfile,
) -> None:
    """Validate release settlement while the store owns the mutation lock."""

    receipt = _reconstruct_interaction_transition_receipt(
        record,
        transition=transition,
    )
    _validate_invocation_release_settlement_receipt_authority(
        receipt,
        current_session=current_session,
        expected_session_instance_id=expected_session_instance_id,
        expected_active_invocation_profile=expected_active_invocation_profile,
    )
    if receipt.status_changed:
        completion_finalization_committed = (
            transition.to_status is SessionStatus.RUNNING
            and transition.event.type == EventType.INTERACTION_COMPLETED
            and transition.checkpoint_mutation is not None
            and receipt.session.status is SessionStatus.RUNNING
            and current_session.status is SessionStatus.COMPLETED
        )
        settlement_status_matches = completion_finalization_committed or (
            receipt.session.status is transition.to_status
            and current_session.status is transition.to_status
        )
    else:
        # A conditional successful interaction is durably terminal even when
        # queued input deliberately keeps the session RUNNING for its next
        # interaction.  The immutable receipt, rather than a terminal session
        # status, is the positive settlement evidence in that branch.
        settlement_status_matches = (
            transition.only_if_no_queued_messages
            and receipt.session.status in transition.from_statuses
            and current_session.status is receipt.session.status
        )
    if (
        not settlement_status_matches
        or current_session.run_epoch != expected_active_invocation_profile.run_epoch
    ):
        raise SessionRunFenced("Invocation release lacks exact durable terminal settlement.")


def _require_invocation_release_recovery_claim(
    checkpoint: Mapping[str, Any] | None,
    *,
    current_session: Session,
    recovery_claim_id: str,
) -> None:
    """Authenticate a completed recovery owner's exact durable claim."""

    claim = session_recovery_rules._incomplete_recovery_claim_from_checkpoint(checkpoint)
    if (
        claim is None
        or claim[0] != recovery_claim_id
        or current_session.status
        not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }
    ):
        raise SessionRunFenced(
            "Invocation recovery release lacks its exact terminal recovery claim."
        )


def _require_invocation_release_terminal_session_event(
    record: object | None,
    retained_event: Event | None,
    *,
    current_session: Session,
    expected_event: Event,
    expected_session_instance_id: str,
    expected_active_invocation_profile: ActiveInvocationExecutionProfile,
) -> None:
    """Authenticate exact terminal session evidence for quiescent cleanup."""

    if record is None:
        raise SessionRunFenced("Invocation release lacks exact durable terminal session evidence.")
    receipt = _load_invocation_terminal_event_receipt(record)
    if (
        receipt.session_id != current_session.id
        or receipt.session_instance_id != current_session.instance_id
        or receipt.session_instance_id != expected_session_instance_id
        or receipt.active_profile != expected_active_invocation_profile
        or current_session.run_epoch != expected_active_invocation_profile.run_epoch
    ):
        raise SessionRunFenced("Terminal session event belongs to another invocation authority.")
    expected_type = {
        SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
        SessionStatus.FAILED: EventType.SESSION_FAILED,
        SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
    }.get(current_session.status)
    if (
        retained_event != expected_event
        or receipt.event.model_dump(mode="json") != expected_event.model_dump(mode="json")
        or expected_event.type is not expected_type
    ):
        raise SessionRunFenced("Invocation release lacks its exact terminal session evidence.")


def _validate_invocation_release_settlement_receipt_authority(
    receipt: _InteractionTransitionReceipt,
    *,
    current_session: Session,
    expected_session_instance_id: str,
    expected_active_invocation_profile: ActiveInvocationExecutionProfile,
) -> None:
    """Bind settlement to one interaction across typed recovery rebind epochs."""

    receipt_active = receipt.invocation_active_profile
    if (
        receipt.invocation_session_instance_id != expected_session_instance_id
        or current_session.id != receipt.session.id
        or current_session.instance_id != expected_session_instance_id
        or receipt.session.instance_id != expected_session_instance_id
        or receipt_active is None
        or receipt_active.session_id != expected_active_invocation_profile.session_id
        or receipt_active.interaction_id != expected_active_invocation_profile.interaction_id
        or receipt_active.profile != expected_active_invocation_profile.profile
        or receipt_active.run_epoch != receipt.session.run_epoch
        or receipt_active.run_epoch > expected_active_invocation_profile.run_epoch
        or current_session.run_epoch != expected_active_invocation_profile.run_epoch
    ):
        raise SessionRunFenced(
            "Interaction-transition receipt belongs to another invocation authority."
        )


def _checkpoint_after_exact_invocation_terminal_decision(
    checkpoint: dict[str, Any] | None,
    *,
    session: Session,
    expected: InvocationTerminalDecision | None,
) -> dict[str, Any] | None:
    """Authenticate and retire a paired terminal-publication winner."""

    current = invocation_terminal_decision_from_checkpoint(checkpoint)
    if expected is None:
        if current is not None:
            raise SessionRunFenced("Interaction transition is owned by another terminal decision.")
        return checkpoint
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        current != expected
        or active_profile is None
        or active_profile.session_id != session.id
        or active_profile.run_epoch != session.run_epoch
        or not invocation_terminal_decision_matches_recovery_profile(
            expected,
            session_id=session.id,
            session_instance_id=session.instance_id,
            current_run_epoch=session.run_epoch,
            interaction_id=active_profile.interaction_id,
            execution_profile_fingerprint=active_profile.profile.fingerprint,
        )
    ):
        raise SessionRunFenced("Interaction transition lost its exact terminal decision authority.")
    return checkpoint_after_invocation_terminal_decision(
        checkpoint,
        expected=expected,
    )


def _checkpoint_after_queued_interaction_profile_handoff(
    session: Session,
    checkpoint: dict[str, Any] | None,
    handoff: QueuedInteractionProfileHandoff,
    *,
    settlement_record: object,
    replayed_delivery: bool,
    active_model_stage: ActiveModelCompletionStage | None = None,
    stage_dispatch: ModelCompletionStageDispatch | None = None,
) -> dict[str, Any]:
    """CAS A to B only from the exact queue-conditional settlement receipt."""

    from cayu.runtime._producer_output_store import producer_owns_invocation

    if producer_owns_invocation(checkpoint):
        raise SessionRunFenced("Queued successor requires settled producer responsibility.")

    expected = handoff.expected_active_profile
    target = handoff.target_active_profile
    if (
        session.id != expected.session_id
        or session.instance_id != handoff.expected_session_instance_id
        or session.run_epoch != expected.run_epoch
    ):
        raise SessionRunFenced("Queued interaction handoff lost its session run authority.")
    receipt = _load_interaction_transition_receipt(settlement_record)
    _validate_interaction_transition_receipt_authority(
        receipt,
        current_session=session,
        current_checkpoint=checkpoint,
        expected_session_instance_id=handoff.expected_session_instance_id,
        expected_active_invocation_profile=expected,
    )
    if (
        receipt.event.id != handoff.predecessor_settlement_event_id
        or receipt.event.type is not EventType.INTERACTION_COMPLETED
        or receipt.event.interaction_id != expected.interaction_id
        or not receipt.only_if_no_queued_messages
        or receipt.to_status not in {SessionStatus.COMPLETED, SessionStatus.RUNNING}
        or receipt.status_changed
        or receipt.terminal_event is not None
    ):
        raise SessionRunFenced(
            "Queued interaction handoff lacks the exact predecessor settlement receipt."
        )
    if receipt.to_status is SessionStatus.RUNNING:
        # A completion-critical predecessor remains RUNNING only to install
        # its finalization marker at an empty-queue boundary. Reconstruct the
        # full transition contract: it requires exactly one unconditional set
        # of pending_completion_finalization, not an arbitrary checkpoint write.
        # status_changed=False above proves that this mutation was not applied.
        try:
            transition = _interaction_transition_spec_from_receipt(receipt)
            if transition.checkpoint_mutation is None:
                raise ValueError
            RuntimePublicationMutation.model_validate(transition.checkpoint_mutation)
        except (TypeError, ValueError):
            raise SessionRunFenced(
                "Queued interaction handoff lacks a completion-finalization transition."
            ) from None
    current = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if current == target:
        if not replayed_delivery:
            raise SessionRunFenced("Queued interaction handoff was already committed elsewhere.")
        assert checkpoint is not None
        return copy_durable_json_object(checkpoint, "checkpoint")
    if session.status is not SessionStatus.RUNNING:
        raise SessionRunFenced("Queued interaction handoff lost its running session authority.")
    if current != expected:
        raise SessionRunFenced("Queued interaction handoff lost its active-profile CAS.")
    if replayed_delivery:
        _validate_historical_queued_interaction_profile_handoff_stage(
            handoff,
            active_model_stage=active_model_stage,
            stage_dispatch=stage_dispatch,
        )
    return checkpoint_with_active_invocation_execution_profile(
        checkpoint,
        session_id=session.id,
        interaction_id=target.interaction_id,
        run_epoch=target.run_epoch,
        profile=target.profile,
        expected=expected,
    )


def _validate_historical_queued_interaction_profile_handoff_stage(
    handoff: QueuedInteractionProfileHandoff,
    *,
    active_model_stage: ActiveModelCompletionStage | None,
    stage_dispatch: ModelCompletionStageDispatch | None,
) -> None:
    """Require exact dispatched B authority before repairing a historical split."""

    target = handoff.target_active_profile
    if active_model_stage is None or stage_dispatch is None:
        raise SessionRunFenced(
            "Historical queued interaction handoff lacks an active dispatched target stage."
        )
    stage = active_model_stage.stage
    try:
        model_stage_rules._validate_model_completion_stage_dispatch(stage_dispatch, stage)
    except SessionModelCompletionStageConflict as exc:
        raise SessionRunFenced(
            "Historical queued interaction handoff has conflicting target dispatch evidence."
        ) from exc
    if (
        stage.session_id != target.session_id
        or stage.source_run_epoch != target.run_epoch
        or stage.intent.get("interaction_id") != target.interaction_id
        or model_stage_rules._model_completion_stage_execution_profile_fingerprint(stage)
        != target.profile.fingerprint
        or stage_dispatch.interaction_id != target.interaction_id
        or stage_dispatch.source_run_epoch != target.run_epoch
        or stage_dispatch.execution_profile_fingerprint != target.profile.fingerprint
    ):
        raise SessionRunFenced(
            "Historical queued interaction handoff target stage lacks exact profile authority."
        )


def _historical_queued_handoff_stage_from_records(
    session_id: str,
    operation_records: Mapping[str, Any],
) -> tuple[ActiveModelCompletionStage | None, ModelCompletionStageDispatch | None]:
    """Reconstruct one active stage and dispatch from a store-atomic record snapshot."""

    active_record = operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
    if active_record is None:
        return None, None
    marker = model_stage_rules._reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session_id,
    )
    _, _, preparation_key, terminal_key = (
        model_stage_rules._model_completion_stage_storage_identity(
            session_id,
            marker.stage_id,
        )
    )
    active = model_stage_rules._reconstruct_active_model_completion_stage(
        active_record,
        operation_records.get(preparation_key),
        operation_records.get(terminal_key),
        session_id=session_id,
    )
    dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(marker.stage_id)
    dispatch_record = operation_records.get(dispatch_key)
    dispatch = (
        None
        if dispatch_record is None
        else model_stage_rules._reconstruct_model_completion_stage_dispatch(
            dispatch_record,
            session_id=session_id,
            stage_id=marker.stage_id,
            storage_key=dispatch_key,
        )
    )
    return active, dispatch

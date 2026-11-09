"""Writes that must happen together, carried by one journaled intent.

When a crash between two durable writes would leave a state nothing repairs,
the intent of the whole pair is installed in the same store transaction as the
first write and settled in the same transaction as the last one. Every write
owed in between is replay-safe and keyed by the intent, so its owner, or
recovery once the owner is gone, finishes the pair from durable state alone.
The intent stays in the session checkpoint until it settles, which is what
recovery lists.

Every terminal interaction outcome is such a pair. A version-2 invocation
terminal decision is installed by the outcome's own transition (COMPLETED,
FAILED or INTERRUPTED), as a set-if-absent checkpoint mutation that every store
applies only when the session reaches that status. The fence release settles
it, and only once the outcome's session event is durable.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_durable_json_object
from cayu.events import Event, EventType
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import (
    _OWED_DECISION_SCHEMA_VERSION,
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
from cayu.sessions.records import Session, SessionStatus

# The session status each owed outcome's transition commits, and its session event.
OUTCOME_SESSION_STATUS: dict[InvocationTerminalOutcome, SessionStatus] = {
    InvocationTerminalOutcome.COMPLETED: SessionStatus.COMPLETED,
    InvocationTerminalOutcome.FAILED: SessionStatus.FAILED,
    InvocationTerminalOutcome.INTERRUPTED: SessionStatus.INTERRUPTED,
}
OUTCOME_SESSION_EVENT: dict[InvocationTerminalOutcome, EventType] = {
    InvocationTerminalOutcome.COMPLETED: EventType.SESSION_COMPLETED,
    InvocationTerminalOutcome.FAILED: EventType.SESSION_FAILED,
    InvocationTerminalOutcome.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
OWED_OUTCOME_STATUSES = frozenset(OUTCOME_SESSION_STATUS.values())


def owed_decision_install_mutation(decision: InvocationTerminalDecision) -> dict[str, Any]:
    """Return the checkpoint mutation that installs a decision with its transition."""

    if decision.schema_version != _OWED_DECISION_SCHEMA_VERSION:
        raise ValueError("Only a version-2 decision rides on its terminal transition.")
    return RuntimePublicationMutation(
        operations=(
            RuntimePublicationCheckpointOperation(
                key=INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                expected_value_digest=None,
                action="set",
                value=decision.model_dump(mode="json"),
            ),
        )
    ).model_dump(mode="json")


def is_owed_decision_install_mutation(mutation: dict[str, Any]) -> bool:
    """Whether a transition's checkpoint mutation installs a terminal decision."""

    operations = RuntimePublicationMutation.model_validate(mutation).operations
    return any(
        operation.key == INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY for operation in operations
    )


def owed_decision_from_install_mutation(
    mutation: dict[str, Any],
    *,
    event: Event,
    to_status: SessionStatus,
) -> InvocationTerminalDecision:
    """Authenticate a terminal transition's mutation as the decision for its own event."""

    operations = RuntimePublicationMutation.model_validate(mutation).operations
    if (
        len(operations) != 1
        or operations[0].key != INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
        or operations[0].action != "set"
        or operations[0].expected_value_digest is not None
    ):
        raise ValueError("A terminal transition may only install its own decision.")
    decision = InvocationTerminalDecision.model_validate(operations[0].value)
    if (
        decision.schema_version != _OWED_DECISION_SCHEMA_VERSION
        or OUTCOME_SESSION_STATUS[decision.outcome] is not to_status
        or decision.interaction_event_id != event.id
        or decision.interaction_id != event.interaction_id
        or decision.session_id != event.session_id
        or decision.observed_at != event.timestamp
    ):
        raise ValueError("A terminal decision belongs to another interaction outcome.")
    return decision


def unsettled_owed_decision(
    checkpoint: dict[str, Any] | None,
) -> InvocationTerminalDecision | None:
    """Return the version-2 decision whose owed writes are not all durable yet."""

    decision = invocation_terminal_decision_from_checkpoint(checkpoint)
    if decision is None or decision.schema_version != _OWED_DECISION_SCHEMA_VERSION:
        return None
    return decision


def owed_decision_of_status(
    checkpoint: dict[str, Any] | None,
    status: SessionStatus,
) -> InvocationTerminalDecision | None:
    """Return the unsettled decision a session in ``status`` still owes, if any."""

    decision = unsettled_owed_decision(checkpoint)
    if decision is None or OUTCOME_SESSION_STATUS[decision.outcome] is not status:
        return None
    return decision


def checkpoint_settling_owed_decision(
    checkpoint: dict[str, Any] | None,
    *,
    terminal_event: Event | None,
) -> dict[str, Any] | None:
    """Settle an owed outcome in the fence release, refusing until it is published.

    ``terminal_event`` is the durable event the store found under the decision's
    terminal event ID, if any. A release with no owed decision is unchanged.
    """

    decision = unsettled_owed_decision(checkpoint)
    if decision is None:
        return checkpoint
    if (
        terminal_event is None
        or terminal_event.type is not OUTCOME_SESSION_EVENT[decision.outcome]
        or terminal_event.id != decision.terminal_event_id
        or terminal_event.session_id != decision.session_id
    ):
        raise SessionRunFenced("The run fence is held until its terminal outcome is published.")
    # The release receipt is the settled outcome's acknowledgement-loss evidence,
    # so nothing of the decision is retained for the next admission.
    settled = copy_durable_json_object(checkpoint, "checkpoint")
    settled.pop(INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY)
    return settled


def require_owed_outcome_kept(
    status: SessionStatus,
    to_status: SessionStatus,
    checkpoint: dict[str, Any] | None,
) -> None:
    """Refuse moving a session off its decided status while that outcome is owed.

    Only the outcome's own publication, or recovery, may finish it; a later
    transition must not leave a decision that no release or sweep can settle.
    """

    if to_status is not status and owed_decision_of_status(checkpoint, status) is not None:
        raise SessionRunFenced(
            "A session keeps its terminal status until its terminal outcome is published."
        )


def require_owed_decision_authority(
    *,
    to_status: SessionStatus,
    checkpoint_mutation: dict[str, Any] | None,
    event: Event,
    session: Session,
    checkpoint: dict[str, Any] | None,
    active_profile: ActiveInvocationExecutionProfile | None,
) -> None:
    """Install only a decision that names the exact invocation the transition ends."""

    if (
        to_status not in OWED_OUTCOME_STATUSES
        or checkpoint_mutation is None
        or not is_owed_decision_install_mutation(checkpoint_mutation)
    ):
        return
    decision = owed_decision_from_install_mutation(
        checkpoint_mutation, event=event, to_status=to_status
    )
    profile = (
        active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_profile is None
        else active_profile
    )
    if profile is None or not invocation_terminal_decision_matches_active_profile(
        decision,
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        interaction_id=profile.interaction_id,
        execution_profile_fingerprint=profile.profile.fingerprint,
    ):
        raise SessionRunFenced("A terminal decision must belong to the invocation it ends.")

"""Writes that must happen together, carried by one journaled intent.

When a crash between two durable writes would leave a state nothing repairs,
the intent of the whole pair is installed in the same store transaction as the
first write and settled in the same transaction as the last one. Every write
owed in between is replay-safe and keyed by the intent, so its owner, or
recovery once the owner is gone, finishes the pair from durable state alone.
The intent stays in the session checkpoint until it settles, which is what
recovery lists.

A completed interaction is such a pair. Its invocation terminal decision is
installed by the COMPLETED transition, as a set-if-absent checkpoint mutation
that every store applies only when the session actually completes. The fence
release settles it, and only once its SESSION_COMPLETED event is durable.
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
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
from cayu.sessions.records import Session, SessionStatus


def completed_decision_install_mutation(decision: InvocationTerminalDecision) -> dict[str, Any]:
    """Return the checkpoint mutation that installs a completed decision with its transition."""

    if decision.outcome is not InvocationTerminalOutcome.COMPLETED:
        raise ValueError("Only a completed decision rides on the completion transition.")
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


def completed_decision_from_install_mutation(
    mutation: dict[str, Any],
    *,
    event: Event,
) -> InvocationTerminalDecision:
    """Authenticate a completion transition's mutation as the decision for its own event."""

    operations = RuntimePublicationMutation.model_validate(mutation).operations
    if (
        len(operations) != 1
        or operations[0].key != INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
        or operations[0].action != "set"
        or operations[0].expected_value_digest is not None
    ):
        raise ValueError("A completion transition may only install its completed decision.")
    decision = InvocationTerminalDecision.model_validate(operations[0].value)
    if (
        decision.outcome is not InvocationTerminalOutcome.COMPLETED
        or decision.interaction_event_id != event.id
        or decision.interaction_id != event.interaction_id
        or decision.session_id != event.session_id
        or decision.observed_at != event.timestamp
    ):
        raise ValueError("A completed decision belongs to another interaction completion.")
    return decision


def unsettled_completed_decision(
    checkpoint: dict[str, Any] | None,
) -> InvocationTerminalDecision | None:
    """Return the completed decision whose owed writes are not all durable yet."""

    decision = invocation_terminal_decision_from_checkpoint(checkpoint)
    if decision is None or decision.outcome is not InvocationTerminalOutcome.COMPLETED:
        return None
    return decision


def checkpoint_settling_completed_decision(
    checkpoint: dict[str, Any] | None,
    *,
    terminal_event: Event | None,
) -> dict[str, Any] | None:
    """Settle an owed completion in the fence release, refusing until it is published.

    ``terminal_event`` is the durable event the store found under the decision's
    terminal event ID, if any. A release with no completed decision is unchanged.
    """

    decision = unsettled_completed_decision(checkpoint)
    if decision is None:
        return checkpoint
    if (
        terminal_event is None
        or terminal_event.type is not EventType.SESSION_COMPLETED
        or terminal_event.id != decision.terminal_event_id
        or terminal_event.session_id != decision.session_id
    ):
        raise SessionRunFenced("The run fence is held until its completed outcome is published.")
    # The release receipt is the settled completion's acknowledgement-loss evidence,
    # so nothing of the decision is retained for the next admission.
    settled = copy_durable_json_object(checkpoint, "checkpoint")
    settled.pop(INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY)
    return settled


def require_owed_completion_kept(
    status: SessionStatus,
    to_status: SessionStatus,
    checkpoint: dict[str, Any] | None,
) -> None:
    """Refuse moving a completed session off COMPLETED while its completion is owed.

    Only the completion's own publication, or recovery, may finish it; a later
    failure must not leave a decision that no release or sweep can settle.
    """

    if (
        status is SessionStatus.COMPLETED
        and to_status is not SessionStatus.COMPLETED
        and unsettled_completed_decision(checkpoint) is not None
    ):
        raise SessionRunFenced(
            "A completed session keeps its status until its completion is published."
        )


def require_completed_decision_authority(
    *,
    to_status: SessionStatus,
    checkpoint_mutation: dict[str, Any] | None,
    event: Event,
    session: Session,
    checkpoint: dict[str, Any] | None,
    active_profile: ActiveInvocationExecutionProfile | None,
) -> None:
    """Install only a decision that names the exact invocation the transition completes."""

    if to_status is not SessionStatus.COMPLETED or checkpoint_mutation is None:
        return
    decision = completed_decision_from_install_mutation(checkpoint_mutation, event=event)
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
        raise SessionRunFenced("A completed decision must belong to the invocation it completes.")

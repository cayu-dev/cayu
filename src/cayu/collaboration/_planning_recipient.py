"""Prepared local admission composition; never a recipient launcher or writer."""

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._planning_stages import (
    _PlannedStage,
    admission_stage_intent,
    retain_preparation_stage,
)
from cayu.collaboration._planning_store import (
    _event,
    _write_transition,
    admission_command,
    read_plan_in_transaction,
)
from cayu.collaboration._preparation import require_exact_contract
from cayu.collaboration._request_arbitration import admit_in_transaction
from cayu.collaboration.planning import RequestPlanningContinue


async def retain_prepared_stage(store, tx, initialized, prior, *, redactor):
    """Retain the exact receiving command before calling its registered guard."""
    if prior.state != "decided" or not isinstance(prior.decision, RequestPlanningContinue):
        raise CollaborationConflict("Planning has no eligible prepared selection.")
    command = admission_command(prior, redactor)
    return await retain_preparation_stage(
        store,
        tx,
        initialized,
        prior,
        admission_stage_intent(prior, redactor),
        content=command,
        redactor=redactor,
    )


async def admit_prepared_plan(requests, record, *, context, intent=None):
    """Acquire the existing receiver outside the planning mandate's guard."""
    planned = _PlannedStage(
        record.receipt.command,
        admission_stage_intent(record, requests._redactor) if intent is None else intent,
    )

    async def commit(store, tx, initialized, command, *, settlement=None, redactor):
        require_exact_contract(planned.intent.command, command, redactor=redactor)
        # Both receiving admission and its stage settlement belong to this same
        # transaction. The registered receiver has already authenticated the
        # SessionStore selection, which is evidence, not a cross-store lease.
        receipt = await admit_in_transaction(
            store,
            tx,
            initialized,
            command,
            settlement=settlement,
            redactor=redactor,
            _planned_stage=planned,
        )
        found = await read_plan_in_transaction(
            store, tx, initialized, planned.plan, redactor=redactor
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationConflict("Prepared admission lost its exact planning owner.")
        prior = found.receipt
        if prior.state == "admitted":
            return prior
        if prior.state != "preparing" or prior.pending_stages:
            raise CollaborationConflict("Prepared planning frontier conflicts with settlement.")
        anchor = await store._anchor(tx, initialized, redactor)
        event = _event(
            planned.plan,
            anchor.event_sequence + 1,
            "plan_admitted",
            redactor,
            content=receipt.command,
        )
        updated = prior.model_copy(
            update={
                "state": "admitted",
                "revision": prior.revision + 1,
                "event_sequences": (*prior.event_sequences, event.sequence),
                "reserved_bytes": 0,
                "reserved_events": 0,
            }
        )
        return await _write_transition(store, tx, initialized, prior, updated, event, redactor)

    return await requests._trusted_mutation(
        planned.intent.command,
        context=context,
        operation=commit,
        # The outer planner already bounds caller observation. Keep its owned
        # continuation attached to the real receiving settlement, not a second
        # foreground timeout masquerading as an execution failure.
        wait_for_settlement=True,
    )

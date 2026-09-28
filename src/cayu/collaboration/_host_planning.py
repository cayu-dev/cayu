"""Retained host turns through the registered request-planning owner."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._host_ownership import (
    HostOperationIdentity,
    HostOwnership,
    HostReconciledResult,
)
from cayu.collaboration._planning_coordinator import plan_request, service_application_plan
from cayu.collaboration._planning_records import RequestPlanningRecord
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._preparation_progress import PreparationReadFailure
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningRequest


@dataclass(frozen=True, slots=True)
class HostPlanningResult:
    dispatched: bool
    record: RequestPlanningRecord | None = None


async def lookup_host_plan(app, expected, *, context):
    """Join the native read under the host's retained task, not a second observer."""
    return await plan_request(
        app._request_coordinator,
        expected,
        context=context,
        read_only=True,
        wait_for_settlement=True,
    )


def planning_needs_service(record: RequestPlanningRecord) -> bool:
    # Pending stages remain obligations even when a control decision won. An
    # admitted plan with no pending stages is historical evidence, not new work.
    return bool(record.pending_stages) or record.state in (
        "evaluating",
        "decided",
        "deferred",
        "preparing",
    )


def start_planning(
    app,
    ownership: HostOwnership,
    expected: RequestPlanningRequest,
    *,
    context: MandateAccessContext,
    observation_deadline: float,
) -> HostOperationIdentity:
    """Keep native preparation ownership beyond the host observer's lifetime.

    First retention and subsequent reconciliation use the same complete request.
    No host-generated generation, mandate or replacement decision is introduced.
    The planning owner authenticates semantic deadlines and every foreign stage.
    """
    redactor = app._secret_redactor
    expected = prepare_contract(RequestPlanningRequest, expected, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host planning requires a finite observation deadline.")
    encoded = contract_bytes(expected, redactor=redactor)
    material = encoded + b"\0" + contract_bytes(context, redactor=redactor)
    identity = HostOperationIdentity(
        key="planning:" + sha256(encoded).hexdigest(),
        commitment=sha256(material).hexdigest(),
    )

    async def action(stop):
        def stopped():
            return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

        if stopped():
            return HostPlanningResult(False)
        try:
            found = await lookup_host_plan(app, expected, context=context)
            if isinstance(found, ExactMatch):
                if not planning_needs_service(found.receipt):
                    return HostPlanningResult(False)
                require_retained = True
            elif isinstance(found, ExactNotFound):
                require_retained = False
            else:
                # Unavailable/conflicting state cannot authorize a fresh operation.
                raise CollaborationUnavailable("Host planning lacks exact source evidence.")
        except Exception as error:
            # No receiving mutation was entered; this releases only the local turn.
            return HostReconciledResult(HostPlanningResult(False), error)
        if not await ownership.wait_for_dispatch_window(
            identity, initial_deadline=observation_deadline
        ):
            return HostPlanningResult(False)
        record = await service_application_plan(
            app,
            expected,
            context=context,
            require_retained=require_retained,
            wait_for_settlement=True,
        )
        if type(record) is PreparationReadFailure:
            return HostReconciledResult(HostPlanningResult(False), record.error)
        record = prepare_contract(RequestPlanningRecord, record, redactor=redactor)
        if record.receipt.command != expected:
            raise CollaborationUnavailable("Host planning returned another operation.")
        return HostPlanningResult(True, record)

    async def reconcile():
        found = await lookup_host_plan(app, expected, context=context)
        if not isinstance(found, ExactMatch):
            return None
        record = prepare_contract(RequestPlanningRecord, found.receipt, redactor=redactor)
        if record.receipt.command != expected:
            raise CollaborationUnavailable("Host planning readback changed its exact intent.")
        return None if planning_needs_service(record) else HostPlanningResult(False)

    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_planning(ownership: HostOwnership, outcome) -> bool:
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostPlanningResult or result.dispatched != (result.record is not None):
        raise RuntimeError("Planning owner returned invalid host handoff evidence.")
    # This releases a local observation turn, never the plan's retained stages,
    # resource permits, admission or continuation responsibility.
    ownership.release_settled(outcome.identity)
    return True

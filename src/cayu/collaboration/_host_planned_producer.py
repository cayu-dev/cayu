"""Compose an exact retained plan with native producer preparation/attachment."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from pydantic import Field

from cayu.collaboration._contracts import (
    ContractValue,
    ExactMatch,
    ExactNotFound,
    Identifier,
    OperationRef,
)
from cayu.collaboration._host_ownership import HostOperationIdentity, HostReconciledResult
from cayu.collaboration._host_planning import lookup_host_plan
from cayu.collaboration._host_producer_recovery import (
    read_admitted_producer_plan,
    recover_planned_execution,
)
from cayu.collaboration._host_producer_registration import (
    attach_host_producer,
    producer_registration_ready,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_bounds import MAX_OUTPUT_DESTINATIONS
from cayu.collaboration._producer_contracts import (
    ProducerDeliveryDestination,
    ProducerOutputLimits,
    ProducerOutputProposal,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_store import read_request_output
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningRequest
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


class HostPlannedProducer(ContractValue):
    """Finite application choices, not a caller-created admission or launch grant."""

    plan: RequestPlanningRequest
    operation: OperationRef
    binding_incarnation: Identifier
    execution_key: Identifier
    recovery_inactive_for_seconds: int | None = Field(default=None, strict=True, ge=1, le=86400)
    limits: ProducerOutputLimits
    destinations: tuple[ProducerDeliveryDestination, ...] = Field(
        min_length=1, max_length=MAX_OUTPUT_DESTINATIONS
    )


@dataclass(frozen=True, slots=True)
class _PlannedProducerRule:
    selected: HostPlannedProducer
    context: CollaborationAccessContext
    producer_context: MandateAccessContext


def snapshot_planned_producer_rule(app, rule):
    if type(rule) is not _PlannedProducerRule:
        raise TypeError("Host planned producer rule must be typed.")
    redactor = app._secret_redactor
    selected = prepare_contract(HostPlannedProducer, rule.selected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
    producer_context = prepare_contract(
        MandateAccessContext, rule.producer_context, redactor=redactor
    )
    size = sum(
        len(contract_bytes(value, redactor=redactor))
        for value in (selected, context, producer_context)
    )
    return _PlannedProducerRule(selected, context, producer_context), size


async def execution_for_planned_producer(app, rule, recovery):
    from cayu.collaboration._host_producer_execution import HostProducerExecution

    selected = rule.selected
    if recovery.registration != selected.operation:
        return None
    from cayu.collaboration._producer_readback import lookup_producer_registration

    found = await lookup_producer_registration(
        app, recovery, context=rule.context, wait_for_settlement=True
    )
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Planned producer registration is unavailable.")
    command = prepare_contract(
        ProducerOutputRegistration, found.receipt, redactor=app._secret_redactor
    )
    _require_selected_command(command, selected)
    return HostProducerExecution(
        recovery=recovery,
        expected_plan=selected.plan,
        recovery_inactive_for_seconds=selected.recovery_inactive_for_seconds,
    )


def _require_selected_command(command, selected):
    if (
        command.operation != selected.operation
        or command.admission.expected != selected.plan.expected
        or command.admission.operation != selected.plan.admission_operation
        or command.admission.generation != selected.plan.admission_generation
        or command.binding_incarnation != selected.binding_incarnation
        or command.execution_key != selected.execution_key
        or command.limits != selected.limits
        or command.destinations != selected.destinations
    ):
        raise CollaborationUnavailable("Planned producer differs from configured exact intent.")


@dataclass(frozen=True, slots=True)
class HostPlannedProducerResult:
    dispatched: bool
    command: ProducerOutputRegistration | None = None


def planned_producer_identity(selected, context, producer_context, *, redactor):
    material = b"\0".join(
        contract_bytes(value, redactor=redactor) for value in (selected, context, producer_context)
    )
    return (
        HostOperationIdentity(
            "planned-producer:"
            + sha256(contract_bytes(selected.operation, redactor=redactor)).hexdigest(),
            sha256(material).hexdigest(),
        ),
        len(material),
    )


def start_planned_producer(
    app, ownership, selected, *, context, producer_context, observation_deadline
):
    redactor = app._secret_redactor
    selected = prepare_contract(HostPlannedProducer, selected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    producer_context = prepare_contract(MandateAccessContext, producer_context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Planned producer requires a finite observation deadline.")
    identity, size = planned_producer_identity(
        selected, context, producer_context, redactor=redactor
    )

    async def authenticate_plan(command):
        _require_selected_command(command, selected)
        found = await app._request_coordinator.lookup_admission(
            command.admission, context=producer_context, wait_for_settlement=True
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Planned producer admission is unavailable.")
        await read_admitted_producer_plan(
            app, found.receipt, context=producer_context, expected_plan=selected.plan
        )

    async def action(stop):
        def stopped():
            return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

        if stopped():
            return HostPlannedProducerResult(False)
        try:
            plan = await lookup_host_plan(app, selected.plan, context=producer_context)
            if isinstance(plan, ExactNotFound):
                return HostPlannedProducerResult(False)
            if not isinstance(plan, ExactMatch):
                raise CollaborationUnavailable("Planned producer has no exact planning evidence.")
            if plan.receipt.state != "admitted" or plan.receipt.pending_stages:
                return HostPlannedProducerResult(False)
            snapshot = await app._request_coordinator.inspect(
                selected.plan.expected, context=producer_context, wait_for_settlement=True
            )
            if snapshot is None:
                raise CollaborationUnavailable("Planned producer request is unavailable.")
            command = None
            if snapshot.producer_operation is not None:
                if snapshot.producer_operation != selected.operation:
                    raise CollaborationUnavailable(
                        "Planned request has another producer attachment."
                    )
                store, initialized = app._participant_coordinator._ready()
                app._participant_coordinator._capability(
                    store, initialized, mutation=False, family=REQUEST_FAMILY
                )
                async with store._transaction(
                    initialized.owner.application_scope, write=False
                ) as tx:
                    record = await read_request_output(
                        tx, selected.plan.expected, redactor=redactor
                    )
                if record is None:
                    raise CollaborationUnavailable(
                        "Planned producer responsibility is unavailable."
                    )
                command = record.command
                await authenticate_plan(command)
                if await producer_registration_ready(app, command, context=context):
                    return HostPlannedProducerResult(False, command)
                # The source committed, but native attachment may have lost its
                # acknowledgement or never run. Repair that same registration; do
                # not re-prepare or mint a new operation from the old admission.
            from cayu.collaboration._admission_recovery import recover_admission

            found = await recover_admission(
                app._request_coordinator,
                snapshot,
                context=producer_context,
                wait_for_settlement=True,
            )
            if not isinstance(found, ExactMatch):
                raise CollaborationUnavailable("Planned producer admission is unavailable.")
            admission = found.receipt
            # This reader proves the admission's settled stage and complete retained
            # plan. Comparing its frozen input alone would not identify the plan.
            execution = await recover_planned_execution(
                app,
                admission,
                context=producer_context,
                expected_plan=selected.plan,
            )
            execution = ParticipantSessionExecutionRequest(
                request=execution.request,
                session_instance_id=execution.session_instance_id,
                execution_key=selected.execution_key,
            )
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostPlannedProducerResult(False)
            if command is None:
                from cayu.collaboration._producer_preparation import prepare_producer_output

                command = await prepare_producer_output(
                    app,
                    ProducerOutputProposal(
                        operation=selected.operation,
                        admission=admission.command,
                        binding_incarnation=selected.binding_incarnation,
                        limits=selected.limits,
                        destinations=selected.destinations,
                    ),
                    execution,
                    context=producer_context,
                    wait_for_settlement=True,
                )
            command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
            if await producer_registration_ready(app, command, context=context):
                return HostPlannedProducerResult(False, command)
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostPlannedProducerResult(False)
        except Exception as error:
            # No receiving mutation was entered; this releases only the local turn.
            return HostReconciledResult(HostPlannedProducerResult(False), error)
        failure = await attach_host_producer(
            app, command, execution, context=context, producer_context=producer_context
        )
        result = HostPlannedProducerResult(True, command)
        return result if failure is None else HostReconciledResult(result, failure)

    async def reconcile():
        from cayu.collaboration._producer_readback import lookup_producer_registration

        snapshot = await app._request_coordinator.inspect(
            selected.plan.expected, context=producer_context, wait_for_settlement=True
        )
        if snapshot is None or snapshot.producer_operation is None:
            return None
        if snapshot.producer_operation != selected.operation:
            raise CollaborationUnavailable("Planned request has another producer attachment.")
        store, initialized = app._participant_coordinator._ready()
        app._participant_coordinator._capability(
            store, initialized, mutation=False, family=REQUEST_FAMILY
        )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_request_output(tx, selected.plan.expected, redactor=redactor)
        if record is None:
            return None
        found = await lookup_producer_registration(
            app, record.command, context=context, wait_for_settlement=True
        )
        if not isinstance(found, ExactMatch):
            return None
        command = prepare_contract(ProducerOutputRegistration, found.receipt, redactor=redactor)
        await authenticate_plan(command)
        if not await producer_registration_ready(app, command, context=context):
            return None
        return HostPlannedProducerResult(False, command)

    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=size + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_planned_producer(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostPlannedProducerResult or (
        result.dispatched and result.command is None
    ):
        raise RuntimeError("Planned producer lacks native registration evidence.")
    ownership.release_settled(outcome.identity)
    return True

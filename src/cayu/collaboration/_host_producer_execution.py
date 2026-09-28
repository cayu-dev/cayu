"""Retained host supervision over the existing native producer invocation.

No launch generation or request body is supplied by this adapter. The frozen
registered intent is reconstructed, then the ordinary native handoff authenticates
current execution. Local shutdown never means business-request cancellation.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from pydantic import Field

from cayu.collaboration._contracts import ContractValue, ExactMatch, OperationRef
from cayu.collaboration._host_ownership import (
    HostOperationIdentity,
    HostOwnership,
    HostReconciledResult,
)
from cayu.collaboration._host_producer_recovery import (
    recover_interrupted_producer,
    recover_registered_execution,
)
from cayu.collaboration._host_producer_registration import (
    attach_host_producer,
    producer_registration_ready,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_cleanup_finalization import read_finalization
from cayu.collaboration._producer_contracts import ProducerCleanupRecord
from cayu.collaboration._producer_inspection import inspect_producer_output
from cayu.collaboration._producer_public_control import observe_public_producer_completion
from cayu.collaboration._producer_readback import lookup_producer_registration
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningRequest
from cayu.events import EventType


class HostProducerExecution(ContractValue):
    recovery: ProducerOutputRecovery
    expected_plan: RequestPlanningRequest | None = None
    recovery_inactive_for_seconds: int | None = Field(default=None, strict=True, ge=1, le=86400)


@dataclass(frozen=True, slots=True)
class HostExecutionResult:
    # A deferred turn did not invoke execution. Release, no-start exclusion and
    # finalized cleanup remain distinct evidence for a host-local handoff. This
    # result itself never settles output, budgets or effects.
    dispatched: bool
    completion: OperationRef | None = None
    release_commitment: str | None = None
    exclusion_commitment: str | None = None
    cleanup_commitment: str | None = None


async def _terminal_handoff(app, command):
    """Read exact exclusion/finalization, without addressing a successor invocation."""
    redactor = app._secret_redactor
    registered = app._request_coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Host producer requires its native receiver.")
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, command, redactor=redactor)
        if record is None:
            raise CollaborationUnavailable("Host producer responsibility is unavailable.")
        final = await read_finalization(tx, record, redactor=redactor)
        if final is not None:
            return HostExecutionResult(
                False,
                cleanup_commitment=sha256(contract_bytes(final, redactor=redactor)).hexdigest(),
            )
        if record.state != "excluded" or not isinstance(record.cleanup, ProducerCleanupRecord):
            raise CollaborationUnavailable("Host producer exclusion is unavailable.")
        expected = command.admission.expected
        request = await retained_request(
            store, tx, initialized, expected.intent.request, expected.initiator, redactor
        )
        if (
            request is None
            or request.terminal is None
            or request.producer_operation != command.operation
        ):
            raise CollaborationUnavailable("Host producer lacks exact request closure.")
        control = request.terminal
        require_exact_contract(expected, control.expected.intent.expected, redactor=redactor)
    # Native readback is outside the source transaction. Finalized cleanup above
    # remains authoritative even after deletion or reuse of the public ID.
    exclusion = await receiver._read_producer_exclusion(record, control)
    require_exact_contract(record.cleanup.exclusion, exclusion, redactor=redactor)
    return HostExecutionResult(
        False, exclusion_commitment=sha256(contract_bytes(exclusion, redactor=redactor)).hexdigest()
    )


async def _release_proof(app, command):
    registered = app._request_coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Host producer release requires its native receiver.")
    require_exact_contract(command.receiver, receiver.ref, redactor=app._secret_redactor)
    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, command, redactor=app._secret_redactor)
    if record is None:
        raise CollaborationUnavailable("Host producer responsibility is unavailable.")
    release = await receiver._read_producer_release(record)
    require_exact_contract(command, release.registration, redactor=app._secret_redactor)
    return release


def start_producer_execution(
    app,
    ownership: HostOwnership,
    intent: HostProducerExecution,
    *,
    context: CollaborationAccessContext,
    producer_context: MandateAccessContext,
    observation_deadline: float,
) -> HostOperationIdentity:
    redactor = app._secret_redactor
    intent = prepare_contract(HostProducerExecution, intent, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    producer_context = prepare_contract(MandateAccessContext, producer_context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host execution requires a finite observation deadline.")
    if app.session_store._supports_producer_attachment_protocol() is not True:
        raise CollaborationUnavailable(
            "Host execution requires qualified native producer ownership."
        )
    encoded = contract_bytes(intent, redactor=redactor)
    material = b"\0".join(
        (
            encoded,
            contract_bytes(context, redactor=redactor),
            contract_bytes(producer_context, redactor=redactor),
        )
    )
    identity = HostOperationIdentity(
        "producer-execution:" + sha256(encoded).hexdigest(), sha256(material).hexdigest()
    )
    phase = "observation"

    async def action(stop):
        nonlocal phase

        def stopped():
            return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

        if stopped():
            return HostExecutionResult(False)
        try:
            inspection = await inspect_producer_output(
                app, intent.recovery, context=context, wait_for_settlement=True
            )
            if not isinstance(inspection, ExactMatch):
                raise CollaborationUnavailable("Host producer lacks exact source evidence.")
            command_lookup = await lookup_producer_registration(
                app, intent.recovery, context=context, wait_for_settlement=True
            )
            if not isinstance(command_lookup, ExactMatch):
                raise CollaborationUnavailable("Host producer registration is unavailable.")
            command = command_lookup.receipt
        except Exception as error:
            # No receiving mutation was entered; this releases only the local turn.
            return HostReconciledResult(HostExecutionResult(False), error)
        if inspection.receipt.state == "excluded" or inspection.receipt.cleanup_ack is not None:
            return await _terminal_handoff(app, command)
        dispatched = False
        if inspection.receipt.completion is None and inspection.receipt.state == "registered":
            try:
                command, execution = await recover_registered_execution(
                    app,
                    intent.recovery,
                    context=context,
                    producer_context=producer_context,
                    expected_plan=intent.expected_plan,
                )
                attached = await producer_registration_ready(app, command, context=context)
            except Exception as error:
                return HostReconciledResult(HostExecutionResult(False), error)
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostExecutionResult(False)
            if not attached:
                if not await ownership.wait_for_dispatch_window(
                    identity, initial_deadline=observation_deadline
                ):
                    return HostExecutionResult(False)
                phase = "attachment"
                failure = await attach_host_producer(
                    app, command, execution, context=context, producer_context=producer_context
                )
                if failure is not None:
                    # Attachment recovery is not a successful execution turn.
                    # Report the original failure before any later dispatch.
                    return HostReconciledResult(HostExecutionResult(False), failure)
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostExecutionResult(False)
            # Slow reconstruction retains its exact result while waiting for a
            # later service window. The native entrance still repeats current
            # mandate, semantic deadline, profile and budget gates.
            # The retained task drains the stream even after its observer leaves.
            phase = "execution"
            async with aclosing(
                app.execute_producer_output(
                    command, execution, context=context, producer_context=producer_context
                )
            ) as stream:
                async for event in stream:
                    if event.type == EventType.MODEL_STARTED:
                        ownership.mark_active(identity)
            dispatched = True
        elif (
            inspection.receipt.completion is None
            and inspection.receipt.state == "launch_claimed"
            and intent.recovery_inactive_for_seconds is not None
        ):
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostExecutionResult(False)
            phase = "execution"
            if not await recover_interrupted_producer(
                app,
                command,
                context=context,
                inactive_for_seconds=intent.recovery_inactive_for_seconds,
            ):
                return HostExecutionResult(False)
            dispatched = True
        # Never rerun a claimed/completed producer to repair retention. Missing
        # invocation-release proof leaves this exact local task capacity-counted
        # for native reconciliation, rather than authorizing another invocation.
        completion = await observe_public_producer_completion(app, command, context=context)
        release = await _release_proof(app, command)
        # A human pause releases the invocation but does not complete the
        # producer. Only the exact native release frees this host-local slot;
        # the original durable attachment and human gate still forbid relaunch.
        return HostExecutionResult(
            dispatched,
            None if completion is None else completion.operation,
            release.release_commitment,
        )

    async def reconcile():
        current = await inspect_producer_output(
            app, intent.recovery, context=context, wait_for_settlement=True
        )
        if not isinstance(current, ExactMatch):
            return None
        found = await lookup_producer_registration(
            app, intent.recovery, context=context, wait_for_settlement=True
        )
        if not isinstance(found, ExactMatch):
            return None
        if current.receipt.state == "excluded" or current.receipt.cleanup_ack is not None:
            return await _terminal_handoff(app, found.receipt)
        if phase == "attachment":
            # This turn never entered native execution. Exact attachment ends
            # only its local preparation ownership; a later turn must repeat
            # launch authorization. Another worker's invocation is not ours.
            if await producer_registration_ready(app, found.receipt, context=context):
                return HostExecutionResult(False)
            return None
        release = await _release_proof(app, found.receipt)
        return HostExecutionResult(False, current.receipt.completion, release.release_commitment)

    ownership.start(
        identity,
        role="execution",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_producer_execution(ownership: HostOwnership, outcome) -> bool:
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostExecutionResult or (
        (result.completion is not None and result.release_commitment is None)
        or (result.dispatched and result.release_commitment is None)
        or (
            (result.exclusion_commitment is not None or result.cleanup_commitment is not None)
            and (
                result.dispatched
                or result.completion is not None
                or result.release_commitment is not None
                or (
                    result.exclusion_commitment is not None
                    and result.cleanup_commitment is not None
                )
            )
        )
    ):
        raise RuntimeError("Native producer execution lacks exact handoff evidence.")
    ownership.release_settled(outcome.identity)
    return True

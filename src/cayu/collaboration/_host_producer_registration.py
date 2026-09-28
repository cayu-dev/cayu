"""Attach admitted work through the existing native producer-registration owner."""

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
from cayu.collaboration._host_producer_recovery import recover_planned_execution
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRecord, ProducerOutputRegistration
from cayu.collaboration._producer_readback import lookup_producer_registration
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._producer_registration import (
    _native_execution_commitment,
    register_producer_output,
)
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._producer_output_store import NativeProducerAttachment
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


@dataclass(frozen=True, slots=True)
class HostRegistrationResult:
    dispatched: bool
    recovery: ProducerOutputRecovery | None = None


async def producer_registration_ready(app, expected, *, context):
    """Exact native attachment or authenticated final cleanup, never source-only proof."""
    redactor = app._secret_redactor
    expected = prepare_contract(ProducerOutputRegistration, expected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    found = await lookup_producer_registration(
        app, expected, context=context, wait_for_settlement=True
    )
    if isinstance(found, ExactNotFound):
        return False
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Host producer registration is unavailable.")
    require_exact_contract(expected, found.receipt, redactor=redactor)
    configured = app._request_coordinator._registration
    receiver = None if configured is None else configured.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Host attachment requires its registered native receiver.")
    require_exact_contract(expected.receiver, receiver.ref, redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    app._participant_coordinator._capability(
        store, initialized, mutation=False, family=REQUEST_FAMILY
    )
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, expected, redactor=redactor)
    if record is None:
        raise CollaborationUnavailable("Host attachment source responsibility disappeared.")
    if record.cleanup_ack is not None:
        # read_output_registration authenticates the full native cleanup receipt
        # and every destination settlement. A completed obligation is not fresh
        # attachment work, even if its session was subsequently retired.
        return True
    attachment = await receiver._read_producer_attachment(record)
    if attachment is None:
        return False
    require_exact_contract(
        NativeProducerAttachment.from_registration(record), attachment, redactor=redactor
    )
    return True


async def attach_host_producer(app, expected, execution, *, context, producer_context):
    """Retain native attachment ownership, including its post-commit ACK window."""
    failure = None
    try:
        registered = await register_producer_output(
            app, expected, execution, context=producer_context, wait_for_settlement=True
        )
    except Exception as error:
        try:
            ready = await producer_registration_ready(app, expected, context=context)
            if not ready:
                # Registration and native attachment are separate transactions.
                # A retained exact registration owns the remaining attachment;
                # repair it through the same receiver, never create a fresh
                # registration merely because the failed attempt returned.
                source = await lookup_producer_registration(
                    app, expected, context=context, wait_for_settlement=True
                )
                if isinstance(source, ExactMatch):
                    require_exact_contract(expected, source.receipt, redactor=app._secret_redactor)
                    repaired = await register_producer_output(
                        app, expected, execution, context=producer_context, wait_for_settlement=True
                    )
                    repaired = prepare_contract(
                        ProducerOutputRecord, repaired, redactor=app._secret_redactor
                    )
                    require_exact_contract(
                        expected, repaired.command, redactor=app._secret_redactor
                    )
                    ready = await producer_registration_ready(app, expected, context=context)
        except Exception as recovery_error:
            raise ExceptionGroup(
                "Producer attachment and exact readback failed", [error, recovery_error]
            ) from None
        if not ready:
            raise
        failure = error
    else:
        registered = prepare_contract(
            ProducerOutputRecord, registered, redactor=app._secret_redactor
        )
        require_exact_contract(expected, registered.command, redactor=app._secret_redactor)
    return failure


def start_producer_registration(
    app,
    ownership: HostOwnership,
    expected: ProducerOutputRegistration,
    *,
    context: CollaborationAccessContext,
    producer_context: MandateAccessContext,
    observation_deadline: float,
) -> HostOperationIdentity:
    """Retain output responsibility before permitting any model dispatch.

    The application supplies an exact native-prepared registration, not a host
    authority grant. The existing receiving owner authenticates registration;
    execution remains a later, independently authorized native handoff.
    """
    redactor = app._secret_redactor
    expected = prepare_contract(ProducerOutputRegistration, expected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    producer_context = prepare_contract(MandateAccessContext, producer_context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host registration requires a finite observation deadline.")
    encoded = contract_bytes(expected, redactor=redactor)
    material = b"\0".join(
        (
            encoded,
            contract_bytes(context, redactor=redactor),
            contract_bytes(producer_context, redactor=redactor),
        )
    )
    commitment = sha256(encoded).hexdigest()
    identity = HostOperationIdentity(
        key="producer-registration:" + commitment,
        commitment=sha256(material).hexdigest(),
    )

    async def action(stop):
        def stopped():
            return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

        if stopped():
            return HostRegistrationResult(False)
        try:
            if await producer_registration_ready(app, expected, context=context):
                return HostRegistrationResult(False)
            admission = await app._request_coordinator.lookup_admission(
                expected.admission, context=producer_context, wait_for_settlement=True
            )
            if not isinstance(admission, ExactMatch):
                raise CollaborationUnavailable("Host registration admission is unavailable.")
            recovered = await recover_planned_execution(
                app, admission.receipt, context=producer_context
            )
            execution = ParticipantSessionExecutionRequest(
                request=recovered.request,
                session_instance_id=recovered.session_instance_id,
                execution_key=expected.execution_key,
            )
            if (
                await _native_execution_commitment(app, expected.admission.prepared, execution)
                != expected.execution_commitment
            ):
                raise CollaborationUnavailable(
                    "Host registration input conflicts with native preparation."
                )
        except Exception as error:
            # No receiving mutation was entered; this releases only the local turn.
            return HostReconciledResult(HostRegistrationResult(False), error)
        if not await ownership.wait_for_dispatch_window(
            identity, initial_deadline=observation_deadline
        ):
            return HostRegistrationResult(False)
        failure = await attach_host_producer(
            app, expected, execution, context=context, producer_context=producer_context
        )
        result = HostRegistrationResult(
            True,
            ProducerOutputRecovery(
                registration=expected.operation,
                registration_commitment="sha256:" + commitment,
            ),
        )
        return result if failure is None else HostReconciledResult(result, failure)

    async def reconcile():
        if await producer_registration_ready(app, expected, context=context):
            return HostRegistrationResult(False)
        return None

    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_producer_registration(ownership: HostOwnership, outcome) -> bool:
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostRegistrationResult or result.dispatched != (
        result.recovery is not None
    ):
        raise RuntimeError("Producer registration returned invalid host handoff evidence.")
    ownership.release_settled(outcome.identity)
    return True

"""Application-owned producer registration; deliberately not a dispatch entrance.

The proposal is authenticated against real admission and native preparation.
Native launch remains a separate runtime-owned, current-authority boundary.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from cayu.collaboration._contracts import ExactMatch, OwnerRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_bounds import supports_peer_delivery
from cayu.collaboration._producer_contracts import ProducerOutputRecord, ProducerOutputRegistration
from cayu.collaboration._producer_store import (
    claim_output_launch_in_transaction,
    read_output_registration,
    register_output_in_transaction,
)
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.collaboration.request_access import RequestReceivingAuthorization
from cayu.collaboration.requests import RequestControlCommand, RequestControlReceipt
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.sessions._participant_execution_identity import (
    participant_execution_identity,
    require_execution_creation,
)
from cayu.sessions.context_views import ParticipantSessionExecutionRequest

if TYPE_CHECKING:
    from cayu.applications import CayuApp


def _snapshot_execution(execution):
    if type(execution) is not ParticipantSessionExecutionRequest:
        raise TypeError("Producer registration requires a typed native execution request.")
    return ParticipantSessionExecutionRequest(
        request=execution.request,
        session_instance_id=execution.session_instance_id,
        execution_key=execution.execution_key,
    )


async def _native_execution_commitment(app, prepared, execution) -> str:
    """One native identity algorithm for preparation and registration revalidation."""
    binding = await app.session_store.load_participant_session_binding(prepared.target.session_id)
    creation = await app.session_store.load_participant_session_creation_receipt(
        prepared.target.session_id
    )
    if (
        binding is None
        or creation is None
        or creation.binding != binding
        or binding.participant != prepared.recipient
        or creation.receipt_commitment != prepared.target.creation_receipt_commitment
        or creation.execution_profile_json != prepared.execution_profile_json
    ):
        raise CollaborationAccessDenied("Producer execution conflicts with native creation.")
    require_execution_creation(execution, creation)
    profile = ExecutionProfileIdentity.model_validate_json(creation.execution_profile_json)
    identity = participant_execution_identity(
        execution, binding, execution_profile_fingerprint=profile.fingerprint
    )
    return "sha256:" + identity.admission_commitment


def _require_delivery_configuration(app: CayuApp, command: ProducerOutputRegistration) -> None:
    """Reject known unsupported delivery configuration before claiming launch.

    Registered adapters remain trusted implementations of their public contract.
    This checks their declared coverage, not future receiving acceptance or a
    disclosure grant; those must still be resolved at each actual dispatch.
    """
    from cayu.collaboration._producer_acceptance import ProducerOutputAcceptanceReader
    from cayu.collaboration._session_export_coordinator import _RESERVATION_BYTES

    exports = app._session_export_coordinator
    registration = exports.ready()
    # Export history and byte reservations remain retained after settlement.
    # Serial delivery can reuse pending capacity, but cannot reuse these slots.
    if (
        registration.limits.max_exports < len(command.destinations)
        or registration.limits.max_retained_bytes < len(command.destinations) * _RESERVATION_BYTES
    ):
        raise CollaborationUnavailable("Producer delivery capacity is not qualified.")
    from cayu.collaboration._producer_export_absence import preflight_exclusion_capacity

    preflight_exclusion_capacity(exports, command)
    if (
        exports.mandate_ref != app._request_coordinator._resolver_ref
        or registration.policy.ref != exports.policy_ref
        or not supports_peer_delivery(app.session_store)
    ):
        raise CollaborationUnavailable("Producer delivery configuration is not qualified.")
    for destination in command.destinations:
        projector = exports.projectors.get(destination.projector)
        audience = OwnerRef(
            application_scope=destination.recipient.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
        )
        reader = exports.readers.get(audience)
        if (
            destination.disclosure_policy != exports.policy_ref
            or projector is None
            or projector.ref != destination.projector
            or reader is None
            or reader.owner != audience
        ):
            raise CollaborationUnavailable("Producer delivery configuration is not qualified.")
        if isinstance(reader, ProducerOutputAcceptanceReader):
            reader._require_producer_namespace(command.operation)


async def register_producer_output(
    app: CayuApp,
    command: ProducerOutputRegistration,
    execution: ParticipantSessionExecutionRequest,
    *,
    context: MandateAccessContext,
) -> ProducerOutputRecord:
    """Retain authenticated responsibility without starting the prepared session."""
    coordinator = app._request_coordinator
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    from cayu.runtime._producer_output_store import preflight_native_output

    preflight_native_output(command)
    # One private snapshot before any await, including nested mutable input.
    execution = _snapshot_execution(execution)

    require_exact_contract(command.initiator, _initiator(context), redactor=redactor)
    if not app.session_store._supports_producer_attachment_protocol():
        raise CollaborationUnavailable("Native producer attachment is not qualified.")
    if not app._run_limit_controller._supports_producer_budget_readback():
        raise CollaborationUnavailable("Producer accounting readback is not qualified.")
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    if (
        context.participant != prepared.recipient
        or execution.execution_key != command.execution_key
        or execution.request.session_id != prepared.target.session_id
        or execution.session_instance_id != prepared.target.session_instance_id
    ):
        raise CollaborationAccessDenied("Producer execution does not match its admitted target.")

    async def register():
        # This is the production historical reader, not a supplied receipt. It
        # also authenticates current read access before exact operation replay.
        admitted = await app.collaboration_admission_reader().lookup(
            command.admission, context=context
        )
        if not isinstance(admitted, ExactMatch) or admitted.receipt.state != "admitted":
            raise CollaborationUnavailable("Producer admission evidence is unavailable.")
        require_exact_contract(command.admission, admitted.receipt.command, redactor=redactor)
        if command.execution_commitment != await _native_execution_commitment(
            app, prepared, execution
        ):
            raise CollaborationAccessDenied("Producer execution commitment conflicts.")
        store, initialized = app._participant_coordinator._ready()
        app._participant_coordinator._capability(
            store, initialized, mutation=False, family=REQUEST_FAMILY
        )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            retained = await read_output_registration(tx, command, redactor=redactor)
            if retained is not None:
                return retained
        registration = coordinator._registration
        receiver = None if registration is None else registration.receiving_owner
        if type(receiver) is not RecipientAdmissionReceivingOwner:
            raise CollaborationUnavailable("Producer registration requires the native receiver.")
        require_exact_contract(
            command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor
        )
        require_exact_contract(prepared.receiver, receiver.ref, redactor=redactor)
        # Revalidate current mandate and inert native preparation. Historical
        # admission alone must never create a new responsibility or launch grant.
        async with receiver.acquire(command.admission, context=context) as raw:
            authority = prepare_contract(RequestReceivingAuthorization, raw, redactor=redactor)
            require_exact_contract(authority.command, command.admission, redactor=redactor)
            require_exact_contract(authority.receiver, command.receiver, redactor=redactor)
            app._participant_coordinator._capability(
                store, initialized, mutation=True, family=REQUEST_FAMILY
            )
            async with store._transaction(initialized.owner.application_scope, write=True) as tx:
                return await register_output_in_transaction(
                    store,
                    tx,
                    initialized,
                    command,
                    authority_expires_at_ms=authority.expires_at_ms,
                    redactor=redactor,
                )

    async def owned():
        async def attach():
            record = await register()
            await app.session_store._attach_native_producer(record)
            return record

        return await coordinator._dependency(attach)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_registration", object()),
            expectation=contract_bytes(command, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )


@asynccontextmanager
async def producer_launch_guard(
    app: CayuApp, command: ProducerOutputRegistration, *, context: MandateAccessContext
):
    """Private runtime handoff; keep the guard until native admission settles.

    This does not dispatch work. Its consumer must consume the source election
    through the native admission/exclusion fence, not treat the record as a
    bearer token or release the guard before the native transaction.
    """
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    require_exact_contract(command.initiator, _initiator(context), redactor=redactor)
    coordinator = app._request_coordinator
    configured = coordinator._registration
    receiver = None if configured is None else configured.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer launch requires the registered native receiver.")
    require_exact_contract(command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor)
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    if not app.session_store._supports_producer_attachment_protocol():
        raise CollaborationUnavailable("Native producer admission is not qualified.")
    if not app._run_limit_controller._supports_producer_budget_readback():
        raise CollaborationUnavailable("Producer accounting readback is not qualified.")
    _require_delivery_configuration(app, command)
    async with receiver._acquire_producer_execution(command.admission, context=context) as raw:
        authority = prepare_contract(RequestReceivingAuthorization, raw, redactor=redactor)
        require_exact_contract(authority.command, command.admission, redactor=redactor)
        require_exact_contract(authority.receiver, command.receiver, redactor=redactor)
        store, initialized = app._participant_coordinator._ready()
        app._participant_coordinator._capability(
            store, initialized, mutation=True, family=REQUEST_FAMILY
        )
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            record = await claim_output_launch_in_transaction(
                store,
                tx,
                initialized,
                command,
                authority_expires_at_ms=authority.expires_at_ms,
                redactor=redactor,
            )
        yield record


async def exclude_prepared_producer(
    app: CayuApp,
    command: ProducerOutputRegistration,
    control: RequestControlCommand,
    *,
    context: MandateAccessContext,
):
    """Authenticate retained closure before handing exclusion to the native owner.

    This is cleanup only: it does not renew preparation or execution authority,
    and it does not settle the source permit. The latter requires the exact
    native decision to be acknowledged by the source responsibility owner.
    """
    coordinator = app._request_coordinator
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    control = prepare_contract(RequestControlCommand, control, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    require_exact_contract(command.admission.expected, control.intent.expected, redactor=redactor)
    if not app.session_store._supports_producer_attachment_protocol():
        raise CollaborationUnavailable("Native producer exclusion is not qualified.")

    async def exclude():
        # A caller-supplied receipt, even one equal to retained data, is not the
        # authentication boundary. Resolve the exact command with current read
        # access and use only the owner's authenticated result.
        found = await coordinator.lookup(control, context=context)
        if not isinstance(found, ExactMatch) or not isinstance(
            found.receipt, RequestControlReceipt
        ):
            raise CollaborationUnavailable("Producer closure evidence is unavailable.")
        require_exact_contract(control, found.receipt.expected, redactor=redactor)
        store, initialized = app._participant_coordinator._ready()
        app._participant_coordinator._capability(
            store, initialized, mutation=False, family=REQUEST_FAMILY
        )
        require_exact_contract(
            command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor
        )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            retained = await read_output_registration(tx, command, redactor=redactor)
        if retained is None:
            raise CollaborationUnavailable("Producer registration evidence is unavailable.")
        return await app.session_store._exclude_native_producer(retained, found.receipt)

    async def owned():
        return await coordinator._dependency(exclude)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_exclusion", object()),
            expectation=contract_bytes(command, redactor=redactor)
            + contract_bytes(control, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

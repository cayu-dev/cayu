"""Retained producer obligations serviced through the existing peer entrance."""

from cayu.collaboration._clarification_export import acquire_clarification_source
from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_delivery_store import (
    delivery_operation,
    prepare_delivery,
    reconcile_delivery,
)
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import (
    PeerContentAppendRequest,
    PeerContentOccurrence,
    PeerContentPayload,
)
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget


async def deliver_producer_output(
    app, command, destination_operation, *, context, prepare_only=False
):
    """One owned attempt; append acceptance is not exposure or producer settlement."""
    coordinator = app._request_coordinator
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    destination_operation = prepare_contract(OperationRef, destination_operation, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
    if type(prepare_only) is not bool:
        raise CollaborationAccessDenied("Producer delivery preparation must be explicit.")
    destination = next(
        (item for item in command.destinations if item.operation == destination_operation), None
    )
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    target = prepared.target
    if (
        destination is None
        or context.mandate is None
        or context.mandate.participant != prepared.recipient
    ):
        raise CollaborationAccessDenied(
            "Producer delivery requires its exact participant authority."
        )
    require_exact_contract(
        command.initiator.model_copy(update={"mandate": destination.mandate}),
        _initiator(context.mandate),
        redactor=redactor,
    )
    store, initialized = app._participant_coordinator._ready()
    exports = app._session_export_coordinator
    if exports.mandate_ref != coordinator._resolver_ref:
        raise CollaborationAccessDenied("Producer delivery resolver conflicts.")
    peer_context = CollaborationAccessContext(principal=context.principal)

    async def deliver():
        participants = app._participant_coordinator
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        _, grant = participants._authorize(peer_context, "request_accept")
        participants._require_refs(grant, (prepared.recipient, destination.recipient))
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            exported = await read_export(tx, command, destination, redactor=redactor)
        if exported is None or exported.state != "published":
            raise CollaborationUnavailable("Producer delivery requires published output.")
        async with acquire_clarification_source(
            exports,
            exported.request,
            context=context,
            sender=prepared.recipient,
            audience=destination.recipient,
        ) as source:
            payload_data = {"text": source.text, "artifact_commitments": []}
            payload = prepare_contract(
                PeerContentPayload,
                {**payload_data, "content_sha256": digest(payload_data)},
                redactor=redactor,
            )
            occurrence_data = {
                "occurrence_id": destination.attempt.append_key.occurrence_id,
                "sender_participant_id": prepared.recipient.participant_id,
                "sender_participant_incarnation": prepared.recipient.incarnation,
                "sender_session_id": target.session_id,
                "sender_session_instance_id": target.session_instance_id,
                "producer_receipt_id": command.operation.caller_key,
                "source_export_receipt_id": source.receipt.event_id,
                "payload": payload.model_dump(mode="json"),
                "audience": [destination.recipient.participant_id],
            }
            occurrence = prepare_contract(
                PeerContentOccurrence,
                {**occurrence_data, "provenance_sha256": digest(occurrence_data)},
                redactor=redactor,
            )
            append = PeerContentAppendRequest(
                operation_key=delivery_operation(destination, redactor).caller_key,
                append_key=destination.attempt.append_key,
                attempt_key=destination.attempt,
                occurrence=occurrence,
            )
            async with store._transaction(initialized.owner.application_scope, write=True) as tx:
                preparation = await prepare_delivery(
                    store,
                    tx,
                    initialized,
                    command,
                    destination,
                    append,
                    source.receipt,
                    authority_expires_at_ms=source.expires_at_ms,
                    redactor=redactor,
                )
        # Source responsibility survives releasing this guard. Peer append has
        # its own registered revocation guard; never recursively acquire it.
        if prepare_only:
            return preparation
        current = await reconcile_delivery(
            store,
            initialized,
            command,
            destination,
            app.session_store,
            redactor=redactor,
        )
        if current.receipt is None:
            await app.append_peer_content(current.append, context=peer_context)
            current = await reconcile_delivery(
                store,
                initialized,
                command,
                destination,
                app.session_store,
                redactor=redactor,
            )
        return current

    async def owned():
        return await coordinator._dependency(deliver)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_delivery", object()),
            expectation=contract_bytes(command, redactor=redactor)
            + contract_bytes(destination, redactor=redactor)
            + contract_bytes(context, redactor=redactor)
            + (b"prepare" if prepare_only else b"deliver"),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

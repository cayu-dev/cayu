"""Content-free receiving acknowledgement recovery under current owner access."""

from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import CollaborationConflict, ContractValue, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_bounds import supports_peer_delivery
from cayu.collaboration._producer_contracts import ProducerOutputRecord
from cayu.collaboration._producer_delivery_store import read_delivery, reconcile_delivery
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.exports import SessionExportDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import NativeCommitment


class ProducerDeliveryRecovery(ProducerOutputRecovery):
    """Expected immutable registration; neither disclosure nor dispatch authority."""

    destination: OperationRef


class ProducerDeliveryStatus(ContractValue):
    """No text, export receipt, execution authority or exposure claim."""

    recovery: ProducerDeliveryRecovery
    operation: OperationRef
    state: Literal["pending", "appended", "excluded"]
    receipt_commitment: NativeCommitment | None = None

    @model_validator(mode="after")
    def exact_receiving_state(self):
        if (self.state == "pending") != (self.receipt_commitment is None):
            raise ValueError("Producer delivery status requires exact terminal decision evidence.")
        return self


async def reconcile_producer_delivery(
    app, recovery, *, context, exclude=False, wait_for_settlement=False
):
    """Reconcile or explicitly fence one attempt; never append, expose or release output."""
    requests = app._request_coordinator
    participants = app._participant_coordinator
    redactor = app._secret_redactor
    recovery = prepare_contract(ProducerDeliveryRecovery, recovery, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    if type(exclude) is not bool:
        raise CollaborationConflict("Producer recovery exclusion must be explicit.")
    store, initialized = participants._ready()

    async def reconcile():
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        _, grant = participants._authorize(context, "request_control")
        if (
            recovery.registration.application_scope != initialized.owner.application_scope
            or recovery.registration.namespace_incarnation != initialized.namespace_incarnation
        ):
            raise CollaborationConflict("Producer recovery belongs to another namespace.")
        async with store._transaction(initialized.owner.application_scope, write=exclude) as tx:
            raw = await tx.get("operations", operation_key(recovery.registration))
            if raw is None:
                raise CollaborationUnavailable(
                    "Producer responsibility is unavailable, not excluded."
                )
            candidate = prepare_contract(ProducerOutputRecord, raw, redactor=redactor)
            command = candidate.command
            if command.operation != recovery.registration or (
                "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
                != recovery.registration_commitment
            ):
                raise CollaborationConflict("Producer recovery requires its exact registration.")
            destination = next(
                (item for item in command.destinations if item.operation == recovery.destination),
                None,
            )
            if destination is None:
                raise CollaborationConflict("Producer recovery destination is not registered.")
            prepared = command.admission.prepared
            assert prepared is not None
            participants._require_refs(grant, (prepared.recipient, destination.recipient))
            retained = await read_output_registration(tx, command, redactor=redactor)
            if retained != candidate:
                raise CollaborationUnavailable("Producer recovery responsibility conflicts.")
            delivery = await read_delivery(tx, command, destination, redactor=redactor)
            if delivery is None:
                from cayu.collaboration._producer_destination_exclusion import (
                    exclude_unprepared_destination,
                    read_destination_exclusion,
                )

                excluded = await read_destination_exclusion(
                    tx, command, destination, redactor=redactor
                )
                if excluded is None and exclude:
                    excluded = await exclude_unprepared_destination(
                        store, tx, initialized, command, destination, redactor=redactor
                    )
                if excluded is None:
                    raise CollaborationUnavailable(
                        "Producer delivery is unavailable, not excluded."
                    )
                return ProducerDeliveryStatus(
                    recovery=recovery,
                    operation=excluded.operation,
                    state="excluded",
                    receipt_commitment="sha256:"
                    + sha256(contract_bytes(excluded, redactor=redactor)).hexdigest(),
                )
        if exclude and delivery.receipt is None:
            if not supports_peer_delivery(app.session_store):
                raise CollaborationUnavailable("Receiving store cannot fence producer delivery.")
            try:
                async with app._session_export_coordinator.acquire_peer_exclusion(
                    context,
                    request=delivery.append,
                    receipt=delivery,
                    reason="withdrawn",
                ):
                    await app.session_store.exclude_peer_content(
                        delivery.append, reason="withdrawn"
                    )
            except SessionExportDenied:
                raise CollaborationAccessDenied(
                    "Producer delivery cleanup is not authorized."
                ) from None
        current = await reconcile_delivery(
            store,
            initialized,
            command,
            destination,
            app.session_store,
            redactor=redactor,
        )
        receipt = current.receipt
        return ProducerDeliveryStatus(
            recovery=recovery,
            operation=current.operation,
            state="pending" if receipt is None else receipt.status,
            receipt_commitment=None
            if receipt is None
            else "sha256:" + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest(),
        )

    async def owned():
        return await requests._dependency(reconcile)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_delivery_recovery", object()),
            expectation=contract_bytes(recovery, redactor=redactor)
            + contract_bytes(context, redactor=redactor)
            + (b"exclude" if exclude else b"reconcile"),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )

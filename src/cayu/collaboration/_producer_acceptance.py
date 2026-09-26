"""Registered exact peer acceptance reader; never execution-settlement evidence."""

from hashlib import sha256

from cayu.collaboration._contracts import (
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_bounds import supports_peer_delivery
from cayu.collaboration._producer_contracts import ProducerDeliveryIndex, ProducerOutputRecord
from cayu.collaboration._producer_delivery_store import acceptance_index_operation, read_delivery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.exports import (
    SessionExportAcceptance,
    SessionExportAcceptanceReader,
    SessionExportReceipt,
)
from cayu.collaboration.peer_content import PeerContentReceipt
from cayu.vaults.redaction import SecretRedactor


class ProducerOutputAcceptanceReader(SessionExportAcceptanceReader):
    """Pinned source namespace, receiving store and audience configured by the host.

    The index locates evidence; it does not authenticate caller receipts. Both
    complete source registration and the exact native receiving attempt must
    match. This reader deliberately inherits unavailable execution settlement.
    """

    def __init__(self, *, collaboration_store, session_store, namespace, audience, redactor=None):
        self._redactor = SecretRedactor() if redactor is None else redactor
        self._namespace = prepare_contract(OperationRef, namespace, redactor=self._redactor)
        self._audience = prepare_contract(OwnerRef, audience, redactor=self._redactor)
        if self._namespace.application_scope != self._audience.application_scope:
            raise ValueError("Producer acceptance registration belongs to another scope.")
        self._collaboration = collaboration_store
        self._sessions = session_store

    @property
    def owner(self) -> OwnerRef:
        return self._audience

    def _require_producer_namespace(self, operation: OperationRef) -> None:
        """Check routing coverage, not current disclosure or execution authority."""
        from cayu.collaboration.participants import CollaborationUnavailable

        expected = self._namespace.model_copy(update={"caller_key": operation.caller_key})
        if operation != expected or not supports_peer_delivery(self._sessions):
            raise CollaborationUnavailable("Producer receiving readback is not qualified.")

    async def lookup(self, receipt: SessionExportReceipt) -> ExactLookup[SessionExportAcceptance]:
        redactor = self._redactor
        receipt = prepare_contract(SessionExportReceipt, receipt, redactor=redactor)
        request = receipt.expected.intent.request
        if request.audience != self._audience:
            return ExactConflict()
        operation = acceptance_index_operation(self._namespace, request.ref, redactor)
        async with self._collaboration._transaction(
            self._namespace.application_scope, write=False
        ) as tx:
            raw = await tx.get("operations", operation_key(operation))
            if raw is None:
                return ExactNotFound()
            index = prepare_contract(ProducerDeliveryIndex, raw, redactor=redactor)
            if (
                index.operation != operation
                or index.export_receipt_commitment
                != "sha256:" + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest()
            ):
                return ExactConflict()
            raw = await tx.get("operations", operation_key(index.registration))
            if raw is None:
                return ExactUnavailable()
            registration = prepare_contract(ProducerOutputRecord, raw, redactor=redactor)
            retained = await read_output_registration(tx, registration.command, redactor=redactor)
            if retained != registration or registration.command.operation != index.registration:
                return ExactUnavailable()
            destination = next(
                (
                    item
                    for item in registration.command.destinations
                    if item.operation == index.destination
                ),
                None,
            )
            if destination is None:
                return ExactConflict()
            delivery = await read_delivery(tx, registration.command, destination, redactor=redactor)
            if (
                delivery is None
                or delivery.operation != index.delivery
                or delivery.operation not in registration.deliveries
            ):
                return ExactUnavailable()
            require_exact_contract(receipt, delivery.source_receipt, redactor=redactor)
        if not supports_peer_delivery(self._sessions):
            return ExactUnavailable()
        raw = await self._sessions.read_peer_content_attempt(delivery.append)
        if raw is None:
            return ExactUnavailable()
        received = prepare_contract(PeerContentReceipt, raw, redactor=redactor)
        expected = delivery.append
        if (
            received.operation_key != expected.operation_key
            or received.append_key != expected.append_key
            or received.attempt_generation != expected.attempt_key.attempt_generation
        ):
            return ExactConflict()
        if received.status != "appended" or received.disclosure != "available":
            return ExactUnavailable()
        if received.occurrence != expected.occurrence or (
            expected.append_key.creation_target is None
            and (received.target_session_id, received.target_session_instance_id)
            != (
                expected.append_key.target_session_id,
                expected.append_key.target_session_instance_id,
            )
        ):
            return ExactConflict()
        receiving_id = (
            "peer:"
            + sha256(
                contract_bytes(received.model_copy(update={"replayed": False}), redactor=redactor)
            ).hexdigest()
        )
        return ExactMatch[SessionExportAcceptance](
            receipt=SessionExportAcceptance(
                export_receipt=receipt,
                receiving_owner=self._audience,
                receipt_id=receiving_id,
            )
        )

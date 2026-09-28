"""Cleanup-only export retirement from authenticated closed producer responsibility.

The source owner supplies a private, exact handoff. It is not an export,
disclosure, mandate-renewal or acquisition grant. Native publication continues
to use the existing export transaction and participant settlement owners.
"""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_bounds import supports_peer_delivery
from cayu.collaboration._producer_contracts import (
    ProducerDeliveryRecord,
    ProducerExportRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_destination_exclusion import ProducerDestinationExclusion
from cayu.collaboration._session_export_publication import publish_export_mutation
from cayu.collaboration._session_export_store import (
    ExportPreparation,
    ExportRecord,
    SettlementRecord,
    operation_key,
    read_scope,
)
from cayu.collaboration.exports import (
    SessionExportConflict,
    SessionExportSettlementReceipt,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.peer_content import PeerContentReceipt
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.collaboration.requests import RequestControlReceipt
from cayu.events import EventType

_SEAL = object()


async def read_retired_export_native(
    sessions, command, closure, intent, *, redactor, allow_pending=False
):
    """Fixed native-owner readback; historical cleanup grants no content access."""
    if type(allow_pending) is not bool:
        raise TypeError("Producer retirement observation requires an explicit mode.")
    if not sessions._supports_producer_attachment_protocol():
        raise SessionExportUnavailable()
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    closure = prepare_contract(
        ProducerDestinationExclusion
        if isinstance(closure, ProducerDestinationExclusion)
        else RequestControlReceipt,
        closure,
        redactor=redactor,
    )
    intent = prepare_contract(ProducerExportRecord, intent, redactor=redactor)
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    request = intent.request
    if intent.registration != command.operation or (
        request.ref.session_id,
        request.ref.session_instance_id,
    ) != (prepared.target.session_id, prepared.target.session_instance_id):
        raise SessionExportConflict()
    with read_scope(request.ref.session_id):
        raw = await sessions.load_session_operation(
            request.ref.session_id, operation_key(request.ref.operation)
        )
    from cayu.collaboration._producer_export_absence import read_exclusion

    future = read_exclusion(raw, command, closure, intent, redactor=redactor)
    if future is not None:
        return future
    if type(raw) is dict and "admission" in raw and "receipt" not in raw:
        retained = prepare_contract(ExportPreparation, raw, redactor=redactor)
        pending = retained.state != "excluded" or not retained.admission.settled
        require_exact_contract(request, retained.admission.request, redactor=redactor)
        authorization = retained.admission.authorization
        admission = retained.admission
    else:
        retained = prepare_contract(ExportRecord, raw, redactor=redactor)
        pending = retained.state == "pending" and retained.settlement is None
        if not pending and (
            retained.state not in ("retired", "released") or retained.settlement is None
        ):
            raise SessionExportUnavailable()
        require_exact_contract(request, retained.receipt.expected.intent.request, redactor=redactor)
        admission = retained.admission
        authorization = retained.receipt.expected.intent.authorization
        if not pending:
            assert retained.settlement is not None
            with read_scope(request.ref.session_id):
                raw = await sessions.load_session_operation(
                    request.ref.session_id, operation_key(retained.settlement.request.operation)
                )
            settlement = prepare_contract(SettlementRecord, raw, redactor=redactor)
            require_exact_contract(retained.settlement, settlement.settlement, redactor=redactor)
    if admission is None or admission.permit.intent.request.participant != prepared.recipient:
        raise SessionExportUnavailable()
    require_exact_contract(intent.initiator, authorization.initiating_identity(), redactor=redactor)
    if pending or not admission.settled:
        if allow_pending:
            return None
        raise SessionExportUnavailable()
    return retained


def _expectation(command, closure, intent, redactor):
    return tuple(contract_bytes(value, redactor=redactor) for value in (command, closure, intent))


@dataclass(frozen=True)
class _ExportRetirementAuthority:
    owner: object
    expected: tuple[bytes, ...]
    seal: object

    def require(self, owner, command, closure, intent):
        if (
            self.seal is not _SEAL
            or self.owner is not owner
            or self.expected != _expectation(command, closure, intent, owner.redactor)
        ):
            raise PermissionError("Export cleanup requires exact source-owned responsibility.")


def _received_export_retirement(owner, command, closure, intent):
    """Called only after source-owner readback proves closure and no delivery intent."""
    return _ExportRetirementAuthority(
        owner, _expectation(command, closure, intent, owner.redactor), _SEAL
    )


def _received_delivery_retirement(owner, command, delivery, intent):
    """Source-owner handoff; the native owner independently rechecks exclusion."""
    return _ExportRetirementAuthority(
        owner, _expectation(command, delivery, intent, owner.redactor), _SEAL
    )


async def _require_delivery_exclusion(exports, command, delivery, intent):
    destination = next(
        (item for item in command.destinations if item.operation == delivery.destination), None
    )
    if (
        destination is None
        or delivery.registration != command.operation
        or delivery.export != intent.operation
        or delivery.destination != intent.destination
        or intent.state != "published"
        or delivery.receipt is None
        or delivery.receipt.status != "excluded"
        or delivery.acceptance is None
        or delivery.append.attempt_key != destination.attempt
        or delivery.source_receipt.expected.intent.request != intent.request
        or not supports_peer_delivery(exports.store)
    ):
        raise SessionExportConflict()
    raw = await exports.store.read_peer_content_attempt(delivery.append)
    if raw is None:
        raise SessionExportUnavailable()
    received = prepare_contract(PeerContentReceipt, raw, redactor=exports.redactor)
    if received.model_copy(update={"replayed": False}) != delivery.receipt:
        raise SessionExportUnavailable()


async def retire_producer_export_native(exports, command, closure, intent, *, authority):
    """Exclude retained preparation or retire publication, without reprojecting.

    A missing native export requires a durable no-late-publication tombstone;
    it cannot be settled by inferring exclusion from absence.
    """
    redactor = exports.redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    delivery_exclusion = isinstance(closure, ProducerDeliveryRecord)
    source_exclusion = isinstance(closure, ProducerDestinationExclusion)
    closure = prepare_contract(
        ProducerDeliveryRecord
        if delivery_exclusion
        else ProducerDestinationExclusion
        if source_exclusion
        else RequestControlReceipt,
        closure,
        redactor=redactor,
    )
    intent = prepare_contract(ProducerExportRecord, intent, redactor=redactor)
    if type(authority) is not _ExportRetirementAuthority:
        raise PermissionError("Producer export cleanup requires its registered source owner.")
    authority.require(exports, command, closure, intent)
    if not exports.store._supports_producer_attachment_protocol():
        raise SessionExportUnavailable()
    if intent.registration != command.operation:
        raise PermissionError("Producer export cleanup is outside its closed responsibility.")
    if delivery_exclusion:
        await _require_delivery_exclusion(exports, command, closure, intent)
    elif source_exclusion:
        assert isinstance(closure, ProducerDestinationExclusion)
        if (
            closure.registration != command.operation
            or closure.destination != intent.destination
            or closure.registration_commitment
            != "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
        ):
            raise SessionExportConflict()
    else:
        assert isinstance(closure, RequestControlReceipt)
        require_exact_contract(
            command.admission.expected, closure.expected.intent.expected, redactor=redactor
        )
        if closure.state not in ("cancelled", "expired"):
            raise PermissionError("Producer export cleanup requires durable closure.")
    exports.ready()
    request = intent.request
    session = await exports.session(request.ref.session_id, request.ref.session_instance_id)
    from cayu.collaboration._producer_export_absence import exclude_absent_export

    future = (
        None
        if delivery_exclusion
        else await exclude_absent_export(
            exports, session, command, closure, intent, authority=authority
        )
    )
    if future is not None:
        return future
    current = await exports._record(session, request)
    if current is None or exports.participants is None:
        raise SessionExportUnavailable()
    if delivery_exclusion:
        assert isinstance(closure, ProducerDeliveryRecord)
        if not isinstance(current, ExportRecord):
            raise SessionExportUnavailable()
        require_exact_contract(closure.source_receipt, current.receipt, redactor=redactor)
    authorization = (
        current.admission.authorization
        if isinstance(current, ExportPreparation)
        else current.receipt.expected.intent.authorization
    )
    require_exact_contract(intent.initiator, exports.initiator(authorization), redactor=redactor)

    # Attribution stays bound to the original accepted export responsibility.
    # Its historical mandate is not validated or used as a new acquisition grant.
    def guard(_now):
        authority.require(exports, command, closure, intent)

    async def publish(root, desired, old, updated):
        await publish_export_mutation(
            exports.store,
            session,
            root,
            desired,
            operation_key(request.ref.operation),
            updated.model_dump(mode="json"),
            [],
            commit_guard=guard,
            expected_old=old.model_dump(mode="json"),
        )

    async def complete(record):
        return await exports.participants._complete_admission(session, record, publish=publish)

    if isinstance(current, ExportPreparation):
        current = await exports.participants._exclude_preparation(
            session,
            request,
            excluded_by=exports.initiator(authorization),
            mandate_commitment=exports.mandate_commitment(authorization),
            publish=publish,
            complete=complete,
        )
        if isinstance(current, ExportPreparation):
            if current.state != "excluded" or not current.admission.settled:
                raise SessionExportUnavailable()
            return current
    current = await complete(current)
    if current.state in ("released", "retired"):
        return current
    settlement_request = SessionExportSettlementRequest(
        request=request,
        mode="retire",
        operation=request.ref.operation.model_copy(
            update={
                "caller_key": "producer-retirement:"
                + sha256(
                    b"".join(
                        contract_bytes(value, redactor=redactor)
                        for value in (command, closure, request)
                    )
                ).hexdigest(),
            }
        ),
    )
    event = exports.event(session.id, EventType.SESSION_EXPORT_RETIRED, current.receipt.expected)
    receipt = exports.prepare(
        SessionExportSettlementReceipt,
        {
            "request": settlement_request,
            "initiator": exports.initiator(authorization),
            "mandate_commitment": exports.mandate_commitment(authorization),
            "acceptance": None,
            "event_id": event.id,
        },
    )
    updated = exports.prepare(
        ExportRecord, current.model_copy(update={"state": "retired", "settlement": receipt})
    )
    key = operation_key(settlement_request.operation)
    settlement = SettlementRecord(settlement=receipt)
    for _ in range(4):
        root = await exports.root(session)
        observed = await exports._record(session, request)
        if isinstance(observed, ExportRecord) and observed.state in ("released", "retired"):
            return observed
        if observed != current or root is None or root.pending_count == 0:
            raise SessionExportConflict()
        try:
            await publish_export_mutation(
                exports.store,
                session,
                root,
                root.model_copy(update={"pending_count": root.pending_count - 1}),
                key,
                settlement.model_dump(mode="json"),
                [event],
                commit_guard=guard,
                additional_records={
                    operation_key(request.ref.operation): updated.model_dump(mode="json")
                },
            )
            return updated
        except Exception as error:
            observed = await exports._record(session, request)
            if isinstance(observed, ExportRecord) and observed.state in ("released", "retired"):
                return observed
            if not isinstance(error, SessionExportConflict):
                raise
    raise SessionExportConflict()

"""Explicit per-destination servicing through the registered session-export owner."""

from hashlib import sha256

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_completion import retain_producer_completion
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_export_store import (
    acknowledge_export,
    prepare_export,
    read_export,
    require_export_interest,
)
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportRef,
    SessionExportRequest,
)
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget


async def export_producer_output(app, command, destination_operation, *, context):
    """Do not dispatch production, elect an answer, append, expose or release pins."""
    coordinator = app._request_coordinator
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    destination_operation = prepare_contract(OperationRef, destination_operation, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
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
        or context.mandate.mandate != destination.mandate
    ):
        raise CollaborationAccessDenied("Producer export requires its exact destination authority.")
    require_exact_contract(
        command.initiator.model_copy(update={"mandate": destination.mandate}),
        _initiator(context.mandate),
        redactor=redactor,
    )
    initiator = _initiator(context.mandate)
    exports = app._session_export_coordinator
    exports.ready()
    if exports.mandate_ref != coordinator._resolver_ref:
        raise CollaborationAccessDenied("Producer export resolver registration conflicts.")
    store, initialized = app._participant_coordinator._ready()

    async def dispatch(intent):
        from cayu.collaboration._producer_output_failure import retain_output_failure
        from cayu.collaboration.exports import SessionExportDenied

        if await retain_output_failure(app, command, destination, intent) is not None:
            raise SessionExportDenied()
        found = await exports.lookup(intent.request, context=context)
        if isinstance(found, ExactMatch):
            return found.receipt
        if isinstance(found, ExactConflict):
            raise CollaborationConflict("Producer export conflicts with its native owner.")
        if not isinstance(found, (ExactNotFound, ExactUnavailable)):
            raise CollaborationUnavailable("Producer export readback is not qualified.")
        if intent.state == "published":
            raise CollaborationUnavailable("Acknowledged producer export is unavailable.")
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            await require_export_interest(
                store, tx, initialized, command, destination, intent, redactor=redactor
            )
        try:
            return await exports.export(intent.request, context=context)
        except Exception:
            await retain_output_failure(app, command, destination, intent)
            raise

    async def export():
        completion = await retain_producer_completion(app, command)
        if completion.output.disposition != "answer":
            raise CollaborationUnavailable("Producer output requires failure election, not export.")
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            retained = await read_export(tx, command, destination, redactor=redactor)
        if retained is not None:
            # An acknowledged or uncertain export has a frozen namespace/key.
            # Replay does not require namespace initialization or a new projector.
            receipt = await dispatch(retained)
            async with store._transaction(initialized.owner.application_scope, write=True) as tx:
                return await acknowledge_export(
                    store,
                    tx,
                    initialized,
                    command,
                    destination,
                    retained,
                    receipt,
                    redactor=redactor,
                )
        namespace = await exports.initialize(target.session_id, context=context)
        request = SessionExportRequest(
            ref=SessionExportRef(
                session_id=target.session_id,
                session_instance_id=target.session_instance_id,
                operation=OperationRef(
                    application_scope=namespace.owner.application_scope,
                    namespace_incarnation=namespace.namespace_incarnation,
                    generation=namespace.generation,
                    caller_key="producer:"
                    + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest(),
                ),
            ),
            source_indices=completion.output.source_indices,
            source_selection="assistant_visible_text_v1",
            audience=OwnerRef(
                application_scope=destination.recipient.owner.application_scope,
                owner_id=destination.recipient.participant_id,
                incarnation=destination.recipient.incarnation,
            ),
            projector=destination.projector,
            policy=destination.disclosure_policy,
        )
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            intent = await prepare_export(
                store,
                tx,
                initialized,
                command,
                destination,
                request,
                initiator,
                redactor=redactor,
            )
        # The export owner revalidates current policy and mandate, pins source and
        # resolves lost ACK under this exact request. No collaboration lock is held.
        receipt = await dispatch(intent)
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            return await acknowledge_export(
                store, tx, initialized, command, destination, intent, receipt, redactor=redactor
            )

    async def owned():
        return await coordinator._dependency(export)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_export", object()),
            expectation=contract_bytes(command, redactor=redactor)
            + contract_bytes(destination_operation, redactor=redactor)
            + contract_bytes(context, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

"""Explicit producer milestones through the existing request progress election."""

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_progress_contracts import (
    ProducerProgressEvidence,
    ProducerProgressOccurrence,
    ProducerProgressReference,
)
from cayu.collaboration._producer_progress_store import (
    _progress_authority,
    require_producer_progress,
)
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_arbitration import progress_in_transaction
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration._request_store import operation_key, require_request_event, retained_request
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext, MandateResolution, ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import ProducerProgressCommand, RequestProgressReceipt


async def record_producer_progress(app, registration, occurrence, *, context):
    redactor = app._secret_redactor
    registration = prepare_contract(ProducerOutputRegistration, registration, redactor=redactor)
    occurrence = prepare_contract(ProducerProgressOccurrence, occurrence, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    require_exact_contract(registration.initiator, _initiator(context), redactor=redactor)
    prepared = registration.admission.prepared
    assert prepared is not None
    if context.participant != prepared.recipient:
        raise CollaborationAccessDenied("Producer progress names another participant.")
    requests, participants = app._request_coordinator, app._participant_coordinator
    store, initialized = participants._ready()
    expected = registration.admission.expected

    async def read(tx):
        record = await read_output_registration(tx, registration, redactor=redactor)
        prior = await retained_request(
            store, tx, initialized, expected.intent.request, expected.initiator, redactor
        )
        if record is None or prior is None:
            raise CollaborationUnavailable("Producer progress responsibility is unavailable.")
        return record, prior

    async def replay(tx, prior):
        raw = await tx.get("operations", operation_key(occurrence.operation))
        if raw is None:
            return None
        receipt = prepare_contract(RequestProgressReceipt, raw, redactor=redactor)
        command = receipt.command
        if not isinstance(command, ProducerProgressCommand) or (
            command.producer != registration.operation
            or command.expected != expected
            or command.publisher != registration.initiator
            or (command.operation, command.expected_revision, command.sequence, command.kind)
            != (
                occurrence.operation,
                occurrence.expected_revision,
                occurrence.sequence,
                occurrence.kind,
            )
            or ProducerProgressReference.from_receipt(receipt, redactor=redactor)
            not in prior.progress
        ):
            raise CollaborationConflict("Producer progress replay conflicts.")
        await require_producer_progress(tx, prior, command, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt

    async def record():
        participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
        await requests._authorize_retained_source(expected, context=context)
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            retained, prior = await read(tx)
            previous = await replay(tx, prior)
            if previous is not None:
                return previous
        configured = requests._registration
        receiver = None if configured is None else configured.receiving_owner
        if type(receiver) is not RecipientAdmissionReceivingOwner or requests._resolver_ref is None:
            raise CollaborationUnavailable("Native progress owner is not registered.")
        require_exact_contract(receiver.ref, registration.receiver, redactor=redactor)
        require_exact_contract(configured.mandates.ref, requests._resolver_ref, redactor=redactor)
        require_exact_contract(requests.prepared_receiver_ref(), receiver.ref, redactor=redactor)
        _, grant = participants._authorize(
            CollaborationAccessContext(principal=context.principal), "request_accept"
        )
        participants._require_refs(
            grant, (prepared.recipient, expected.intent.selection.sender.reference)
        )
        async with configured.mandates.acquire(context) as raw:
            resolution = prepare_contract(MandateResolution, raw, redactor=redactor)
            async with store._transaction(initialized.owner.application_scope, write=False) as tx:
                now = await tx.now_ms()
            validate_mandate_resolution(
                resolution,
                context=context,
                resolver=requests._resolver_ref,
                use=MandateUse(
                    audience=initialized.owner,
                    scope=initialized.owner.application_scope,
                    actions=("readback", "publish"),
                    resources=tuple(
                        ResourceSelector(resource=ref) for ref in expected.intent.request.inputs
                    ),
                    inputs=tuple(
                        MandateInput(source=ref, channel="prompt")
                        for ref in expected.intent.request.inputs
                    ),
                ),
                now_ms=now,
                resource_owners=requests._resource_owners,
                redactor=redactor,
            )
            evidence = prepare_contract(
                ProducerProgressEvidence,
                await receiver._read_producer_progress(retained, kind=occurrence.kind),
                redactor=redactor,
            )
            command = ProducerProgressCommand(
                operation=occurrence.operation,
                expected=expected,
                expected_revision=occurrence.expected_revision,
                admission_generation=registration.publisher_generation,
                publisher=registration.initiator,
                sequence=occurrence.sequence,
                kind=occurrence.kind,
                commitment=evidence.native_commitment,
                producer=registration.operation,
                evidence=evidence,
            )
            authority = _progress_authority(
                command,
                expires_at_ms=min(
                    registration.limits.deadline_at_ms,
                    resolution.principal.expires_at_ms,
                    *(entry.expires_at_ms for entry in resolution.chain.entries),
                ),
                redactor=redactor,
            )
            participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
            async with store._transaction(initialized.owner.application_scope, write=True) as tx:
                _, prior = await read(tx)
                previous = await replay(tx, prior)
                if previous is not None:
                    return previous
                participant = await store._participant(
                    tx, prepared.recipient, initialized.owner, redactor
                )
                if participant.lifecycle != "active" or (
                    participant.lifecycle_revision,
                    participant.configuration_revision,
                    participant.admission_generation,
                ) != (
                    prepared.lifecycle_revision,
                    prepared.configuration_revision,
                    prepared.admission_generation,
                ):
                    raise CollaborationAccessDenied("Producer progress participant changed.")
                return await progress_in_transaction(
                    store, tx, initialized, command, producer_authority=authority, redactor=redactor
                )

    async def owned():
        return await requests._dependency(record)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_progress", object()),
            expectation=contract_bytes(registration, redactor=redactor)
            + contract_bytes(occurrence, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

"""Registered producer publication through the existing request arbitration owner."""

from contextlib import asynccontextmanager
from hashlib import sha256

from cayu.collaboration._clarification_export import acquire_clarification_source
from cayu.collaboration._contracts import CollaborationConflict, OperationRef
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_completion import retain_producer_completion
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._producer_outcome_store import (
    _publication_authority,
    outcome_operation,
    require_producer_outcome,
)
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_arbitration import outcome_in_transaction
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.mandates import MandateResolution, ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import ProducerOutcomeCommand, RequestOutcomeReceipt


async def publish_producer_outcome(
    app, registration, *, destination_operation=None, context, wait_for_settlement=False
):
    """Elect once; leave mandatory delivery, native effects and budget cleanup owned."""
    requests = app._request_coordinator
    redactor = app._secret_redactor
    registration = prepare_contract(ProducerOutputRegistration, registration, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
    if destination_operation is not None:
        destination_operation = prepare_contract(
            OperationRef, destination_operation, redactor=redactor
        )
    destination = next(
        (item for item in registration.destinations if item.operation == destination_operation),
        None,
    )
    prepared = registration.admission.prepared
    assert prepared is not None
    if (
        context.mandate is None
        or context.mandate.participant != prepared.recipient
        or (destination_operation is not None and destination is None)
    ):
        raise CollaborationAccessDenied("Producer publication requires its exact participant.")
    initiating = _initiator(context.mandate)
    require_exact_contract(
        registration.initiator.model_copy(
            update={
                "mandate": registration.initiator.mandate
                if destination is None
                else destination.mandate
            }
        ),
        initiating,
        redactor=redactor,
    )
    store, initialized = app._participant_coordinator._ready()
    operation = outcome_operation(registration, redactor)
    expected = registration.admission.expected

    async def read(tx):
        record = await read_output_registration(tx, registration, redactor=redactor)
        prior = await retained_request(
            store, tx, initialized, expected.intent.request, expected.initiator, redactor
        )
        if record is None or prior is None:
            raise CollaborationUnavailable("Producer publication responsibility is unavailable.")
        return record, prior

    async def replay(tx, prior):
        raw = await tx.get("operations", operation_key(operation))
        if raw is None:
            return None
        receipt = prepare_contract(RequestOutcomeReceipt, raw, redactor=redactor)
        command = receipt.command
        if not isinstance(command, ProducerOutcomeCommand) or (
            command.producer != registration.operation
            or command.destination != destination_operation
            or command.initiator != initiating
            or prior.outcome != receipt
        ):
            raise CollaborationConflict("Producer outcome identity conflicts.")
        await require_producer_outcome(tx, prior, command, redactor=redactor)
        return receipt

    async def validate(resolution):
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            now = await tx.now_ms()
        assert context.mandate is not None
        return validate_mandate_resolution(
            resolution,
            context=context.mandate,
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

    @asynccontextmanager
    async def authorize(completion, exported):
        if exported is not None:
            assert destination is not None
            exports = app._session_export_coordinator
            if exports.mandate_ref != requests._resolver_ref:
                raise CollaborationAccessDenied("Producer publication resolver conflicts.")
            # This existing source-owner guard is also used by clarification,
            # but no question, service or caller-produced receipt is invented.
            async with acquire_clarification_source(
                exports,
                exported.request,
                context=context,
                sender=prepared.recipient,
                audience=destination.recipient,
            ) as source:
                resolution = source.authorization.mandate
                if resolution is None:
                    raise CollaborationAccessDenied("Producer publication lacks a current mandate.")
                await validate(resolution)
                if (
                    "sha256:"
                    + sha256(contract_bytes(source.receipt, redactor=redactor)).hexdigest()
                    != exported.receipt_commitment
                    or "sha256:" + source.receipt.expected.intent.source_commitment
                    != completion.output.source_commitment
                ):
                    raise CollaborationConflict("Producer publication export changed.")
                yield min(
                    source.expires_at_ms,
                    registration.limits.deadline_at_ms,
                    destination.attempt.deadline_at_ms,
                )
        else:
            configured = requests._registration
            if configured is None or requests._resolver_ref is None:
                raise CollaborationUnavailable("Producer publication owner is not registered.")
            require_exact_contract(
                configured.mandates.ref, requests._resolver_ref, redactor=redactor
            )
            async with configured.mandates.acquire(context.mandate) as raw:
                resolution = prepare_contract(MandateResolution, raw, redactor=redactor)
                await validate(resolution)
                yield min(
                    registration.limits.deadline_at_ms,
                    resolution.principal.expires_at_ms,
                    *(entry.expires_at_ms for entry in resolution.chain.entries),
                )

    async def publish():
        assert context.mandate is not None
        # Historical election readback has current read permission, not renewed
        # source disclosure or a requirement to rerun native production.
        await requests._authorize_retained_source(expected, context=context.mandate)
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record, prior = await read(tx)
            existing = await replay(tx, prior)
            if existing is not None:
                return existing
        if record.completion is None:
            await retain_producer_completion(
                app, registration, wait_for_settlement=wait_for_settlement
            )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record, _ = await read(tx)
            completion = prepare_contract(
                ProducerCompletionRecord,
                await tx.get("operations", operation_key(record.completion)),
                redactor=redactor,
            )
            exported = (
                None
                if destination is None
                else await read_export(tx, registration, destination, redactor=redactor)
            )
            failure = None
            if destination is None and completion.output.disposition == "answer":
                from cayu.collaboration._producer_output_failure import read_output_failure

                for item in registration.destinations:
                    candidate = await read_export(tx, registration, item, redactor=redactor)
                    if candidate is not None:
                        failure = await read_output_failure(tx, candidate, redactor=redactor)
                        if failure is not None:
                            break
        if completion.output.disposition == "answer":
            if failure is None and (exported is None or exported.state != "published"):
                raise CollaborationUnavailable("Producer answer requires exact accepted export.")
        elif destination is not None:
            raise CollaborationConflict("Producer failure cannot elect an answer export.")
        app._participant_coordinator._capability(
            store, initialized, mutation=True, family=REQUEST_FAMILY
        )
        _, grant = app._participant_coordinator._authorize(
            CollaborationAccessContext(principal=context.principal), "request_accept"
        )
        app._participant_coordinator._require_refs(
            grant, (prepared.recipient, expected.intent.selection.sender.reference)
        )
        async with (
            authorize(completion, exported) as expiry,
            store._transaction(initialized.owner.application_scope, write=True) as tx,
        ):
            record, prior = await read(tx)
            existing = await replay(tx, prior)
            if existing is not None:
                return existing
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
                raise CollaborationAccessDenied("Producer publication participant changed.")
            command = ProducerOutcomeCommand(
                operation=operation,
                expected=expected,
                expected_revision=prior.revision,
                outcome="answered" if exported is not None else "failed",
                initiator=initiating,
                producer=registration.operation,
                completion=completion.operation,
                publisher_generation=registration.publisher_generation,
                terminal_frontier=len(prior.progress),
                native_commitment=completion.native_commitment,
                commitment=failure.native_commitment
                if failure is not None
                else completion.native_commitment
                if exported is None
                else exported.output_commitment,
                export=None if exported is None else exported.operation,
                destination=destination_operation,
                output_failure=None if failure is None else failure.operation,
            )
            authority = _publication_authority(command, expires_at_ms=expiry, redactor=redactor)
            return await outcome_in_transaction(
                store, tx, initialized, command, producer_authority=authority, redactor=redactor
            )

    async def owned():
        return await requests._dependency(publish)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_outcome", object()),
            expectation=contract_bytes(registration, redactor=redactor)
            + contract_bytes(context, redactor=redactor)
            + (
                b"failure"
                if destination_operation is None
                else contract_bytes(destination_operation, redactor=redactor)
            ),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )

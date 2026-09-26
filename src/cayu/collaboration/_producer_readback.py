"""Authenticated historical producer readback from an exact expectation or index token."""

from hashlib import sha256

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerOutputRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable


async def lookup_producer_registration(
    app,
    expected: ProducerOutputRegistration | ProducerOutputRecovery,
    *,
    context: CollaborationAccessContext,
) -> ExactLookup[ProducerOutputRegistration]:
    """Read the exact original registration without conferring execution authority."""
    return await _lookup_producer(app, expected, context=context, completion=False)


async def lookup_producer_completion(
    app,
    expected: ProducerOutputRegistration | ProducerOutputRecovery,
    *,
    context: CollaborationAccessContext,
) -> ExactLookup[ProducerCompletionRecord]:
    """Read retained completion references, not content or a new disclosure grant."""
    return await _lookup_producer(app, expected, context=context, completion=True)


async def _lookup_producer(app, expected, *, context, completion):
    """Recover the original immutable command; never mint a launch or content grant.

    The token is only a complete-command commitment and an indexed address.
    Current collaboration access and every affected participant are checked
    before returning the reconstructed value. Native execution and new disclosure
    still pass their respective registered receiving-owner boundaries.
    """
    redactor = app._secret_redactor
    if type(expected) not in (ProducerOutputRegistration, ProducerOutputRecovery):
        raise TypeError("Producer readback requires a typed exact expectation.")
    expected = prepare_contract(type(expected), expected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    participants, requests = app._participant_coordinator, app._request_coordinator
    store, initialized = participants._ready()
    operation = (
        expected.operation
        if isinstance(expected, ProducerOutputRegistration)
        else expected.registration
    )

    async def lookup():
        participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
        _, grant = participants._authorize(context, "request_readback")
        if (
            operation.application_scope != initialized.owner.application_scope
            or operation.namespace_incarnation != initialized.namespace_incarnation
        ):
            return ExactConflict()
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            raw = await tx.get("operations", operation_key(operation))
            if raw is None:
                return ExactNotFound()
            try:
                candidate = prepare_contract(ProducerOutputRecord, raw, redactor=redactor)
            except CollaborationContractError:
                return ExactUnavailable()
            command = candidate.command
            prepared = command.admission.prepared
            assert prepared is not None
            # An authenticated principal alone is not authority for another
            # participant's retained registration, including its budget identity.
            participants._require_refs(
                grant, (prepared.recipient, *(item.recipient for item in command.destinations))
            )
            if command.operation != operation:
                return ExactConflict()
            if isinstance(expected, ProducerOutputRegistration):
                if contract_bytes(command, redactor=redactor) != contract_bytes(
                    expected, redactor=redactor
                ):
                    return ExactConflict()
            elif expected.registration_commitment != (
                "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
            ):
                return ExactConflict()
            try:
                retained = await read_output_registration(tx, command, redactor=redactor)
            except CollaborationConflict:
                return ExactConflict()
            except (CollaborationContractError, CollaborationUnavailable):
                return ExactUnavailable()
            if retained != candidate:
                return ExactUnavailable()
            if completion:
                if retained.completion is None:
                    return ExactNotFound()
                completed = prepare_contract(
                    ProducerCompletionRecord,
                    await tx.get("operations", operation_key(retained.completion)),
                    redactor=redactor,
                )
                return ExactMatch[ProducerCompletionRecord](receipt=completed)
            # Do not expose the larger mutable bookkeeping record. Native-output
            # preflight already reserves a larger envelope around this command.
            return ExactMatch[ProducerOutputRegistration](receipt=command)

    async def owned():
        return await requests._dependency(lookup)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_readback", object()),
            expectation=contract_bytes(expected, redactor=redactor)
            + contract_bytes(context, redactor=redactor)
            + (b"completion" if completion else b"registration"),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

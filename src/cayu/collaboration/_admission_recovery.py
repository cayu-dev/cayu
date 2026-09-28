"""Reconstruct current admission from an exact source-owned request observation.

The discovered request is an expectation, not authority. This read never admits,
registers a producer, or upgrades historical evidence to execution permission.
"""

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._prepared_admission_store import lookup_admission_in_transaction
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestAdmissionReceipt, RequestSnapshot


async def recover_admission(coordinator, expected, *, context, wait_for_settlement=False):
    redactor = coordinator._redactor
    expected = prepare_contract(RequestSnapshot, expected, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)

    async def recover():
        # Authenticate before reading even content-free retained identities.
        command = expected.receipt.expected
        await coordinator._authorize_retained_source(
            command, context=context, wait_for_settlement=wait_for_settlement
        )
        store, initialized = coordinator._participants._ready()
        coordinator._participants._capability(
            store, initialized, mutation=False, family=REQUEST_FAMILY
        )
        try:
            async with store._transaction(initialized.owner.application_scope, write=False) as tx:
                current = await retained_request(
                    store, tx, initialized, command.intent.request, command.initiator, redactor
                )
                if current is None:
                    return ExactNotFound()
                if current != expected:
                    return ExactConflict()
                if current.admission_operation is None:
                    return ExactNotFound()
                raw = await tx.get("operations", operation_key(current.admission_operation))
                if raw is None:
                    return ExactUnavailable()
                receipt = prepare_contract(RequestAdmissionReceipt, raw, redactor=redactor)
                if (
                    receipt.command.operation != current.admission_operation
                    or receipt.command.expected != command
                    or receipt.command.generation != current.admission_generation
                    or receipt.command.decision != current.admission_decision
                ):
                    return ExactUnavailable()
                found = await lookup_admission_in_transaction(
                    store, tx, initialized, receipt.command, redactor=redactor
                )
                if not isinstance(found, ExactMatch) or found.receipt != receipt:
                    return ExactUnavailable()
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationContractError, CollaborationUnavailable):
            return ExactUnavailable()
        # Reuse the production reader's current mandate, per-operation grants,
        # owner-time expiry and full receipt/permit/event verification. Foreign
        # authority I/O is outside the source transaction. Subsequent attachment
        # must still pass its independent live receiver/native admission guards.
        return await coordinator.lookup_admission(
            receipt.command, context=context, wait_for_settlement=wait_for_settlement
        )

    async def owned():
        return await coordinator._dependency(recover)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("admission-recovery", object()),
            expectation=contract_bytes(expected, redactor=redactor)
            + contract_bytes(context, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )

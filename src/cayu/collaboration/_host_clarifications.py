"""Bounded host selection over native clarification service ownership."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from cayu.collaboration._clarification_service_api import (
    ClarificationServiceReceipt,
    ClarificationServiceRequest,
)
from cayu.collaboration._host_ownership import HostOperationIdentity, HostReconciledResult
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._preparation_progress import PreparationReadFailure
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.participants import CollaborationUnavailable


@dataclass(frozen=True, slots=True)
class _ClarificationRule:
    request: ClarificationServiceRequest
    context: SessionExportAccessContext
    delivery_context: SessionExportAccessContext | None = None


def snapshot_clarification_rule(app, rule):
    if type(rule) is not _ClarificationRule:
        raise TypeError("Host clarification selection must be typed.")
    redactor = app._secret_redactor
    request = prepare_contract(ClarificationServiceRequest, rule.request, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, rule.context, redactor=redactor)
    delivery_context = (
        None
        if rule.delivery_context is None
        else prepare_contract(
            SessionExportAccessContext,
            rule.delivery_context,
            redactor=redactor,
        )
    )
    size = sum(len(contract_bytes(value, redactor=redactor)) for value in (request, context))
    if delivery_context is not None:
        size += len(contract_bytes(delivery_context, redactor=redactor))
    return _ClarificationRule(request, context, delivery_context), size


@dataclass(frozen=True, slots=True)
class HostClarificationResult:
    dispatched: bool
    receipt: ClarificationServiceReceipt | None = None


def start_clarification(
    app, ownership, request, *, context, delivery_context, observation_deadline
):
    redactor = app._secret_redactor
    request = prepare_contract(ClarificationServiceRequest, request, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
    delivery_context = (
        None
        if delivery_context is None
        else prepare_contract(
            SessionExportAccessContext,
            delivery_context,
            redactor=redactor,
        )
    )
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host clarification requires a finite observation deadline.")
    encoded = contract_bytes(request, redactor=redactor)
    material = b"\0".join(
        (
            encoded,
            contract_bytes(context, redactor=redactor),
            b"none"
            if delivery_context is None
            else contract_bytes(delivery_context, redactor=redactor),
        )
    )
    identity = HostOperationIdentity(
        "clarification:" + sha256(contract_bytes(request.operation, redactor=redactor)).hexdigest(),
        sha256(material).hexdigest(),
    )

    async def action(stop):
        if stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline:
            return HostClarificationResult(False)
        # Current recipient/source disclosure, target identity, original wait,
        # participant and shared-budget gates remain in the native coordinator.
        receipt = await app._clarification_coordinator._service_owned(
            app,
            request,
            context=context,
            delivery_context=delivery_context,
        )
        if type(receipt) is PreparationReadFailure:
            return HostReconciledResult(HostClarificationResult(False), receipt.error)
        receipt = prepare_contract(ClarificationServiceReceipt, receipt, redactor=redactor)
        if (
            receipt.operation != request.operation
            or receipt.question != request.delivery.question.operation
            or receipt.session_id != request.delivery.append.append_key.target_session_id
            or receipt.session_instance_id
            != request.delivery.append.append_key.target_session_instance_id
            or receipt.state not in {"returned", "excluded"}
        ):
            raise CollaborationUnavailable("Host clarification lacks exact native return evidence.")
        return HostClarificationResult(True, receipt)

    async def reconcile():
        # Disclosure can expire after completion. Cleanup reads authenticate
        # current readback, not renewed disclosure or another service dispatch.
        receipt = await app._clarification_coordinator._inspect_service_owned(
            app, request, context=CollaborationAccessContext(principal=context.principal)
        )
        if receipt is None:
            return None
        receipt = prepare_contract(ClarificationServiceReceipt, receipt, redactor=redactor)
        if (
            receipt.operation != request.operation
            or receipt.question != request.delivery.question.operation
            or receipt.session_id != request.delivery.append.append_key.target_session_id
            or receipt.session_instance_id
            != request.delivery.append.append_key.target_session_instance_id
            or receipt.state not in {"returned", "excluded"}
        ):
            raise CollaborationUnavailable("Host clarification lacks exact native return evidence.")
        return HostClarificationResult(False, receipt)

    ownership.start(
        identity,
        role="execution",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_clarification(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostClarificationResult or (
        result.dispatched and result.receipt is None
    ):
        raise RuntimeError("Clarification owner returned invalid handoff evidence.")
    # Native invocation return is not question closure or original wait election.
    # Those duties retain their own durable source records and maintenance turns.
    ownership.release_settled(outcome.identity)
    return True

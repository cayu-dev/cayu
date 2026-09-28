"""Retained native producer-maintenance tasks shared by host service modes."""

from hashlib import sha256
from math import isfinite

from cayu.collaboration._host_ownership import HostOperationIdentity, HostOwnership
from cayu.collaboration._host_producer_maintenance import (
    _DISCLOSURE_ACTIONS,
    HostMaintenanceResult,
    HostProducerMaintenance,
    service_producer_maintenance,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.exports import SessionExportAccessContext


def producer_maintenance_key(intent, *, redactor):
    return "producer-maintenance:" + sha256(contract_bytes(intent, redactor=redactor)).hexdigest()


def start_producer_maintenance(
    app,
    ownership: HostOwnership,
    intent: HostProducerMaintenance,
    *,
    context: CollaborationAccessContext,
    disclosure_context: SessionExportAccessContext | None,
    observation_deadline: float,
) -> HostOperationIdentity:
    """Reserve local maintenance capacity before entering any receiving owner.

    Exact replay joins the original task; only explicit servicing can renew its
    local dispatch window. Changing current authority cannot join a task started
    for another principal. Native semantic identity remains the original intent;
    none of these local keys replace the durable operation or its generation.
    """
    redactor = app._secret_redactor
    intent = prepare_contract(HostProducerMaintenance, intent, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    disclosure_context = (
        None
        if disclosure_context is None
        else prepare_contract(SessionExportAccessContext, disclosure_context, redactor=redactor)
    )
    if (intent.action in _DISCLOSURE_ACTIONS) != (disclosure_context is not None):
        raise ValueError("Producer maintenance requires role-specific current access.")
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host maintenance requires a finite observation deadline.")
    encoded_intent = contract_bytes(intent, redactor=redactor)
    encoded_access = contract_bytes(context, redactor=redactor)
    encoded_disclosure = (
        b"null"
        if disclosure_context is None
        else contract_bytes(disclosure_context, redactor=redactor)
    )
    material = b"\0".join((encoded_intent, encoded_access, encoded_disclosure))
    identity = HostOperationIdentity(
        key=producer_maintenance_key(intent, redactor=redactor),
        commitment=sha256(material).hexdigest(),
    )

    async def action(stop):
        async def dispatch_window():
            return await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            )

        return await service_producer_maintenance(
            app,
            intent,
            context=context,
            disclosure_context=disclosure_context,
            observation_deadline=observation_deadline,
            stop=stop,
            dispatch_window=dispatch_window,
        )

    async def reconcile():
        from cayu.collaboration._host_producer_reconciliation import reconcile_maintenance

        retained = await reconcile_maintenance(app, intent, context=context)
        return None if retained is None else HostMaintenanceResult(True, retained)

    # Reserve the bounded native receipt as well as the retained intent before
    # dispatch. A small request does not imply a small returned owner envelope.
    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=len(material) + 64 * 1024,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_producer_maintenance(ownership: HostOwnership, outcome) -> bool:
    """Release this local turn only after the native adapter returned its result.

    The producer's durable responsibility is not released here. A failed or
    cancelled native await retains this local ownership until exact owner
    reconciliation; task completion alone is deliberately insufficient.
    """
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostMaintenanceResult or (
        result.dispatched != (result.receipt is not None)
    ):
        raise RuntimeError("Native producer maintenance returned invalid handoff evidence.")
    ownership.release_settled(outcome.identity)
    return True

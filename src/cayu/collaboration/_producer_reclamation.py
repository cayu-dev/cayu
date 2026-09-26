"""Current maintenance access plus positive source retirement authorizes native drain."""

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import LIFECYCLE_FAMILY
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._producer_retirement import (
    ProducerCleanupReclamation,
    ProducerCleanupRetirement,
    _accepted_source_retirement,
)


async def reclaim_producer_cleanup(app, namespace, *, context, limit=32):
    """Drain at most limit native ACKs, not issue a new per-operation cleanup grant.

    This is state-based maintenance: repeated calls may reclaim the next batch.
    A lost acknowledgement does not undo the fence or permit work to restart.
    No terminal status, caller receipt, or mere absence authorizes reclamation.
    """
    redactor = app._secret_redactor
    namespace = prepare_contract(NamespaceRef, namespace, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Producer cleanup reclamation requires a batch of 1 to 32 records.")
    participants, requests = app._participant_coordinator, app._request_coordinator
    store, initialized = participants._ready()
    _, grant = participants._authorize(context, "namespace_prune")
    participants._require_refs(grant, (), create=True)
    configured = requests._registration
    receiver = None if configured is None else configured.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer reclamation requires its registered native owner.")
    require_exact_contract(requests.prepared_receiver_ref(), receiver.ref, redactor=redactor)
    retirement = ProducerCleanupRetirement(receiver=receiver.ref, namespace=namespace)

    async def reclaim():
        participants._capability(store, initialized, mutation=False, family=LIFECYCLE_FAMILY)
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            anchor = await store._anchor(tx, initialized, redactor)
            if (
                namespace.owner != initialized.owner
                or namespace.namespace_incarnation != initialized.namespace_incarnation
                or namespace.generation > anchor.pruned_through
            ):
                raise CollaborationUnavailable("Producer source history is not durably reclaimed.")
        # Pruned-through is monotonic. No source lock is held across native IO.
        # Native publication/reclamation share a durable namespace mutation fence.
        authority = _accepted_source_retirement(retirement)
        result = prepare_contract(
            ProducerCleanupReclamation,
            await receiver._retire_producer_cleanup(retirement, authority=authority, limit=limit),
            redactor=redactor,
        )
        require_exact_contract(retirement, result.retirement, redactor=redactor)
        if result.removed > limit:
            raise CollaborationUnavailable("Producer reclamation exceeded its bounded request.")
        return result

    async def owned():
        return await requests._dependency(reclaim)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_reclamation", object()),
            expectation=contract_bytes(retirement, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )

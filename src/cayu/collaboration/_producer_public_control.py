"""Current public control access, separate from mandatory owner-internal cleanup."""

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._producer_cleanup import settle_producer_output
from cayu.collaboration._producer_cleanup_finalization import ProducerCleanupFinalized
from cayu.collaboration._producer_completion import retain_producer_completion
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY


def _public_control(app, command, context) -> ProducerOutputRegistration:
    """Snapshot and authorize before invoking an owned, already-registered operation.

    A receipt, recovery token or equal command does not confer control access.
    Conversely, owner-internal cleanup can continue after disclosure revocation;
    it must not be routed back through this public access check.
    """
    command = prepare_contract(ProducerOutputRegistration, command, redactor=app._secret_redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=app._secret_redactor)
    participants = app._participant_coordinator
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
    _, grant = participants._authorize(context, "request_control")
    if (
        command.operation.application_scope != initialized.owner.application_scope
        or command.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Producer control belongs to another namespace.")
    prepared = command.admission.prepared
    assert prepared is not None
    participants._require_refs(
        grant, (prepared.recipient, *(item.recipient for item in command.destinations))
    )
    return command


async def retain_public_producer_completion(
    app, command: ProducerOutputRegistration, *, context: CollaborationAccessContext
) -> ProducerCompletionRecord:
    command = _public_control(app, command, context)
    return await retain_producer_completion(app, command)


async def settle_public_producer_output(
    app, command: ProducerOutputRegistration, *, context: CollaborationAccessContext
) -> ProducerCleanupFinalized:
    command = _public_control(app, command, context)
    return await settle_producer_output(app, command)

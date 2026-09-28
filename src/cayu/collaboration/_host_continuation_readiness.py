"""Observe selected producer delivery before scheduling a native continuation.

An elected answer is historical completion, not evidence that its mandatory
input has reached this particular waiting incarnation. This is a read-only
scheduling gate; native continuation admission still owns execution authority.
"""

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._producer_destination_exclusion import read_destination_exclusion
from cayu.collaboration._producer_store import read_request_output
from cayu.collaboration._request_store import retained_request
from cayu.collaboration._wait_coordinator import _elected_latch
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.waits import WaitSnapshot, request_object_ref, wait_operation_key
from cayu.runtime._host_continuation_discovery import recover_session_continuation
from cayu.runtime._session_continuation import require_latch_identity, require_ticket_identity
from cayu.sessions.creation_fence import SessionCreationDecision


async def continuation_delivery_ready(app, expected, service, *, context):
    """Return false for positively pending delivery, never for missing evidence."""
    redactor = app._secret_redactor
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    retained = await recover_session_continuation(app, expected, context=context)
    require_ticket_identity(retained.ticket, service.ticket)
    if retained.latch is None:
        return False
    require_latch_identity(retained.latch, service.latch)
    # Exact already-consumed replay remains the native owner's responsibility.
    # Its sources can legitimately be pruned after all obligations settle.
    if retained.consumption is not None:
        return True
    latch = retained.latch
    if latch.wait_operation is None:
        raise CollaborationUnavailable("Host continuation lacks its collaboration wait source.")
    participants = app._participant_coordinator
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
    _, grant = participants._authorize(context, "request_readback")
    operation = latch.wait_operation
    if (
        operation.application_scope != initialized.owner.application_scope
        or operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationUnavailable("Host continuation belongs to another wait owner.")
    creation_deliveries = {}
    retained_bytes = 0
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        await store._anchor(tx, initialized, redactor)
        raw = await tx.get("operations", wait_operation_key(operation))
        if raw is None:
            raise CollaborationUnavailable("Host continuation wait source is unavailable.")
        snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
        require_latch_identity(_elected_latch(snapshot, redactor), latch)
        if snapshot.delivery not in {"pending", "accepted"}:
            raise CollaborationUnavailable("Host continuation wait delivery conflicts.")
        for target in snapshot.registration.wait.targets:
            if request_object_ref(target.intent.selection.reference) not in latch.selected_manifest:
                continue
            participants._require_refs(
                grant,
                (target.intent.request.sender, target.intent.selection.recipient.reference),
            )
            request = await retained_request(
                store, tx, initialized, target.intent.request, target.initiator, redactor
            )
            if request is None:
                raise CollaborationUnavailable("Host continuation selected source is unavailable.")
            require_exact_contract(target, request.receipt.expected, redactor=redactor)
            output = await read_request_output(tx, target, redactor=redactor)
            if output is None:
                if request.producer_operation is not None:
                    raise CollaborationUnavailable(
                        "Host continuation producer index is unavailable."
                    )
                # Requests without native producers have their existing wait
                # semantics; no missing producer marker is inferred as cleanup.
                continue
            if request.producer_operation != output.command.operation:
                raise CollaborationUnavailable("Host continuation producer attachment conflicts.")
            if request.state != "answered":
                continue
            for destination in output.command.destinations:
                key = destination.attempt.append_key
                if key.creation_target is None and (
                    key.target_session_id,
                    key.target_session_instance_id,
                ) != (
                    expected.session.session_id,
                    expected.session.session_instance_id,
                ):
                    continue
                if destination.recipient != expected.session.participant:
                    if key.creation_target is not None:
                        continue
                    raise CollaborationUnavailable(
                        "Host continuation delivery recipient conflicts."
                    )
                delivered = await read_delivery(tx, output.command, destination, redactor=redactor)
                excluded = await read_destination_exclusion(
                    tx, output.command, destination, redactor=redactor
                )
                if excluded is not None and delivered is not None:
                    raise CollaborationUnavailable("Host continuation delivery decisions conflict.")
                settled = excluded is not None or (
                    delivered is not None
                    and delivered.receipt is not None
                    and delivered.receipt.status in {"appended", "excluded"}
                )
                if key.creation_target is None:
                    if not settled:
                        return False
                    continue
                creation = key.creation_target
                prior = creation_deliveries.get(creation.key)
                if prior is not None:
                    require_exact_contract(prior[0], creation, redactor=redactor)
                    settled = settled and prior[1]
                else:
                    retained_bytes += len(contract_bytes(creation, redactor=redactor)) + 64
                    if retained_bytes > 65536:
                        raise CollaborationUnavailable(
                            "Host continuation creation readback exceeds its observation bound."
                        )
                creation_deliveries[creation.key] = (creation, settled)
    # Creation decisions belong to SessionStore. Never enter a foreign owner
    # while holding the collaboration transaction. These exact creation and
    # terminal delivery decisions are immutable; pending evidence stays pending.
    for creation, settled in creation_deliveries.values():
        found = await app.session_store.read_session_creation_decision(creation)
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Host continuation creation target is unavailable.")
        decision = prepare_contract(SessionCreationDecision, found.receipt, redactor=redactor)
        require_exact_contract(decision.target, creation, redactor=redactor)
        if decision.state == "pending":
            return False
        if decision.state == "excluded" or (
            decision.session_id,
            decision.session_instance_id,
        ) != (expected.session.session_id, expected.session.session_instance_id):
            continue
        if not settled:
            return False
    return True

"""Exact terminal decision and finite receiving frontier for producer cleanup."""

from cayu.collaboration._preparation import require_exact_contract
from cayu.collaboration._producer_delivery_store import delivery_operation
from cayu.collaboration._producer_export_store import export_operation
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import ProducerOutcomeCommand


def producer_terminal(request, command, *, redactor):
    """Read one positive terminal decision; closure never becomes an answer."""
    if request is None or request.producer_operation != command.operation:
        raise CollaborationUnavailable("Producer terminal responsibility is unavailable.")
    require_exact_contract(command.admission.expected, request.receipt.expected, redactor=redactor)
    if request.state in ("answered", "failed") and request.outcome is not None:
        if not isinstance(request.outcome.command, ProducerOutcomeCommand):
            raise CollaborationUnavailable("Producer outcome authority is unavailable.")
        return "outcome", request.outcome.command.operation
    if request.state in ("cancelled", "expired") and request.terminal is not None:
        require_exact_contract(
            command.admission.expected,
            request.terminal.expected.intent.expected,
            redactor=redactor,
        )
        return "closure", request.terminal.expected.operation
    raise CollaborationUnavailable("Producer terminal disposition remains unresolved.")


def cleanup_destinations(record, completion, request, *, redactor):
    """Determine obligations from the closed source's transactional dispatch index.

    Every export intent is recorded before foreign dispatch; closed request
    arbitration forbids new intents. An empty frontier together with that
    positive closure proves no delivery was dispatched by this owner. It does
    not prove exclusion of a prepared attempt or an unrelated external effect.
    """
    kind, _ = producer_terminal(request, record.command, redactor=redactor)
    if kind == "outcome":
        if completion.output.disposition == "answer":
            return record.command.destinations
        if record.exports or record.deliveries:
            raise CollaborationUnavailable("Failed producer has conflicting delivery evidence.")
        return ()
    destinations = tuple(
        item
        for item in record.command.destinations
        if export_operation(item, redactor) in record.exports
    )
    if set(record.exports) != {export_operation(item, redactor) for item in destinations}:
        raise CollaborationUnavailable("Closed producer export frontier conflicts.")
    if not set(record.deliveries).issubset(
        {delivery_operation(item, redactor) for item in destinations}
    ):
        raise CollaborationUnavailable("Closed producer receiving frontier conflicts.")
    return destinations

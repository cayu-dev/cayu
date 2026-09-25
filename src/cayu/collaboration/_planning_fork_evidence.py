"""Corroborate foreign terminal readback with existing durable permit settlements."""

from hashlib import sha256

from cayu.collaboration._permit_store import require_terminal_permit_receipt
from cayu.collaboration.participants import CollaborationUnavailable


async def require_view_source_settlement(tx, receipt, *, redactor):
    target = receipt.command.preparation.view
    excluded = receipt.state == "excluded"
    identity = ("view-exclusion:" if excluded else "view-retention:") + sha256(
        target.model_dump_json().encode()
    ).hexdigest()
    for permit in target.permits:
        retained = await require_terminal_permit_receipt(tx, permit, redactor=redactor)
        receiving = retained.receiving_receipt
        if (
            receiving.expected != permit
            or receiving.receiving_owner != target.request.source_owner
            or receiving.receipt_id != identity
            or receiving.proves_exclusion != excluded
            or receiving.outcome != ("excluded" if excluded else "quiescent")
        ):
            raise CollaborationUnavailable("View stage contradicts its source settlement.")

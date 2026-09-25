"""Private native readback token and corroborating source-permit settlement."""

from dataclasses import dataclass

from cayu.collaboration._permit_store import require_terminal_permit_receipt
from cayu.collaboration._planning_creation_types import RequestCreationStageReceipt
from cayu.collaboration.participants import CollaborationUnavailable


@dataclass(frozen=True, slots=True)
class _CreationReadback:
    """Mint only after the native owner authenticates the exact creation decision."""

    receipt: RequestCreationStageReceipt


async def require_creation_source_settlement(tx, receipt, *, redactor):
    from cayu.sessions._recipient_admission import recipient_creation_settlement_id

    target = receipt.command.preparation.creation
    permit = target.permit
    retained = await require_terminal_permit_receipt(tx, permit, redactor=redactor)
    receiving = retained.receiving_receipt
    if (
        receiving.expected != permit
        or receiving.receiving_owner != target.receiving_owner
        or receiving.receipt_id != recipient_creation_settlement_id(target)
        or receiving.proves_exclusion != (receipt.decision.state == "excluded")
        or receiving.outcome
        != ("excluded" if receipt.decision.state == "excluded" else "quiescent")
    ):
        raise CollaborationUnavailable("Creation stage contradicts its source settlement.")

"""Explicit host driver for one already-created, inert FRESH recipient.

The host supplies authenticated contexts and an exact decision from its request
owner. This is not a planner, worker, recipient launcher or permissive policy.
Run under an application configured with PreparedAdmissionRegistration, a held
MandateResolver and its production budget-binding receiver. See
docs/collaboration-requests.md for registration and input-revision semantics.
"""

from cayu import (
    CayuApp,
    CollaborationAccessContext,
    MandateAccessContext,
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
)
from cayu.collaboration._contracts import ExactMatch
from cayu.sessions import RecipientSessionCreationRequest


async def admit_prepared_recipient(
    app: CayuApp,
    creation: RecipientSessionCreationRequest,
    decision: RequestAdmissionCommand,
    *,
    creation_context: CollaborationAccessContext,
    receiving_context: MandateAccessContext,
) -> RequestAdmissionReceipt:
    """Authenticate preparation, admit once and check exact receiving-owner readback.

    Preserve `decision` across interruption. Reconcile its complete expectation
    before preparing a replacement. Repeating preparation below is appropriate
    for a new decision, not required for historical acknowledgement recovery.
    """
    proposed = await app.prepare_recipient_admission(creation, context=creation_context)
    if decision.prepared != proposed:
        raise ValueError("Decision does not name this exact prepared recipient.")
    receipt = await app.admit_collaboration_request(decision, context=receiving_context)
    found = await app.collaboration_admission_reader().lookup(decision, context=receiving_context)
    if not isinstance(found, ExactMatch) or found.receipt != receipt:
        raise RuntimeError("Admission acknowledgement requires exact reconciliation.")
    return receipt

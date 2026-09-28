"""Positive native evidence for a failed maintenance turn, never blind retry."""

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration._producer_inspection import (
    ProducerOutputInspection,
    inspect_producer_output,
)


async def reconcile_maintenance(app, intent, *, context):
    """Read the original registered effect; do not dispatch a replacement.

    A retained native outcome permits releasing this local turn only. It does
    not settle the producer, authorize new disclosure, or clear another owner's
    pending work. Missing/conflicting/unavailable evidence stays fenced.
    """
    found = await inspect_producer_output(
        app, intent.recovery, context=context, wait_for_settlement=True
    )
    if not isinstance(found, ExactMatch):
        return None
    snapshot = prepare_contract(
        ProducerOutputInspection, found.receipt, redactor=app._secret_redactor
    )
    require_exact_contract(snapshot.recovery, intent.recovery, redactor=app._secret_redactor)
    destination = next(
        (item for item in snapshot.destinations if item.destination == intent.destination), None
    )
    match intent.action:
        case "retain_completion":
            settled = snapshot.completion is not None
        case "export":
            # Rejection is a durable output-contract failure, not successful
            # export. Subsequent selection must elect failure through its owner.
            settled = destination is not None and destination.export in ("published", "rejected")
        case "publish_answer" | "publish_failure":
            # Terminal elections are immutable. A competing terminal outcome
            # excludes this local effect without converting its original error
            # into a successful answer/failure or settling producer cleanup.
            settled = snapshot.request_state in {
                "answered",
                "failed",
                "cancelled",
                "expired",
                "declined",
            }
        case "deliver" | "exclude":
            # Append and exclusion are mutually exclusive native decisions.
            # Either positively settles this turn, even when the other won.
            settled = destination is not None and destination.delivery in {"appended", "excluded"}
        case "retire":
            settled = destination is not None and destination.export_cleanup in (
                "retired",
                "excluded",
            )
        case "release_export":
            settled = destination is not None and destination.export_cleanup == "released"
        case "settle" | "service_disposition":
            settled = snapshot.cleanup_ack is not None
        case _:
            raise ValueError("Unknown producer maintenance effect.")
    return snapshot if settled else None

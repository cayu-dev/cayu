"""Observe one public service launch through its exact maintenance entrance."""

import asyncio

from cayu import CayuApp
from cayu.collaboration._clarification_service_api import (
    ClarificationServiceReceipt,
    ClarificationServiceRequest,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.participants import CollaborationUnavailable


async def await_service_return(
    app: CayuApp,
    request: ClarificationServiceRequest,
    *,
    context: SessionExportAccessContext,
    delivery_context: SessionExportAccessContext | None,
    recovery_context: CollaborationAccessContext,
    timeout: float,
    nested_errors: list[BaseException],
    retain_settlement_debt: bool = False,
) -> ClarificationServiceReceipt:
    launched = False
    last_error: CollaborationUnavailable | None = None
    last_state = "unobserved"
    try:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    if not launched:
                        launched = True
                        receipt = await app.service_clarification(
                            request, context=context, delivery_context=delivery_context
                        )
                    elif retain_settlement_debt:
                        # These tests deliberately keep settlement ACK broken
                        # until reconstruction. Reconciliation cannot finish
                        # while that fault is installed. Observe native return
                        # without discharging the debt, then authenticate the
                        # exact terminal public replay (never a new dispatch).
                        page = await app.inspect_clarification_services(
                            request.ticket, context=recovery_context
                        )
                        matching = [
                            item
                            for item in page.items
                            if item.recovery.operation == request.operation
                        ]
                        if not matching or matching[0].state != "returned":
                            if matching:
                                last_state = matching[0].state
                            await asyncio.sleep(0.25)
                            continue
                        receipt = await app.service_clarification(
                            request, context=context, delivery_context=delivery_context
                        )
                    else:
                        # Observation loss must not repeatedly reacquire source
                        # export authority or restart the service entrance. The
                        # original owner may still be preparing or executing.
                        receipt = await app.reconcile_clarification_service(
                            request, context=recovery_context
                        )
                except CollaborationUnavailable as error:
                    if nested_errors:
                        raise nested_errors[0] from None
                    last_error = error
                else:
                    last_error = None
                    last_state = receipt.state
                    if receipt.state == "returned":
                        return receipt
                    assert receipt.state != "excluded", "Service was excluded, not completed"
                await asyncio.sleep(0.25)
    except TimeoutError as error:
        error.add_note(f"Exact service did not return; last observed state: {last_state}")
        # Retain the actual readback failure instead of hiding it behind only
        # the observer's final CancelledError/TimeoutError.
        raise error from last_error

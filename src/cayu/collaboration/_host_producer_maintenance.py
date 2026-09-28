"""One native producer-maintenance effect per explicit host scheduling turn.

The host selects work; existing receiving owners authenticate and arbitrate it.
No local success, exception, timeout, or stopped observer is settlement evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, ExactMatch, OperationRef
from cayu.collaboration._host_ownership import HostReconciledResult
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_delivery import deliver_producer_output
from cayu.collaboration._producer_delivery_recovery import (
    ProducerDeliveryRecovery,
    reconcile_producer_delivery,
)
from cayu.collaboration._producer_disposition import service_closed_producer
from cayu.collaboration._producer_export import export_producer_output
from cayu.collaboration._producer_export_cleanup import retire_unneeded_producer_export
from cayu.collaboration._producer_outcome import publish_producer_outcome
from cayu.collaboration._producer_public_control import (
    observe_public_producer_completion,
    settle_public_producer_output,
)
from cayu.collaboration._producer_readback import lookup_producer_registration
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.participants import CollaborationUnavailable

Action = Literal[
    "retain_completion",
    "export",
    "publish_answer",
    "publish_failure",
    "deliver",
    "exclude",
    "retire",
    "settle",
    "service_disposition",
    "release_export",
]
_DESTINATION_ACTIONS = frozenset(
    {
        "export",
        "publish_answer",
        "deliver",
        "exclude",
        "retire",
        "release_export",
    }
)
_DISCLOSURE_ACTIONS = frozenset(
    {
        "export",
        "publish_answer",
        "publish_failure",
        "deliver",
        "release_export",
    }
)


class HostProducerMaintenance(ContractValue):
    recovery: ProducerOutputRecovery
    action: Action
    destination: OperationRef | None = None

    @model_validator(mode="after")
    def exact_destination(self):
        if (self.action in _DESTINATION_ACTIONS) != (self.destination is not None):
            raise ValueError("Producer maintenance requires its exact destination selection.")
        return self


@dataclass(frozen=True, slots=True)
class HostMaintenanceResult:
    dispatched: bool
    receipt: ContractValue | None = None


async def service_producer_maintenance(
    app,
    intent: HostProducerMaintenance,
    *,
    context: CollaborationAccessContext,
    disclosure_context: SessionExportAccessContext | None,
    observation_deadline: float,
    stop: asyncio.Event,
    dispatch_window=None,
) -> HostMaintenanceResult | HostReconciledResult:
    """Invoke one existing owner with original identity and current authority.

    This private adapter runs inside retained host ownership. The monotonic
    deadline limits *starting* this turn; it cannot authorize effects past the
    native owner's semantic deadline or prove dispatched work has stopped.
    Cleanup-only intents never manufacture a disclosure context from read access.
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
    if type(stop) is not asyncio.Event:
        raise TypeError("Host maintenance requires its owned stop signal.")

    def stopped():
        return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

    async def can_dispatch():
        if stop.is_set():
            return False
        if dispatch_window is None:
            return not stopped()
        return await dispatch_window()

    if stopped():
        return HostMaintenanceResult(False)
    try:
        found = await lookup_producer_registration(
            app, intent.recovery, context=context, wait_for_settlement=True
        )
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Host maintenance lacks exact producer registration.")
        command = prepare_contract(ProducerOutputRegistration, found.receipt, redactor=redactor)
        if intent.destination is not None and not any(
            item.operation == intent.destination for item in command.destinations
        ):
            raise CollaborationUnavailable("Host maintenance destination is not registered.")
    except Exception as error:
        # No receiving mutation has been called in this turn. Release only its
        # local effect reservation and report the original preflight failure;
        # do not strand maintenance capacity behind a failed read. This proves
        # neither producer settlement nor quiescence of other native owners.
        return HostReconciledResult(HostMaintenanceResult(False), error)
    # Lookup may block or overlap shutdown. Readback is not a renewed dispatch grant.
    if not await can_dispatch():
        return HostMaintenanceResult(False)

    async def dispatch():
        match intent.action:
            case "retain_completion":
                receipt = await observe_public_producer_completion(app, command, context=context)
                if receipt is None:
                    return HostMaintenanceResult(False)
            case "export":
                receipt = await export_producer_output(
                    app,
                    command,
                    intent.destination,
                    context=disclosure_context,
                    wait_for_settlement=True,
                )
            case "publish_answer" | "publish_failure":
                receipt = await publish_producer_outcome(
                    app,
                    command,
                    destination_operation=intent.destination,
                    context=disclosure_context,
                    wait_for_settlement=True,
                )
            case "deliver":
                receipt = await deliver_producer_output(
                    app,
                    command,
                    intent.destination,
                    context=disclosure_context,
                    wait_for_settlement=True,
                )
            case "exclude":
                assert intent.destination is not None
                receipt = await reconcile_producer_delivery(
                    app,
                    ProducerDeliveryRecovery(
                        registration=intent.recovery.registration,
                        registration_commitment=intent.recovery.registration_commitment,
                        destination=intent.destination,
                    ),
                    context=context,
                    exclude=True,
                    wait_for_settlement=True,
                )
            case "retire":
                receipt = await retire_unneeded_producer_export(
                    app, command, intent.destination, context=context, wait_for_settlement=True
                )
            case "settle":
                receipt = await settle_public_producer_output(
                    app, command, context=context, wait_for_settlement=True
                )
            case "service_disposition":
                receipt = await service_closed_producer(
                    app, command, context=context, wait_for_settlement=True
                )
            case "release_export":
                receipt = await _release_export(
                    app,
                    command,
                    intent.destination,
                    context=disclosure_context,
                    can_dispatch=can_dispatch,
                )
                if receipt is None:
                    return HostMaintenanceResult(False)
        # The receipt retains its owner's meaning. Dispatch is not necessarily a new
        # mutation (exact replay is normal), and certainly not blanket cleanup proof.
        return HostMaintenanceResult(True, receipt)

    try:
        return await dispatch()
    except Exception as primary:
        from cayu.collaboration._host_producer_reconciliation import reconcile_maintenance

        try:
            retained = await reconcile_maintenance(app, intent, context=context)
        except Exception as recovery:
            raise ExceptionGroup(
                "Producer maintenance and exact readback failed", [primary, recovery]
            ) from None
        if retained is None:
            raise
        return HostReconciledResult(HostMaintenanceResult(True, retained), primary)


async def _release_export(app, command, destination_operation, *, context, can_dispatch):
    """Reuse native acceptance/settlement, including an earlier owner's exact key."""
    from hashlib import sha256

    from cayu.collaboration._preparation import contract_bytes, require_exact_contract
    from cayu.collaboration._producer_delivery_store import read_delivery
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
    from cayu.collaboration.exports import SessionExportSettlementRequest

    configured = app._request_coordinator._registration
    receiver = None if configured is None else configured.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Host export cleanup requires its native receiver.")
    redactor = app._secret_redactor
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    destination = next(
        item for item in command.destinations if item.operation == destination_operation
    )
    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, command, redactor=redactor)
        delivery = await read_delivery(tx, command, destination, redactor=redactor)
    if (
        record is None
        or delivery is None
        or delivery.receipt is None
        or delivery.receipt.status != "appended"
    ):
        raise CollaborationUnavailable("Host export release requires exact receiving acceptance.")
    prior = await receiver._read_producer_export_settlement(record, delivery, allow_pending=True)
    if prior is not None:
        return prior
    if not await can_dispatch():
        return None
    request = delivery.source_receipt.expected.intent.request
    settlement = SessionExportSettlementRequest(
        request=request,
        mode="release",
        operation=request.ref.operation.model_copy(
            update={
                "caller_key": "host-producer-release:"
                + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest(),
            }
        ),
    )
    return await app._session_export_coordinator.settle(
        settlement, context=context, wait_for_settlement=True
    )

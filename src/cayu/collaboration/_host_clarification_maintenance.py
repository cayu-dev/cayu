"""Source-selected clarification maintenance, never disclosure or dispatch."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal

from cayu.collaboration._clarification_deliveries import ClarificationDeliveryReceipt
from cayu.collaboration._clarification_question_recovery import _QuestionExpirySuperseded
from cayu.collaboration._clarification_recovery_types import (
    ClarificationDeliveryRecovery,
    ClarificationDueQuestion,
    ClarificationExpiryReceipt,
    ClarificationExpiryRequest,
    ClarificationPendingDelivery,
    ClarificationPendingService,
    ClarificationServiceRecovery,
)
from cayu.collaboration._clarification_service_api import ClarificationServiceReceipt
from cayu.collaboration._host_discovery import observe_host_source
from cayu.collaboration._host_ownership import HostCapacityExceeded, HostOperationIdentity
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.participants import CollaborationUnavailable


@dataclass(frozen=True, slots=True)
class _ClarificationMaintenanceSource:
    source: Literal["questions", "deliveries", "services"]
    context: CollaborationAccessContext


def snapshot_maintenance_source(app, source):
    if type(source) is not _ClarificationMaintenanceSource or source.source not in (
        "questions",
        "deliveries",
        "services",
    ):
        raise TypeError("Host clarification maintenance source is unsupported.")
    context = prepare_contract(
        CollaborationAccessContext, source.context, redactor=app._secret_redactor
    )
    return _ClarificationMaintenanceSource(source.source, context), len(
        contract_bytes(context, redactor=app._secret_redactor)
    )


class HostClarificationMaintenanceSweep:
    """Local bounded scan cursors; all pending responsibility stays source-owned."""

    def __init__(self, app, sources, reads):
        self._app = app
        self._reads = reads
        self._sources = sources
        self._cursors = [None] * len(sources)
        self._next = 0
        self.errors = {}

    async def step(self, ownership, deadline, stop, *, batch_size, page_bytes):
        loop = asyncio.get_running_loop()
        for _ in range(min(batch_size, len(self._sources))):
            if stop.is_set() or loop.time() >= deadline:
                break
            index = self._next
            self._next = (index + 1) % len(self._sources)
            source = self._sources[index]
            try:
                observed = await observe_host_source(
                    self._reads,
                    "clarification-source:" + str(index),
                    self._app,
                    source.source,
                    context=source.context,
                    cursor=self._cursors[index],
                    limit=batch_size,
                    max_bytes=page_bytes,
                )
                if observed is None:
                    continue
                page = observed.value
                self.errors.pop(index, None)
                for item in page.items:
                    if stop.is_set() or loop.time() >= deadline:
                        break
                    if source.source == "questions" and type(item) is ClarificationDueQuestion:
                        token = item.recovery
                        digest = sha256(
                            contract_bytes(token, redactor=self._app._secret_redactor)
                        ).hexdigest()
                        expected = ClarificationExpiryRequest(
                            operation=token.operation.model_copy(
                                update={"caller_key": "host-expiry:" + digest}
                            ),
                            recovery=token,
                        )
                    elif (
                        source.source == "deliveries" and type(item) is ClarificationPendingDelivery
                    ) or (
                        source.source == "services" and type(item) is ClarificationPendingService
                    ):
                        expected = item.recovery
                    else:
                        raise TypeError(
                            "Host maintenance discovery returned another source family."
                        )
                    start_clarification_maintenance(
                        self._app,
                        ownership,
                        expected,
                        context=source.context,
                        observation_deadline=deadline,
                    )
                    # Advance only past a scheduled item, not past the entire
                    # fetched page. A pending first item must not monopolize a
                    # single slot when later items can already settle.
                    self._cursors[index] = ClarificationDueCursor(
                        deadline_at_ms=item.deadline_at_ms,
                        operation=item.recovery.operation,
                    )
                else:
                    self._cursors[index] = page.next_cursor
            except HostCapacityExceeded:
                self._next = index
                break
            except Exception as error:
                self.errors[index] = error


@dataclass(frozen=True, slots=True)
class HostClarificationMaintenanceResult:
    dispatched: bool
    receipt: (
        ClarificationExpiryReceipt
        | ClarificationDeliveryReceipt
        | ClarificationServiceReceipt
        | _QuestionExpirySuperseded
        | None
    ) = None


def start_clarification_maintenance(app, ownership, expected, *, context, observation_deadline):
    if type(expected) not in (
        ClarificationExpiryRequest,
        ClarificationDeliveryRecovery,
        ClarificationServiceRecovery,
    ):
        raise TypeError("Clarification maintenance requires an exact source-owned selector.")
    redactor = app._secret_redactor
    expected = prepare_contract(type(expected), expected, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host maintenance requires a finite observation deadline.")
    encoded = contract_bytes(expected, redactor=redactor)
    material = encoded + b"\0" + contract_bytes(context, redactor=redactor)
    identity = HostOperationIdentity(
        "clarification-maintenance:" + sha256(encoded).hexdigest(), sha256(material).hexdigest()
    )

    async def action(stop):
        if stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline:
            return HostClarificationMaintenanceResult(False)
        coordinator = app._clarification_coordinator
        if type(expected) is ClarificationExpiryRequest:
            raw = await coordinator.expire_question(
                expected, context=context, wait_for_settlement=True
            )
            receipt = prepare_contract(ClarificationExpiryReceipt, raw, redactor=redactor)
            if receipt.expected != expected:
                raise CollaborationUnavailable("Question expiry returned another exact decision.")
        elif type(expected) is ClarificationDeliveryRecovery:
            raw = await coordinator.reconcile_delivery(
                expected, context=context, wait_for_settlement=True
            )
            receipt = prepare_contract(ClarificationDeliveryReceipt, raw, redactor=redactor)
            if receipt.operation != expected.operation:
                raise CollaborationUnavailable("Delivery maintenance returned another operation.")
        else:
            raw = await coordinator.reconcile_service(
                app, expected, context=context, wait_for_settlement=True
            )
            receipt = prepare_contract(ClarificationServiceReceipt, raw, redactor=redactor)
            # The selector names the original waiting ticket's incarnation,
            # whereas the receipt names the service target (possibly a side
            # session). The native receiving owner authenticates both through
            # the selector's complete selection and dispatch commitments before
            # reconciling. They must not be conflated in this projection.
            if receipt.operation != expected.operation:
                raise CollaborationUnavailable("Service maintenance returned another operation.")
        # A still-pending native record remains discoverable. Finishing this
        # reconciliation turn does not claim exclusion or stop the service.
        return HostClarificationMaintenanceResult(True, receipt)

    async def reconcile():
        # Read existing decisions only. Current readback authority is independent
        # of permission to expire, disclose, or dispatch again.
        receipt = await app._clarification_coordinator._inspect_maintenance_owned(
            app, expected, context=context
        )
        return None if receipt is None else HostClarificationMaintenanceResult(False, receipt)

    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_clarification_maintenance(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostClarificationMaintenanceResult or (
        result.dispatched and result.receipt is None
    ):
        raise RuntimeError("Clarification maintenance returned invalid owner evidence.")
    ownership.release_settled(outcome.identity)
    return True

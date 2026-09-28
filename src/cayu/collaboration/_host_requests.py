"""Owner-time request expiry; no host clock or historical execution grant."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._host_discovery import observe_host_source
from cayu.collaboration._host_ownership import (
    HostCapacityExceeded,
    HostOperationIdentity,
    HostReconciledResult,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl, RequestControlReceipt, RequestSnapshot


@dataclass(frozen=True, slots=True)
class _RequestMaintenanceSource:
    context: MandateAccessContext


def snapshot_request_source(app, source):
    if type(source) is not _RequestMaintenanceSource:
        raise TypeError("Host request maintenance requires a typed source.")
    context = prepare_contract(MandateAccessContext, source.context, redactor=app._secret_redactor)
    return _RequestMaintenanceSource(context), len(
        contract_bytes(context, redactor=app._secret_redactor)
    )


class HostRequestMaintenanceSweep:
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
                    "request-source:" + str(index),
                    self._app,
                    "requests",
                    context=source.context,
                    cursor=self._cursors[index],
                    limit=batch_size,
                    max_bytes=page_bytes,
                )
                if observed is None:
                    continue
                page = observed.value
                if type(page.observed_at_ms) is not int:
                    raise CollaborationUnavailable("Request discovery has no owner-time evidence.")
                self._cursors[index] = page.next_cursor
                self.errors.pop(index, None)
                for item in page.items:
                    if stop.is_set() or loop.time() >= deadline:
                        break
                    if type(item) is not RequestSnapshot:
                        raise TypeError("Request discovery returned another source family.")
                    if (
                        item.state != "open"
                        or item.receipt.expected.intent.selection.expires_at_ms
                        > page.observed_at_ms
                    ):
                        continue
                    start_request_expiry(
                        self._app,
                        ownership,
                        item,
                        context=source.context,
                        observation_deadline=deadline,
                    )
            except HostCapacityExceeded:
                break
            except Exception as error:
                self.errors[index] = error


@dataclass(frozen=True, slots=True)
class HostRequestMaintenanceResult:
    dispatched: bool
    receipt: RequestControlReceipt | None = None
    superseded_revision: int | None = None


def start_request_expiry(app, ownership, snapshot, *, context, observation_deadline):
    redactor = app._secret_redactor
    snapshot = prepare_contract(RequestSnapshot, snapshot, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
        raise ValueError("Host request expiry requires a finite observation deadline.")
    expected = snapshot.receipt.expected
    # One deterministic control per complete observed revision. Receiving-owner
    # CAS and current authorization still arbitrate expiry against other work.
    identity_material = b"\0".join(
        (contract_bytes(expected, redactor=redactor), str(snapshot.revision).encode("ascii"))
    )
    control = RequestControl(
        operation=expected.operation.model_copy(
            update={"caller_key": "host-request-expiry:" + sha256(identity_material).hexdigest()}
        ),
        expected=expected,
        expected_revision=snapshot.revision,
        kind="expire",
    )
    material = b"\0".join(
        (contract_bytes(control, redactor=redactor), contract_bytes(context, redactor=redactor))
    )
    identity = HostOperationIdentity(
        "request-expiry:" + sha256(identity_material).hexdigest(), sha256(material).hexdigest()
    )

    async def action(stop):
        if stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline:
            return HostRequestMaintenanceResult(False)
        # The native owner rechecks its own clock and exact revision under the
        # existing request-control transaction. Readback never authorizes expiry.
        try:
            raw_receipt = await app._request_coordinator.control(
                control, context=context, wait_for_settlement=True
            )
        except Exception as error:
            # A completed native await is not proof of commit or exclusion.
            # Exact success or a later authenticated source revision can settle
            # this local turn. Neither authorizes another mutation or settles
            # any underlying producer responsibility.
            try:
                recovered = await reconcile()
            except Exception as recovery_error:
                raise ExceptionGroup(
                    "Request expiry and exact readback failed", [error, recovery_error]
                ) from None
            if recovered is None:
                raise
            return HostReconciledResult(recovered, error)
        receipt = prepare_contract(RequestControlReceipt, raw_receipt, redactor=redactor)
        if receipt.expected.intent != control or receipt.state != "expired":
            raise CollaborationUnavailable("Request expiry returned another exact decision.")
        result = HostRequestMaintenanceResult(True, receipt)
        return result

    async def reconcile():
        found = await app._request_coordinator.lookup_control_intent(control, context=context)
        if isinstance(found, ExactNotFound):
            current = await app._request_coordinator.inspect(
                expected, context=context, wait_for_settlement=True
            )
            if current is not None:
                current = prepare_contract(RequestSnapshot, current, redactor=redactor)
                require_exact_contract(current.receipt.expected, expected, redactor=redactor)
                # Request control CAS checks this exact revision under the
                # source transaction. A monotonic later revision positively
                # excludes the stale command, even if the request stays open.
                if current.revision > control.expected_revision:
                    return HostRequestMaintenanceResult(False, superseded_revision=current.revision)
        if not isinstance(found, ExactMatch):
            return None
        receipt = prepare_contract(RequestControlReceipt, found.receipt, redactor=redactor)
        if receipt.expected.intent != control or receipt.state != "expired":
            raise CollaborationUnavailable("Request expiry readback changed its exact decision.")
        return HostRequestMaintenanceResult(False, receipt)

    ownership.start(
        identity,
        role="maintenance",
        reserved_bytes=len(material) + 65536,
        action=action,
        reconcile=reconcile,
    )
    return identity


def acknowledge_request_maintenance(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostRequestMaintenanceResult or (
        result.dispatched and result.receipt is None
    ):
        raise RuntimeError("Request maintenance returned invalid owner evidence.")
    ownership.release_settled(outcome.identity)
    return True

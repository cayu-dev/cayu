"""Native finite-wait observation and latch delivery without model polling."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._host_ownership import (
    HostOperationIdentity,
    HostOwnership,
    HostReconciledResult,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._wait_coordinator import _WaitObservationNotStarted
from cayu.collaboration._wait_discovery import WaitRecovery, resolve_wait
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.waits import WaitSnapshot
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner


@dataclass(frozen=True, slots=True)
class HostWaitResult:
    dispatched: bool
    snapshot: WaitSnapshot | None = None


class HostWaitOwner:
    """One retained receiving owner per host; never a second wait or latch store."""

    def __init__(self, app):
        self._app = app
        self._continuations = {}
        self._finished = set()

    def acknowledge(self, ownership, outcome):
        acknowledged = acknowledge_wait(ownership, outcome)
        result = outcome.value
        if (
            acknowledged
            and result.snapshot is not None
            and result.snapshot.delivery in ("accepted", "excluded")
        ):
            # At most the 32 immutable registered wait rules can enter here.
            # This local hint skips already-settled maintenance; it grants no
            # execution, disclosure, retirement or recovery authority.
            self._finished.add(outcome.identity)
        return acknowledged

    def _receiver(self, wait):
        app = self._app
        _, initialized = app._participant_coordinator._ready()
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Host wait lacks a native delivery ticket.")
        owner = wait.delivery_ticket.owner
        key = (owner, initialized.owner)
        if key not in self._continuations:
            if len(self._continuations) >= 32:
                raise CollaborationUnavailable("Host wait receiving owners exceed their bound.")
            self._continuations[key] = SessionContinuationOwner(
                store=app.session_store,
                owner=owner,
                receiver=app.collaboration_wait_latch_receiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=app._secret_redactor,
            )
        return self._continuations[key]

    def start(self, ownership: HostOwnership, expected, *, context, observation_deadline):
        app = self._app
        redactor = app._secret_redactor
        expected = prepare_contract(WaitRecovery, expected, redactor=redactor)
        context = prepare_contract(MandateAccessContext, context, redactor=redactor)
        if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
            raise ValueError("Host wait observation requires a finite deadline.")
        encoded = contract_bytes(expected, redactor=redactor)
        material = encoded + b"\0" + contract_bytes(context, redactor=redactor)
        identity = HostOperationIdentity(
            key="wait-observation:" + sha256(encoded).hexdigest(),
            commitment=sha256(material).hexdigest(),
        )
        if identity in self._finished:
            return identity
        delivery_started = False

        async def action(stop):
            nonlocal delivery_started

            def stopped():
                return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

            if stopped():
                return HostWaitResult(False)
            try:
                found = await resolve_wait(
                    app._wait_coordinator, expected, context=context, wait_for_settlement=True
                )
                if not isinstance(found, ExactMatch):
                    raise CollaborationUnavailable("Host wait lacks exact source registration.")
                wait = found.receipt
            except Exception as error:
                # No receiving mutation was entered; this releases only the local turn.
                return HostReconciledResult(HostWaitResult(False), error)
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostWaitResult(False)
            try:
                snapshot = await app._wait_coordinator._observe_owned(wait, context=context)
            except _WaitObservationNotStarted as rejected:
                # Native proof concerns this local turn, not the durable wait.
                # Preserve the original error and retry only in a later pass.
                return HostReconciledResult(HostWaitResult(False), rejected.error)
            snapshot = prepare_contract(WaitSnapshot, snapshot, redactor=redactor)
            if snapshot.registration.registration_digest != expected.registration_digest:
                raise CollaborationUnavailable("Host wait observation belongs to another intent.")
            if (
                snapshot.state == "elected"
                and snapshot.delivery == "pending"
                and await ownership.wait_for_dispatch_window(
                    identity, initial_deadline=observation_deadline
                )
            ):
                delivery_started = True
                snapshot = await app._wait_coordinator._deliver_owned(
                    wait, context=context, continuation_owner=self._receiver(wait)
                )
                snapshot = prepare_contract(WaitSnapshot, snapshot, redactor=redactor)
                if snapshot.registration.registration_digest != expected.registration_digest:
                    raise CollaborationUnavailable("Host wait delivery changed its exact intent.")
            return HostWaitResult(True, snapshot)

        async def reconcile():
            found = await resolve_wait(
                app._wait_coordinator, expected, context=context, wait_for_settlement=True
            )
            if not isinstance(found, ExactMatch):
                return None
            snapshot = await app._wait_coordinator.inspect(
                found.receipt, context=context, wait_for_settlement=True
            )
            if snapshot is None:
                return None
            snapshot = prepare_contract(WaitSnapshot, snapshot, redactor=redactor)
            if snapshot.registration.registration_digest != expected.registration_digest:
                raise CollaborationUnavailable("Host wait readback changed its exact intent.")
            if delivery_started and snapshot.delivery == "pending":
                snapshot = await app._wait_coordinator._reconcile_delivery_owned(
                    found.receipt, context=context, continuation_owner=self._receiver(found.receipt)
                )
                if snapshot is None:
                    return None
                snapshot = prepare_contract(WaitSnapshot, snapshot, redactor=redactor)
                if snapshot.registration.registration_digest != expected.registration_digest:
                    raise CollaborationUnavailable("Host wait recovery changed its exact intent.")
            # A registration/election alone does not prove that delivery settled.
            if snapshot.source_pins or snapshot.state == "pending":
                return None
            if delivery_started and snapshot.delivery not in ("accepted", "excluded"):
                return None
            return HostWaitResult(False, snapshot)

        ownership.start(
            identity,
            role="maintenance",
            reserved_bytes=len(material) + 65536,
            action=action,
            reconcile=reconcile,
        )
        return identity


def acknowledge_wait(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if type(result) is not HostWaitResult or (result.dispatched and result.snapshot is None):
        raise RuntimeError("Wait owner returned invalid host handoff evidence.")
    if result.snapshot is not None and type(result.snapshot) is not WaitSnapshot:
        raise RuntimeError("Wait owner returned an unqualified source snapshot.")
    # The local observation turn can finish while the native wait stays parked.
    # No execution slot, cancellation decision or continuation is inferred.
    ownership.release_settled(outcome.identity)
    return True

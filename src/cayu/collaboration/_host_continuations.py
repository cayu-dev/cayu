"""Retained native continuation servicing; a latch is not execution authority."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._host_continuation_readiness import continuation_delivery_ready
from cayu.collaboration._host_ownership import HostOperationIdentity, HostReconciledResult
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._host_continuation_discovery import (
    ContinuationRecovery,
    recover_session_continuation,
)
from cayu.runtime._invocation_lifecycle import (
    _require_released_invocation_command_receipt,
    reconcile_invocation_admission_from_state,
)
from cayu.runtime._session_continuation import (
    ContinuationRecord,
    ContinuationService,
    ContinuationUnavailable,
    require_latch_identity,
    require_ticket_identity,
)
from cayu.runtime._session_continuation_owner import (
    LATCH_FAMILY,
    SessionContinuationOwner,
    _ContinuationServiceNotStarted,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.sessions.base import ResumeRequest, copy_resume_request


@dataclass(frozen=True, slots=True)
class _ContinuationRule:
    expected: ContinuationRecovery
    request: ResumeRequest
    service: ContinuationService
    context: CollaborationAccessContext
    recovery_inactive_for_seconds: int | None = None


def snapshot_continuation_rule(app, rule):
    if type(rule) is not _ContinuationRule:
        raise TypeError("Host continuation selection must be typed.")
    redactor = app._secret_redactor
    inactivity = rule.recovery_inactive_for_seconds
    if inactivity is not None and (type(inactivity) is not int or not 1 <= inactivity <= 86400):
        raise ValueError("Host continuation recovery requires bounded explicit inactivity.")
    expected = prepare_contract(ContinuationRecovery, rule.expected, redactor=redactor)
    service = prepare_contract(ContinuationService, rule.service, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, rule.context, redactor=redactor)
    request = copy_resume_request(rule.request)
    if (
        request.session_id != expected.session.session_id
        or service.ticket.session_id != expected.session.session_id
        or service.ticket.session_instance_id != expected.session.session_instance_id
        or service.ticket.registration_key != expected.registration_key
    ):
        raise ValueError("Host continuation selection mixes native identities.")
    app._session_engine.work_attempt_source_request_sha256(request, kind="continuation")
    size = len(
        canonical_durable_json_bytes(
            request.model_dump(mode="json", warnings=False),
            "host continuation request",
            max_bytes=65536,
        )
    ) + sum(len(contract_bytes(value, redactor=redactor)) for value in (expected, service, context))
    return _ContinuationRule(expected, request, service, context, inactivity), size + 8


@dataclass(frozen=True, slots=True)
class HostContinuationResult:
    dispatched: bool
    release_commitment: str | None = None
    admission_commitment: str | None = None
    recovery_commitment: str | None = None
    exclusion_commitment: str | None = None


def _excluded_service_result(app, record):
    """Only an exact service-owner read may supply this terminal non-dispatch."""
    if record.consumption is None or record.consumption.receipt_stage != "excluded":
        return None
    record = prepare_contract(ContinuationRecord, record, redactor=app._secret_redactor)
    if record.ticket.state != "RETIRED" or record.retirement is None:
        raise ContinuationUnavailable("Host continuation exclusion lacks exact retirement.")
    return HostContinuationResult(
        False,
        exclusion_commitment=sha256(
            contract_bytes(record, redactor=app._secret_redactor)
        ).hexdigest(),
    )


class HostContinuationOwner:
    def __init__(self, app):
        self._app = app
        self._receivers = {}
        self._finished = set()

    def _receiver(self, retained):
        app = self._app
        _, initialized = app._participant_coordinator._ready()
        if retained.preparation.registration.child.destination != initialized.owner:
            raise ContinuationUnavailable("Host continuation belongs to another receiver.")
        key = (retained.ticket.owner, initialized.owner)
        if key not in self._receivers:
            if len(self._receivers) >= 32:
                raise ContinuationUnavailable(
                    "Host continuation receiving owners exceed their bound."
                )
            self._receivers[key] = SessionContinuationOwner(
                store=app.session_store,
                owner=retained.ticket.owner,
                receiver=app.collaboration_wait_latch_receiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=initialized.owner,
                    mutations=(),
                    readbacks=(LATCH_FAMILY,),
                ),
                redactor=app._secret_redactor,
            )
        return self._receivers[key]

    def acknowledge(self, ownership, outcome):
        acknowledged = acknowledge_continuation(ownership, outcome)
        result = outcome.value
        if acknowledged and (
            result.release_commitment is not None
            or result.admission_commitment is not None
            or result.recovery_commitment is not None
            or result.exclusion_commitment is not None
        ):
            # Positive release, exact receiving admission or exact exclusion
            # ends this selection. A deferred turn must remain eligible.
            # Immutable registration bounds this set; it grants no execution.
            self._finished.add(outcome.identity)
        return acknowledged

    def start(
        self,
        ownership,
        expected,
        request,
        service,
        *,
        context,
        observation_deadline,
        recovery_inactive_for_seconds=None,
    ):
        app = self._app
        redactor = app._secret_redactor
        expected = prepare_contract(ContinuationRecovery, expected, redactor=redactor)
        service = prepare_contract(ContinuationService, service, redactor=redactor)
        context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
        request = copy_resume_request(request)
        request_digest = app._session_engine.work_attempt_source_request_sha256(
            request, kind="continuation"
        )
        request_bytes = canonical_durable_json_bytes(
            request.model_dump(mode="json", warnings=False),
            "host continuation request",
            max_bytes=65536,
        )
        if type(observation_deadline) not in (int, float) or not isfinite(observation_deadline):
            raise ValueError("Host continuation requires a finite observation deadline.")
        if request.session_id != expected.session.session_id:
            raise ValueError("Host continuation request belongs to another session.")
        material = b"\0".join(
            (
                contract_bytes(expected, redactor=redactor),
                contract_bytes(service, redactor=redactor),
                contract_bytes(context, redactor=redactor),
                request_digest.encode("ascii"),
                str(recovery_inactive_for_seconds).encode("ascii"),
            )
        )
        identity = HostOperationIdentity(
            "continuation:" + sha256(contract_bytes(expected, redactor=redactor)).hexdigest(),
            sha256(material).hexdigest(),
        )
        if identity in self._finished:
            return identity

        async def action(stop):
            def stopped():
                return stop.is_set() or asyncio.get_running_loop().time() >= observation_deadline

            if stopped():
                return HostContinuationResult(False)
            try:
                if not await continuation_delivery_ready(app, expected, service, context=context):
                    return HostContinuationResult(False)
                retained = await recover_session_continuation(app, expected, context=context)
                require_ticket_identity(retained.ticket, service.ticket)
                if retained.latch is None:
                    raise ContinuationUnavailable("Host continuation has no authenticated latch.")
                require_latch_identity(retained.latch, service.latch)
                receiver = self._receiver(retained)
            except Exception as error:
                # No receiving mutation was entered; this releases only the local turn.
                return HostReconciledResult(HostContinuationResult(False), error)
            if not await ownership.wait_for_dispatch_window(
                identity, initial_deadline=observation_deadline
            ):
                return HostContinuationResult(False)
            # The existing service repeats participant, profile, budget and
            # human-action gates and owns the native stream to its completion.
            try:
                result = await receiver._service_owned(
                    app,
                    request,
                    service,
                    participant_context=context,
                )
            except _ContinuationServiceNotStarted as error:
                return HostReconciledResult(HostContinuationResult(False), error.failure)
            consumption = result.record.consumption
            excluded = _excluded_service_result(app, result.record)
            if excluded is not None:
                if result.dispatched:
                    raise ContinuationUnavailable("Excluded continuation reports dispatch.")
                return excluded
            if consumption is None or consumption.receipt_stage != "admitted":
                raise ContinuationUnavailable("Host continuation admission is not settled.")
            session = await app.session_store.load(expected.session.session_id)
            checkpoint = await runtime_checkpoint_session_store(app.session_store).load_checkpoint(
                expected.session.session_id
            )
            if not result.dispatched:
                if recovery_inactive_for_seconds is not None:
                    from cayu.collaboration._host_continuation_recovery import (
                        recover_interrupted_continuation,
                    )

                    released = await recover_interrupted_continuation(
                        app,
                        expected,
                        result.record,
                        context=context,
                        inactive_for_seconds=recovery_inactive_for_seconds,
                    )
                    return HostContinuationResult(False, recovery_commitment=released)
                # This service only read/reconciled receiving acceptance. The
                # native invocation may still run under another host's owned
                # stream; release only this observer's local slot, not its work.
                if (
                    session is None
                    or reconcile_invocation_admission_from_state(
                        session,
                        checkpoint,
                        session_id=expected.session.session_id,
                        session_instance_id=expected.session.session_instance_id,
                        expected_run_epoch=consumption.admission_expected_run_epoch,
                        command_sha256=consumption.admission_command_digest,
                        profile_sha256=consumption.profile_digest,
                    )
                    is None
                ):
                    raise ContinuationUnavailable(
                        "Host continuation lacks exact receiving admission evidence."
                    )
                return HostContinuationResult(
                    False, admission_commitment=consumption.admission_command_digest
                )
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                session is None
                or active is None
                or active.run_epoch != consumption.admission_expected_run_epoch + 1
                or active.profile.fingerprint != consumption.profile_digest
            ):
                raise ContinuationUnavailable("Host continuation release lacks original authority.")
            release = _require_released_invocation_command_receipt(
                session,
                checkpoint,
                session_id=expected.session.session_id,
                session_instance_id=expected.session.session_instance_id,
                active_profile=active,
            )
            return HostContinuationResult(True, release.record_sha256)

        async def reconcile():
            from cayu.collaboration._host_continuation_recovery import read_continuation_release

            recovered = await recover_session_continuation(app, expected, context=context)
            retained = await self._receiver(recovered)._inspect_service(
                app, request, service, expected=expected, participant_context=context
            )
            excluded = _excluded_service_result(app, retained)
            if excluded is not None:
                return excluded
            if retained.consumption is None or retained.consumption.receipt_stage != "admitted":
                return None
            release = await read_continuation_release(app, expected, retained, context=context)
            if release is None:
                return None
            return HostContinuationResult(False, recovery_commitment=release)

        ownership.start(
            identity,
            role="execution",
            reserved_bytes=len(material) + len(request_bytes) + 65536,
            action=action,
            reconcile=reconcile,
        )
        return identity


def acknowledge_continuation(ownership, outcome):
    if outcome.error is not None:
        return False
    result = outcome.value
    if (
        type(result) is not HostContinuationResult
        or result.dispatched != (result.release_commitment is not None)
        or (result.dispatched and result.admission_commitment is not None)
        or (
            result.exclusion_commitment is not None
            and (
                result.dispatched
                or result.release_commitment is not None
                or result.admission_commitment is not None
                or result.recovery_commitment is not None
            )
        )
        or (
            result.recovery_commitment is not None
            and (result.dispatched or result.admission_commitment is not None)
        )
    ):
        raise RuntimeError("Continuation owner returned invalid release evidence.")
    ownership.release_settled(outcome.identity)
    return True

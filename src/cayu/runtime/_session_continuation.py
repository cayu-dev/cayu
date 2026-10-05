"""Continuation publication guards, admission and compatibility imports."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

# Preserve established imports and historical pickle paths.
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES as CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_DIGEST_BYTES as CONTINUATION_MAX_DIGEST_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_EVENT_BYTES as CONTINUATION_MAX_EVENT_BYTES,
)
from cayu.sessions._session_continuation import CONTINUATION_MAX_EVENTS as CONTINUATION_MAX_EVENTS
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_LATCH_EVIDENCE_BYTES as CONTINUATION_MAX_LATCH_EVIDENCE_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_RECOVERY_WRITER_BYTES as CONTINUATION_MAX_RECOVERY_WRITER_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES as CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES as CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_SERVICE_REFERENCE_BYTES as CONTINUATION_MAX_SERVICE_REFERENCE_BYTES,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_SERVICES as CONTINUATION_MAX_SERVICES,
)
from cayu.sessions._session_continuation import CONTINUATION_MAX_TARGETS as CONTINUATION_MAX_TARGETS
from cayu.sessions._session_continuation import (
    CONTINUATION_NAMESPACE_KEY as CONTINUATION_NAMESPACE_KEY,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_OPERATION_PREFIX as CONTINUATION_OPERATION_PREFIX,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_SCHEMA_VERSION as CONTINUATION_SCHEMA_VERSION,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_SERVICE_PREFIX as CONTINUATION_SERVICE_PREFIX,
)
from cayu.sessions._session_continuation import ContinuationConflict as ContinuationConflict
from cayu.sessions._session_continuation import ContinuationConsumption as ContinuationConsumption
from cayu.sessions._session_continuation import ContinuationEvent as ContinuationEvent
from cayu.sessions._session_continuation import ContinuationLatch as ContinuationLatch
from cayu.sessions._session_continuation import (
    ContinuationLatchReceiver as ContinuationLatchReceiver,
)
from cayu.sessions._session_continuation import ContinuationNamespace as ContinuationNamespace
from cayu.sessions._session_continuation import ContinuationPreparation as ContinuationPreparation
from cayu.sessions._session_continuation import ContinuationRecord as ContinuationRecord
from cayu.sessions._session_continuation import (
    ContinuationRecoveryWriter as ContinuationRecoveryWriter,
)
from cayu.sessions._session_continuation import (
    ContinuationReleasedExecution as ContinuationReleasedExecution,
)
from cayu.sessions._session_continuation import (
    ContinuationReleasedRetirement as ContinuationReleasedRetirement,
)
from cayu.sessions._session_continuation import ContinuationRetirement as ContinuationRetirement
from cayu.sessions._session_continuation import ContinuationService as ContinuationService
from cayu.sessions._session_continuation import (
    ContinuationServiceReference as ContinuationServiceReference,
)
from cayu.sessions._session_continuation import ContinuationTicket as ContinuationTicket
from cayu.sessions._session_continuation import ContinuationUnavailable as ContinuationUnavailable
from cayu.sessions._session_continuation import ContinuationWait as ContinuationWait
from cayu.sessions._session_continuation import (
    RetainedContinuationLatchReceiver as RetainedContinuationLatchReceiver,
)
from cayu.sessions._session_continuation import _aware_time as _aware_time
from cayu.sessions._session_continuation import _bounded_digest as _bounded_digest
from cayu.sessions._session_continuation import (
    continuation_admission_digest as continuation_admission_digest,
)
from cayu.sessions._session_continuation import (
    continuation_admission_inputs as continuation_admission_inputs,
)
from cayu.sessions._session_continuation import continuation_digest as continuation_digest
from cayu.sessions._session_continuation import (
    continuation_namespace_id as continuation_namespace_id,
)
from cayu.sessions._session_continuation import (
    continuation_operation_key as continuation_operation_key,
)
from cayu.sessions._session_continuation import (
    continuation_operation_key_for_registration as continuation_operation_key_for_registration,
)
from cayu.sessions._session_continuation import (
    continuation_registration_operation as continuation_registration_operation,
)
from cayu.sessions._session_continuation import (
    continuation_writer_frontier as continuation_writer_frontier,
)
from cayu.sessions._session_continuation import latch_identity as latch_identity
from cayu.sessions._session_continuation import record_from_json as record_from_json
from cayu.sessions._session_continuation import require_latch_identity as require_latch_identity
from cayu.sessions._session_continuation import (
    require_record_writer_generation as require_record_writer_generation,
)
from cayu.sessions._session_continuation import require_ticket_identity as require_ticket_identity
from cayu.sessions._session_continuation import (
    require_writer_generation as require_writer_generation,
)
from cayu.sessions._session_continuation import ticket_identity as ticket_identity

if TYPE_CHECKING:
    from cayu.sessions._invocation_lifecycle import AdmitInvocationCommand, InvocationMutationResult
    from cayu.sessions.base import SessionStore


def require_operation_record_owner(key: str, record: object) -> None:
    """Validate the reserved continuation key without granting read authority."""

    if not key.startswith(CONTINUATION_OPERATION_PREFIX):
        return
    from cayu.sessions._session_continuation_scope import require_publication

    if type(record) is not dict:
        raise ContinuationConflict("Continuation operation record is not an object.")
    from cayu.sessions._temporary_service_target import (
        TARGET_PREFIX,
        TemporaryServiceTarget,
        target_service_key,
    )

    if key.startswith(TARGET_PREFIX):
        require_publication(key)
        target = TemporaryServiceTarget.model_validate(record)
        if target_service_key(target.service.intent.operation) != key:
            raise ContinuationConflict("Side-session target key conflicts with its operation.")
        return
    if key.startswith(CONTINUATION_SERVICE_PREFIX):
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceRecord,
            temporary_service_key,
        )

        parsed_service = TemporaryServiceRecord.model_validate(record)
        require_publication(continuation_operation_key(parsed_service.intent.ticket))
        if temporary_service_key(parsed_service.intent.operation) != key:
            raise ContinuationConflict("Temporary service key conflicts with its operation.")
        return
    require_publication(key)
    if key == CONTINUATION_NAMESPACE_KEY:
        ContinuationNamespace.model_validate(record)
        return
    parsed = record_from_json(record)
    if continuation_operation_key(parsed.ticket) != key:
        raise ContinuationConflict("Continuation operation key is not content-bound.")


async def admit_continuation(
    store: SessionStore,
    consumption: ContinuationConsumption,
    command: AdmitInvocationCommand,
) -> tuple[InvocationMutationResult, ContinuationRecord]:
    """Admit exactly one retained continuation through the typed lifecycle command.

    The prepared receipt is committed before admission.  If admission fails or
    its acknowledgement is lost, the prepared responsibility remains durable
    and a retry can reconcile it without creating a second invocation.
    """

    if consumption.receipt_stage != "prepared":
        raise ContinuationConflict("Continuation admission requires a prepared receipt.")
    if (
        getattr(command, "session_id", None) != consumption.ticket.session_id
        or getattr(command, "expected_session_instance_id", None)
        != consumption.ticket.session_instance_id
    ):
        raise ContinuationConflict("Invocation admission belongs to another session authority.")
    if continuation_admission_digest(command) != consumption.admission_command_digest:
        raise ContinuationConflict("Invocation admission does not match its retained receipt.")
    if getattr(command, "expected_run_epoch", None) != consumption.admission_expected_run_epoch:
        raise ContinuationConflict(
            "Invocation admission epoch does not match its retained receipt."
        )
    prepared = await store.consume_continuation(consumption)
    retained = prepared.consumption
    if retained is None:
        raise ContinuationConflict("Prepared continuation readback changed before admission.")
    comparable = retained.model_copy(
        update={
            "ticket": consumption.ticket,
            "receipt_stage": "prepared",
            "admission_claimed": False,
            "admission_claim_id": None,
        }
    )
    if comparable != consumption:
        raise ContinuationConflict("Prepared continuation readback changed before admission.")
    if retained.receipt_stage == "admitted":
        from cayu.sessions._invocation_lifecycle import (
            reconcile_invocation_admission_from_state,
        )

        session = await store.load(consumption.ticket.session_id)
        checkpoint = await store.load_checkpoint(consumption.ticket.session_id)
        result = (
            None
            if session is None
            else reconcile_invocation_admission_from_state(
                session,
                checkpoint,
                session_id=consumption.ticket.session_id,
                session_instance_id=consumption.ticket.session_instance_id,
                expected_run_epoch=consumption.admission_expected_run_epoch,
                command_sha256=consumption.admission_command_digest,
                profile_sha256=consumption.profile_digest,
            )
        )
        if result is None:
            raise ContinuationUnavailable("Continuation admission is pending reconciliation.")
        return result, prepared
    if retained.receipt_stage != "prepared":
        raise ContinuationConflict("Continuation responsibility is no longer admissible.")
    from cayu.sessions._session_continuation_scope import admission_claim_scope

    claimed, _ = await store._claim_continuation_admission(consumption)
    claim_consumption = claimed.consumption
    if claim_consumption is None or not claim_consumption.admission_claimed:
        raise ContinuationConflict("Continuation admission claim was not retained.")
    # Exact retries share the claim. The lifecycle transaction verifies its
    # ID atomically; release fences every still-pending dispatch with that ID.
    admitted = claim_consumption.model_copy(
        update={"receipt_stage": "admitted", "admission_claimed": True}
    )
    try:
        with admission_claim_scope(claim_consumption):
            result = await store.apply_invocation_lifecycle_command(command)
    except Exception:
        # Failed reconciliation retains the claim and the original error.
        with suppress(Exception):
            # The transaction distinguishes exact commitment, positive
            # supersession, and a still-uncommitted claim. No pre-read grants
            # exclusion authority or permits a late dispatch after release.
            await store._release_continuation_admission_claim(claim_consumption)
        raise
    settled = await store._finalize_continuation_admission(admitted, command)
    return result, settled


__all__ = [
    "CONTINUATION_NAMESPACE_KEY",
    "CONTINUATION_OPERATION_PREFIX",
    "ContinuationConflict",
    "ContinuationConsumption",
    "ContinuationLatch",
    "ContinuationLatchReceiver",
    "ContinuationNamespace",
    "ContinuationPreparation",
    "ContinuationRecord",
    "ContinuationRetirement",
    "ContinuationService",
    "ContinuationTicket",
    "ContinuationUnavailable",
    "ContinuationWait",
    "continuation_admission_digest",
    "continuation_digest",
    "continuation_namespace_id",
    "continuation_operation_key",
    "continuation_operation_key_for_registration",
    "continuation_registration_operation",
    "latch_identity",
    "record_from_json",
    "require_latch_identity",
    "require_operation_record_owner",
    "require_ticket_identity",
    "require_writer_generation",
]

"""Exact receiving fence for an existing side-session, not a second permit owner."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    continuation_digest,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServicePreparation,
    TemporaryServiceRecord,
    advance_temporary_service_record,
)

TARGET_PREFIX = "session-continuation:target:"
MAX_TARGET_SERVICES = 32
TARGET_WRAPPER_RESERVED_BYTES = 128


def target_service_key(operation) -> str:
    return TARGET_PREFIX + continuation_digest(operation)


class TemporaryServiceTargetReference(ContractValue):
    key: StrictStr = Field(pattern=r"^session-continuation:target:[0-9a-f]{64}$")
    record_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    expected_run_epoch: StrictInt = Field(ge=1, le=2**53 - 1)
    source_acknowledged: StrictBool


def target_reference(record: TemporaryServiceTarget) -> TemporaryServiceTargetReference:
    return TemporaryServiceTargetReference(
        key=target_service_key(record.service.intent.operation),
        record_sha256=continuation_digest(record),
        expected_run_epoch=record.service.admission.dispatch.expected_run_epoch,
        source_acknowledged=record.source_acknowledged,
    )


class TemporaryServiceTarget(ContractValue):
    """Target-owned exact operation retained until its source acknowledges return.

    The target does not arbitrate the source's final latch. That decision belongs
    to the source reservation. This fence arbitrates admission versus exclusion
    under the receiving session's own transaction.
    """

    service: TemporaryServiceRecord
    source_acknowledged: StrictBool = False

    @model_validator(mode="after")
    def validate_target(self) -> TemporaryServiceTarget:
        if self.service.intent.mode != "side_session":
            raise ValueError("A separate receiving fence requires a side-session target.")
        if self.service.state == "reserved":
            raise ValueError("Source reservation is not a receiving admission state.")
        if self.source_acknowledged and self.service.state not in {"returned", "excluded"}:
            raise ValueError("Target acknowledgement requires a terminal receiving result.")
        return self


def admit_side_target(
    current: TemporaryServiceTarget,
    *,
    admission: TemporaryServiceAdmission,
    checkpoint: dict | None,
    session_id: str,
    session_instance_id: str,
    run_epoch: int,
    now: datetime,
) -> TemporaryServiceTarget:
    """Compose a target fence with positive native admission in one transaction."""
    from cayu.runtime._temporary_continuation_scope import require_temporary_transition
    from cayu.runtime._temporary_continuation_store import (
        native_service_execution,
        require_service_deadline,
    )

    require_temporary_transition(admission)
    intent = admission.dispatch.intent
    if (
        current.service.state != "prepared"
        or current.service.admission != admission.preparation
        or session_id != intent.target.object_id
        or session_instance_id != intent.target.incarnation
        or run_epoch != admission.dispatch.expected_run_epoch + 1
    ):
        raise ContinuationConflict("Side-session admission lost its exact receiving fence.")
    require_service_deadline(current.service, now)
    execution = native_service_execution(admission, checkpoint)
    if execution is None:
        raise ContinuationConflict("Side-session admission lacks native permit consumption.")
    return TemporaryServiceTarget(
        service=TemporaryServiceRecord(admission=admission, state="admitted", execution=execution)
    )


def exclude_side_target(
    current: TemporaryServiceTarget, expected: TemporaryServicePreparation
) -> TemporaryServiceTarget:
    """A compare against prepared is exclusion; missing or timed-out work is not."""
    if current.service.admission != expected:
        raise ContinuationConflict("Side-session exclusion changed its exact preparation.")
    if current.service.state == "excluded":
        return current
    if current.service.state != "prepared":
        raise ContinuationConflict("Side-session work was admitted and cannot be excluded.")
    excluded = TemporaryServiceRecord(
        admission=expected,
        state="excluded",
        settlement=ReceivingSettlementReceipt(
            expected=expected.permit,
            receiving_owner=expected.dispatch.intent.target.owner,
            receipt_id="clarification-target-exclusion:" + continuation_digest(expected),
            outcome="quiescent",
            admission_excluded=True,
        ),
    )
    return TemporaryServiceTarget(
        service=advance_temporary_service_record(current.service, excluded)
    )


def return_side_target(
    current: TemporaryServiceTarget, observed: TemporaryServiceRecord
) -> TemporaryServiceTarget:
    """Retain a receiving owner's validated native return for source reconciliation."""
    if observed.state != "returned":
        raise ContinuationConflict("Side-session return requires positive native release.")
    service = advance_temporary_service_record(current.service, observed)
    return current.model_copy(update={"service": service})


def acknowledge_side_target(
    current: TemporaryServiceTarget, source: TemporaryServiceRecord
) -> TemporaryServiceTarget:
    """Called only after the configured source owner authenticates its terminal record."""
    if (
        source.state == "excluded"
        and isinstance(source.admission, TemporaryServiceAdmission)
        and not isinstance(current.service.admission, TemporaryServiceAdmission)
    ):
        # Exclusion can win before the target ever consumes the registration
        # acknowledgement; the source may already have retained that receipt.
        # Compare every original preparation field, not just the operation key.
        source = source.model_copy(update={"admission": source.admission.preparation})
    # Foreign settlement acknowledgement belongs to the source, not the
    # receiving target's native outcome. Compare all native outcome evidence.
    source = source.model_copy(update={"settlement_acknowledged": False})
    if current.service.state not in {"returned", "excluded"} or current.service != source:
        raise ContinuationConflict("Source did not acknowledge the exact target settlement.")
    return current.model_copy(update={"source_acknowledged": True})


def publish_target_record(session, checkpoint, current, previous, proposed, now):
    """Atomic native target compare, bounded index and receipt-retention update."""
    from cayu.collaboration._preparation import prepare_contract
    from cayu.runtime._session_continuation_scope import require_publication
    from cayu.runtime._session_continuation_store import ROOT_KEY, ContinuationRoot
    from cayu.runtime._temporary_continuation_store import (
        native_service_outcome,
        require_service_deadline,
    )
    from cayu.sessions._session_continuation import (
        CONTINUATION_NAMESPACE_KEY,
        ContinuationNamespace,
        continuation_namespace_id,
    )
    from cayu.sessions.base import SessionOperationPublication
    from cayu.vaults.redaction import SecretRedactor

    proposed = prepare_contract(TemporaryServiceTarget, proposed, redactor=SecretRedactor())
    intent = proposed.service.intent
    key = target_service_key(intent.operation)
    require_publication(key)
    if (session.id, session.instance_id) != (intent.target.object_id, intent.target.incarnation):
        raise ContinuationConflict("Side-session fence belongs to another incarnation.")
    namespace = ContinuationNamespace(
        session_id=session.id,
        session_instance_id=session.instance_id,
        owner=intent.target.owner,
        namespace_id=continuation_namespace_id(
            session.id, session.instance_id, intent.target.owner
        ),
    )
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    root = (
        ContinuationRoot(namespace=namespace)
        if raw_root is None
        else ContinuationRoot.model_validate(raw_root)
    )
    if root.namespace != namespace:
        raise ContinuationConflict("Side-session fence has a different native owner.")
    refs = {item.key: item for item in root.target_services}
    if previous is None:
        if current is not None or key in refs:
            raise ContinuationConflict("Side-session fence already has retained responsibility.")
        if proposed.service.state != "prepared" or proposed.source_acknowledged:
            raise ContinuationConflict("Side-session receiving starts with exact preparation.")
        if len(refs) >= MAX_TARGET_SERVICES:
            raise ContinuationConflict("Side-session receiving capacity is exhausted.")
        from cayu.sessions._temporary_continuation import require_temporary_service_capacity

        require_temporary_service_capacity(
            proposed.service, envelope_overhead=TARGET_WRAPPER_RESERVED_BYTES
        )
        require_service_deadline(proposed.service, now)
        if session.run_epoch != proposed.service.admission.dispatch.expected_run_epoch:
            raise ContinuationConflict("Side-session preparation has a stale writer expectation.")
    else:
        previous = prepare_contract(TemporaryServiceTarget, previous, redactor=SecretRedactor())
        if proposed.service.state in {"returned", "excluded"}:
            for replay in (proposed.model_copy(update={"source_acknowledged": True}), proposed):
                if current == replay.model_dump(mode="json") and refs.get(key) == target_reference(
                    replay
                ):
                    advance_temporary_service_record(previous.service, proposed.service)
                    return SessionOperationPublication(
                        checkpoint=checkpoint, operation_records={key: current}
                    )
        if current != previous.model_dump(mode="json") or refs.get(key) != target_reference(
            previous
        ):
            raise ContinuationConflict("Side-session receiving compare lost its exact record.")
        if proposed != previous:
            if previous.source_acknowledged:
                raise ContinuationConflict("Acknowledged target settlement is immutable.")
            if proposed.source_acknowledged:
                if acknowledge_side_target(previous, proposed.service) != proposed:
                    raise ContinuationConflict("Side-session source acknowledgement conflicts.")
            elif proposed.service.state == "admitted":
                expected = admit_side_target(
                    previous,
                    admission=proposed.service.acknowledged_admission,
                    checkpoint=checkpoint,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    now=now,
                )
                if expected != proposed:
                    raise ContinuationConflict("Side-session admission receipt conflicts.")
            elif proposed.service.state == "excluded":
                preparation = previous.service.admission
                if isinstance(preparation, TemporaryServiceAdmission):
                    preparation = preparation.preparation
                if exclude_side_target(previous, preparation) != proposed:
                    raise ContinuationConflict("Side-session exclusion receipt conflicts.")
            elif proposed.service.state == "returned":
                observed = native_service_outcome(
                    proposed.service.acknowledged_admission, checkpoint
                )
                if observed is None or return_side_target(previous, observed) != proposed:
                    raise ContinuationConflict("Side-session return lacks exact native release.")
            else:
                raise ContinuationConflict("Invalid side-session receiving transition.")
    refs[key] = target_reference(proposed)
    root = ContinuationRoot.model_validate(
        root.model_dump() | {"target_services": tuple(refs[item] for item in sorted(refs))}
    )
    records = {key: proposed.model_dump(mode="json")}
    if raw_root is None:
        records[CONTINUATION_NAMESPACE_KEY] = namespace.model_dump(mode="json")
    return SessionOperationPublication(
        checkpoint=({} if checkpoint is None else checkpoint)
        | {ROOT_KEY: root.model_dump(mode="json")},
        operation_records=records,
    )

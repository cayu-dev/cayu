"""Private bounded continuation index, receiving events and erasure authority."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, ContractValue
from cayu.runtime._session_continuation_scope import (
    continuation_authority_visible,
    current_publication_key,
)
from cayu.runtime._temporary_service_target import (
    MAX_TARGET_SERVICES,
    TemporaryServiceTargetReference,
)
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES,
    CONTINUATION_MAX_EVENT_BYTES,
    CONTINUATION_MAX_EVENTS,
    CONTINUATION_MAX_LATCH_EVIDENCE_BYTES,
    CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES,
    CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES,
    CONTINUATION_MAX_SERVICE_REFERENCE_BYTES,
    CONTINUATION_MAX_SERVICES,
    CONTINUATION_NAMESPACE_KEY,
    CONTINUATION_OPERATION_PREFIX,
    ContinuationConflict,
    ContinuationEvent,
    ContinuationNamespace,
    ContinuationPreparation,
    ContinuationRecord,
    ContinuationReleasedExecution,
    ContinuationTicket,
    continuation_operation_key,
)

if TYPE_CHECKING:
    from cayu.sessions.base import Session, SessionOperationPublication

ROOT_KEY = "session_continuations"
MAX_RETAINED_TICKETS = 64
# One namespace, every retained ticket and its bounded children, plus receiving
# responsibilities owned by this session for services originating elsewhere.
MAX_RETAINED_CONTINUATION_RECORDS = (
    1 + MAX_RETAINED_TICKETS * (1 + CONTINUATION_MAX_SERVICES) + MAX_TARGET_SERVICES
)


def collect_retained_record(records: dict[str, Any], key: str, value: Any) -> None:
    """Bound hydration before the shared owner validates the complete index."""
    if type(value) is not dict or len(records) >= MAX_RETAINED_CONTINUATION_RECORDS:
        raise ValueError("Continuation retention evidence is malformed.")
    records[key] = value


def encoded(value: Any) -> bytes:
    return canonical_durable_json_bytes(value, "continuation ownership")


def digest(value: Any) -> str:
    return sha256(encoded(value)).hexdigest()


class _Entry(ContractValue):
    ticket_key: StrictStr = Field(pattern=r"^session-continuation:[0-9a-f]{64}$")
    state: Literal["ARMING", "WAITING", "SERVICING", "CONSUMED", "RETIRED"]
    record_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    originating_writer_generation: StrictInt = Field(ge=1, le=2**53 - 1)
    reserved_bytes: Literal[65536] = MAX_ENVELOPE_BYTES
    admission_claim_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    admission_command_digest: StrictStr | None = None
    admission_expected_run_epoch: StrictInt | None = Field(default=None, ge=1)
    service_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    service_receipt_epochs: tuple[StrictInt, ...] = Field(
        default=(), max_length=CONTINUATION_MAX_SERVICES
    )

    @model_validator(mode="after")
    def complete_admission_identity(self) -> _Entry:
        if (self.admission_command_digest is None) != (self.admission_expected_run_epoch is None):
            raise ValueError("Continuation index has incomplete admission identity.")
        if self.service_digest is not None and self.admission_command_digest is None:
            raise ValueError("Continuation service index lacks its admission identity.")
        if (
            tuple(sorted(set(self.service_receipt_epochs))) != self.service_receipt_epochs
            or any(epoch < 1 or epoch > 2**53 - 1 for epoch in self.service_receipt_epochs)
            or (self.service_receipt_epochs and self.state != "SERVICING")
        ):
            raise ValueError("Continuation index has invalid service receipt retention.")
        return self


class ContinuationRoot(ContractValue):
    schema_version: Literal[2] = 2
    namespace: ContinuationNamespace
    entries: tuple[_Entry, ...] = Field(default=(), max_length=MAX_RETAINED_TICKETS)
    target_services: tuple[TemporaryServiceTargetReference, ...] = Field(
        default=(), max_length=MAX_TARGET_SERVICES
    )

    @model_validator(mode="after")
    def unique_entries(self) -> ContinuationRoot:
        if len({entry.ticket_key for entry in self.entries}) != len(self.entries):
            raise ValueError("Continuation index contains duplicate responsibility.")
        if len({entry.key for entry in self.target_services}) != len(self.target_services):
            raise ValueError("Continuation index contains duplicate target responsibility.")
        if len(encoded(self.model_dump(mode="json"))) > MAX_ENVELOPE_BYTES:
            raise ValueError("Continuation responsibility index exceeds its envelope.")
        return self


def require_reserved_capacity(preparation: ContinuationPreparation) -> None:
    """Reserve all mandatory future representations, including JSON expansion.

    Evidence limits measure canonical serialized bytes, not characters. The
    immutable ticket occurs five times in the largest complete aggregate. The
    extra 32 bytes per ticket cover every future state/revision spelling; the
    1 KiB allowance covers object framing and optional fields, in addition to
    separately bounded complete event records.
    """
    ticket = preparation.intent
    required = (
        len(encoded(preparation.model_dump(mode="json")))
        + len(encoded(ticket.namespace.model_dump(mode="json")))
        + 5 * (len(encoded(ticket.model_dump(mode="json"))) + 32)
        + 2 * CONTINUATION_MAX_LATCH_EVIDENCE_BYTES
        + CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES
        + CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES
        + CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES
        + CONTINUATION_MAX_EVENTS * CONTINUATION_MAX_EVENT_BYTES
        + (
            0
            if ticket.service_policy == "none"
            else CONTINUATION_MAX_SERVICES * CONTINUATION_MAX_SERVICE_REFERENCE_BYTES
        )
        + 1024
    )
    if required > MAX_ENVELOPE_BYTES:
        raise ContinuationConflict("Continuation cannot reserve complete terminal evidence.")


def service_receipt_epochs(record: ContinuationRecord) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                item.expected_run_epoch + 1
                for item in record.services
                if item.mode == "same_session" and item.state in {"reserved", "admitted"}
            }
        )
    )


def require_released_wait_invocation(
    ticket: ContinuationTicket,
    checkpoint: dict[str, Any] | None,
    *,
    permit_operation: str,
    permit_commitment: str,
) -> ContinuationReleasedExecution:
    """Authenticate original admission and release using the native receipt owner."""
    from cayu.sessions._invocation_lifecycle import (
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_from_checkpoint,
    )

    identity = f"{ticket.session_id}:{ticket.session_instance_id}:{ticket.writer_generation}"
    admission = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint, command_identity="admit:" + identity
    )
    release = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint, command_identity="release:" + identity
    )
    if (
        admission is None
        or release is None
        or admission.kind is not InvocationLifecycleCommandKind.ADMIT
        or release.kind is not InvocationLifecycleCommandKind.RELEASE
        or admission.session_id != ticket.session_id
        or admission.session_instance_id != ticket.session_instance_id
        or release.session_id != ticket.session_id
        or release.session_instance_id != ticket.session_instance_id
        or admission.participant_permit_operation != permit_operation
        or admission.participant_permit_commitment != permit_commitment
        or admission.active_profile.interaction_id != ticket.interaction_id
        or admission.active_profile.run_epoch != ticket.writer_generation
        or release.active_profile != admission.active_profile
        or release.result_session.run_epoch != ticket.writer_generation + 1
    ):
        raise ContinuationConflict("Execution wait lacks exact native writer-release evidence.")
    return ContinuationReleasedExecution(
        permit_operation=permit_operation,
        permit_commitment=permit_commitment,
        admission_receipt_sha256=admission.record_sha256,
        release_receipt_sha256=release.record_sha256,
    )


def checkpoint_visible() -> bool:
    return continuation_authority_visible()


def project_checkpoint_root(current, replacement, *, session_id: str) -> dict[str, Any] | None:
    before = None if current is None else current.get(ROOT_KEY)
    after = replacement.get(ROOT_KEY)
    previous = None if before is None else ContinuationRoot.model_validate(before)
    if previous is not None and previous.namespace.session_id != session_id:
        raise ContinuationConflict("Continuation index belongs to another session.")
    if checkpoint_visible():
        if after is None:
            raise ContinuationConflict("Continuation publication cannot erase its index.")
        proposed = ContinuationRoot.model_validate(after)
        if proposed.namespace.session_id != session_id or (
            previous is not None and proposed.namespace != previous.namespace
        ):
            raise ContinuationConflict("Continuation namespace changed during publication.")
        return proposed.model_dump(mode="json")
    if after is not None and (before is None or encoded(after) != encoded(before)):
        raise ContinuationConflict(
            "Generic checkpoint mutation cannot create continuation authority."
        )
    return None if previous is None else previous.model_dump(mode="json")


def require_history(record: ContinuationRecord) -> None:
    key = continuation_operation_key(record.ticket)
    claim_kinds = {"admission_claimed", "admission_claim_released"}
    seen: set[str] = set()
    claim_state: bool | None = None
    history_valid = True
    for event in record.events:
        if event.kind in claim_kinds:
            # A known pre-commit rejection may release the claim and a later
            # retry may acquire it again.  That is the only repeatable part of
            # the history, and it must alternate rather than duplicate.
            if claim_state is not None and claim_state == (event.kind == "admission_claimed"):
                history_valid = False
            claim_state = event.kind == "admission_claimed"
        elif event.kind in seen:
            history_valid = False
        seen.add(event.kind)
    if record.consumption is not None and record.consumption.receipt_stage == "prepared":
        if claim_state is None:
            # Compact histories intentionally omit in-flight claim events;
            # the authenticated current consumption carries that state.
            claim_state = record.consumption.admission_claimed
        history_valid = history_valid and record.consumption.admission_claimed == (
            claim_state is True
        )
    if (
        not record.events
        or record.events[0].kind != "prepared"
        or record.retirement_acknowledged != ("retirement_acknowledged" in seen)
        or (record.retirement_acknowledged and record.events[-1].kind != "retirement_acknowledged")
        or not history_valid
        or any(
            event.sequence != index or event.ticket_key != key
            for index, event in enumerate(record.events, 1)
        )
        or record.events[-1].record_sha256
        != digest(record.model_dump(mode="json", exclude={"events"}))
    ):
        raise ContinuationConflict("Continuation history does not authenticate its current state.")


def _event_kind(before: ContinuationRecord | None, after: ContinuationRecord):
    if before is None:
        return "prepared"
    if after.retirement_acknowledged != before.retirement_acknowledged:
        if (
            before.retirement_acknowledged
            or before.released_retirement is None
            or before.model_copy(update={"retirement_acknowledged": True}) != after
        ):
            raise ContinuationConflict("Invalid retirement acknowledgement transition.")
        return "retirement_acknowledged"
    if after.retirement != before.retirement:
        return "retired"
    if after.consumption != before.consumption:
        assert after.consumption is not None
        if (
            before.consumption is not None
            and before.consumption.receipt_stage == after.consumption.receipt_stage == "prepared"
        ):
            # Claim/release is mutable in-flight ownership, not terminal
            # lifecycle. Keep it authenticated by the current record digest
            # without consuming one of the bounded terminal event slots.
            return None
        return {"prepared": "consumption_prepared", "admitted": "admitted", "excluded": "excluded"}[
            after.consumption.receipt_stage
        ]
    if after.latch != before.latch:
        return "latched"
    if after.ticket.state == "WAITING" and before.ticket.state == "ARMING":
        return "parked"
    raise ContinuationConflict("Unsupported continuation history transition.")


def index_publication(
    session: Session,
    checkpoint: dict[str, Any] | None,
    current_record: dict[str, Any] | None,
    publication: SessionOperationPublication,
    *,
    key: str,
) -> SessionOperationPublication:
    if current_publication_key() != key or set(publication.operation_records) != {key}:
        raise ContinuationConflict("Continuation publication lost exact ownership.")
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    root = None if raw_root is None else ContinuationRoot.model_validate(raw_root)
    raw = publication.operation_records[key]
    if key == CONTINUATION_NAMESPACE_KEY:
        namespace = ContinuationNamespace.model_validate(raw)
        if root is None:
            if current_record is not None:
                raise ContinuationConflict("Retained namespace has lost its responsibility index.")
            root = ContinuationRoot(namespace=namespace)
        elif root.namespace != namespace:
            raise ContinuationConflict("Continuation namespace conflicts with its index.")
    else:
        record = ContinuationRecord.model_validate(raw)
        if root is None or root.namespace != record.namespace:
            raise ContinuationConflict("Continuation has no authoritative namespace index.")
        entries = {entry.ticket_key: entry for entry in root.entries}
        before = (
            None if current_record is None else ContinuationRecord.model_validate(current_record)
        )
        entry = entries.get(key)
        if before is None:
            if entry is not None or len(entries) >= MAX_RETAINED_TICKETS:
                raise ContinuationConflict("Continuation retained capacity is unavailable.")
            require_reserved_capacity(record.preparation)
            if record.events:
                raise ContinuationConflict("New continuation cannot supply owner history.")
            event = ContinuationEvent(
                sequence=1,
                ticket_key=key,
                kind="prepared",
                record_sha256=digest(record.model_dump(mode="json", exclude={"events"})),
            )
            record = ContinuationRecord.model_validate(record.model_dump() | {"events": (event,)})
        else:
            require_history(before)
            if (
                entry is None
                or entry.record_sha256 != digest(current_record)
                or entry.originating_writer_generation != before.ticket.writer_generation
                or entry.service_receipt_epochs != service_receipt_epochs(before)
                or entry.service_digest
                != (None if before.consumption is None else before.consumption.service_digest)
                or record.events != before.events
                or record.services != before.services
                or (
                    record.released_retirement != before.released_retirement
                    and (before.retirement is not None or record.retirement is None)
                )
            ):
                raise ContinuationConflict(
                    "Continuation index or history conflicts with its receipt."
                )
            if before != record:
                event_kind = _event_kind(before, record)
                if event_kind is None:
                    if before is None or before.events != record.events:
                        raise ContinuationConflict("Invalid compact continuation claim transition.")
                    if (
                        before.ticket != record.ticket
                        or before.latch != record.latch
                        or before.preparation != record.preparation
                        or before.retirement != record.retirement
                        or before.consumption is None
                        or record.consumption is None
                        or before.consumption.model_copy(
                            update={
                                "admission_claimed": False,
                                "admission_claim_id": None,
                            }
                        )
                        != record.consumption.model_copy(
                            update={
                                "admission_claimed": False,
                                "admission_claim_id": None,
                            }
                        )
                    ):
                        raise ContinuationConflict("Invalid compact continuation claim transition.")
                    if record.events:
                        last = record.events[-1]
                        record = record.model_copy(
                            update={
                                "events": (
                                    *record.events[:-1],
                                    last.model_copy(
                                        update={
                                            "record_sha256": digest(
                                                record.model_dump(mode="json", exclude={"events"})
                                            )
                                        }
                                    ),
                                )
                            }
                        )
                    else:
                        raise ContinuationConflict("Continuation claim lacks prepared history.")
                else:
                    event = ContinuationEvent(
                        sequence=len(record.events) + 1,
                        ticket_key=key,
                        kind=event_kind,
                        record_sha256=digest(record.model_dump(mode="json", exclude={"events"})),
                    )
                    record = ContinuationRecord.model_validate(
                        record.model_dump() | {"events": (*record.events, event)}
                    )
        require_history(record)
        raw = record.model_dump(mode="json")
        consumption = record.consumption
        entries[key] = _Entry(
            ticket_key=key,
            state=record.ticket.state,
            record_sha256=digest(raw),
            originating_writer_generation=record.ticket.writer_generation,
            admission_claim_id=None if consumption is None else consumption.admission_claim_id,
            admission_command_digest=(
                None if consumption is None else consumption.admission_command_digest
            ),
            admission_expected_run_epoch=(
                None if consumption is None else consumption.admission_expected_run_epoch
            ),
            service_digest=None if consumption is None else consumption.service_digest,
            service_receipt_epochs=service_receipt_epochs(record),
        )
        root = root.model_copy(update={"entries": tuple(entries[item] for item in sorted(entries))})
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
    ):
        raise ContinuationConflict("Continuation index belongs to another session incarnation.")
    updated = dict(publication.checkpoint)
    updated[ROOT_KEY] = root.model_dump(mode="json")
    return publication.model_copy(update={"checkpoint": updated, "operation_records": {key: raw}})


def pending_admission_receipt_identities(session: Session, checkpoint) -> frozenset[str]:
    """Retain exact native handoffs while their continuation records are retained."""
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return frozenset()
    root = ContinuationRoot.model_validate(raw)
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
    ):
        raise ContinuationConflict("Continuation receipt retention belongs to another session.")
    final_admissions = frozenset(
        f"admit:{session.id}:{session.instance_id}:{entry.admission_expected_run_epoch + 1}"
        for entry in root.entries
        if (
            entry.state in {"WAITING", "SERVICING"}
            or (entry.state == "CONSUMED" and entry.service_digest is not None)
        )
        and entry.admission_expected_run_epoch is not None
    )
    service_receipts = frozenset(
        f"{kind}:{session.id}:{session.instance_id}:{epoch}"
        for entry in root.entries
        for epoch in entry.service_receipt_epochs
        for kind in ("admit", "release")
    )
    target_receipts = frozenset(
        f"{kind}:{session.id}:{session.instance_id}:{item.expected_run_epoch + 1}"
        for item in root.target_services
        if not item.source_acknowledged
        for kind in ("admit", "release")
    )
    original_receipts = frozenset(
        f"{kind}:{session.id}:{session.instance_id}:{entry.originating_writer_generation}"
        for entry in root.entries
        if entry.state in {"ARMING", "WAITING", "SERVICING"}
        for kind in ("admit", "release")
    )
    # An exactly registered native service can outlive its failed observer.
    # Admission-only handoffs do not own native execution/release observation;
    # they retain their existing settlement boundary rather than borrowing the
    # service contract from an absent or caller-shaped host marker.
    # Keep the native release/rebind lineage rather than creating a host receipt
    # cache. Existing admission/release capacity reservation accounts for these
    # protected receipts before dispatch; pruning the owning namespace releases
    # retention. Historical quiescence is never current execution authority.
    from cayu.sessions._invocation_lifecycle import (
        _invocation_lifecycle_receipt_ledger_from_checkpoint,
    )

    ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    consumed = set()
    for entry in root.entries:
        if (
            entry.state != "CONSUMED"
            or entry.service_digest is None
            or entry.admission_expected_run_epoch is None
        ):
            continue
        identity = (
            f"admit:{session.id}:{session.instance_id}:{entry.admission_expected_run_epoch + 1}"
        )
        original = next(
            (item for item in ledger.receipts if item.command_identity == identity), None
        )
        if original is None or original.command_sha256 != entry.admission_command_digest:
            raise ContinuationConflict("Consumed continuation lost its native admission evidence.")
        consumed.update(
            item.command_identity
            for item in ledger.receipts
            if item.session_id == session.id
            and item.session_instance_id == session.instance_id
            and item.active_profile.interaction_id == original.active_profile.interaction_id
            and item.active_profile.profile == original.active_profile.profile
            and item.active_profile.run_epoch >= original.active_profile.run_epoch
        )
    return final_admissions | service_receipts | target_receipts | original_receipts | consumed


def require_admission_claim(session: Session, checkpoint, command) -> None:
    """Compare the exact claim in the same transaction that admits the session."""
    from cayu.runtime._session_continuation_scope import current_admission_claim
    from cayu.runtime._temporary_continuation_scope import require_temporary_admission
    from cayu.sessions._session_continuation import continuation_admission_digest

    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is not None:
        root = ContinuationRoot.model_validate(raw)
        if (
            root.namespace.session_id != session.id
            or root.namespace.session_instance_id != session.instance_id
        ):
            raise ContinuationConflict(
                "Continuation admission index belongs to another incarnation."
            )
        servicing = tuple(entry for entry in root.entries if entry.state == "SERVICING")
        if servicing:
            temporary = require_temporary_admission(command)
            if temporary is None or any(
                entry.ticket_key != continuation_operation_key(temporary.dispatch.intent.ticket)
                for entry in servicing
            ):
                raise ContinuationConflict(
                    "Unsettled temporary service fences a competing invocation."
                )

    claim = current_admission_claim()
    if claim is None:
        return
    ticket = claim.ticket
    root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if (
        session.id != ticket.session_id
        or session.instance_id != ticket.session_instance_id
        or root is None
        or continuation_admission_digest(command) != claim.admission_command_digest
    ):
        raise ContinuationConflict("Continuation admission lost its exact claim.")
    authority = ContinuationRoot.model_validate(root)
    entry = next(
        (
            item
            for item in authority.entries
            if item.ticket_key == continuation_operation_key(ticket)
        ),
        None,
    )
    if (
        authority.namespace != ticket.namespace
        or entry is None
        or entry.state != "WAITING"
        or entry.admission_claim_id != claim.admission_claim_id
        or entry.admission_command_digest != claim.admission_command_digest
    ):
        raise ContinuationConflict("Continuation admission claim was released or retired.")


def require_erasure_quiescence(*, session: Session, checkpoint, records: dict[str, Any]) -> None:
    owned = {
        key: raw for key, raw in records.items() if key.startswith(CONTINUATION_OPERATION_PREFIX)
    }
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw_root is None:
        if owned:
            raise ContinuationConflict("Continuation records have lost their deletion fence.")
        return
    root = ContinuationRoot.model_validate(raw_root)
    namespace = root.namespace
    if namespace.session_id != session.id or namespace.session_instance_id != session.instance_id:
        raise ContinuationConflict("Continuation deletion authority belongs to another session.")
    if owned.pop(CONTINUATION_NAMESPACE_KEY, None) != namespace.model_dump(mode="json"):
        raise ContinuationConflict("Continuation namespace evidence is unavailable.")
    expected_keys = {entry.ticket_key for entry in root.entries}
    from cayu.runtime._temporary_service_target import TemporaryServiceTarget, target_reference

    for reference in root.target_services:
        expected_keys.add(reference.key)
        raw = owned.get(reference.key)
        if raw is None:
            raise ContinuationConflict("Side-session receiving deletion evidence is unavailable.")
        target = TemporaryServiceTarget.model_validate(raw)
        if (
            target_reference(target) != reference
            or target.service.intent.target.object_id != session.id
            or target.service.intent.target.incarnation != session.instance_id
            or not target.source_acknowledged
        ):
            raise ContinuationConflict(
                "Session still owns a side-session receiving responsibility."
            )
    for entry in root.entries:
        raw = owned.get(entry.ticket_key)
        if raw is None:
            raise ContinuationConflict("Continuation deletion evidence is unavailable.")
        record = ContinuationRecord.model_validate(raw)
        expected_keys.update(item.key for item in record.services)
    if set(owned) != expected_keys:
        raise ContinuationConflict("Continuation deletion index is incomplete.")
    for entry in root.entries:
        raw = owned[entry.ticket_key]
        record = ContinuationRecord.model_validate(raw)
        require_history(record)
        if (
            entry.record_sha256 != digest(raw)
            or entry.originating_writer_generation != record.ticket.writer_generation
            or entry.state != record.ticket.state
            or entry.service_receipt_epochs != service_receipt_epochs(record)
            or entry.service_digest
            != (None if record.consumption is None else record.consumption.service_digest)
        ):
            raise ContinuationConflict("Continuation deletion evidence conflicts.")
        if record.ticket.state not in {"CONSUMED", "RETIRED"}:
            raise ContinuationConflict("Session still owns a continuation responsibility.")
        if record.released_retirement is not None and not record.retirement_acknowledged:
            raise ContinuationConflict("Continuation exclusion acknowledgement is pending.")
        from cayu.sessions._session_continuation import require_ticket_identity
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceRecord,
            reference_for_service,
        )

        for reference in record.services:
            service = TemporaryServiceRecord.model_validate(owned[reference.key])
            require_ticket_identity(record.ticket, service.intent.ticket)
            if reference != reference_for_service(
                service, record.services
            ) or service.state not in {"returned", "excluded"}:
                raise ContinuationConflict("Session still owns unresolved temporary service.")
            if not service.settlement_acknowledged:
                raise ContinuationConflict(
                    "Temporary service settlement acknowledgement is pending."
                )


def require_continuation_selection_quiescence(checkpoint, *, session: Session) -> None:
    """Reject retained wait/service responsibility at ordinary recipient selection.

    This reads the native index; it neither retires a ticket nor supplies the
    stronger receiving evidence needed to discharge its obligations.
    """
    if checkpoint is None or ROOT_KEY not in checkpoint:
        return
    root = ContinuationRoot.model_validate(checkpoint[ROOT_KEY])
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
        or any(entry.state not in {"CONSUMED", "RETIRED"} for entry in root.entries)
        or any(not entry.source_acknowledged for entry in root.target_services)
    ):
        raise ContinuationConflict("Recipient still owns continuation responsibility.")


def import_history_checkpoint(checkpoint, *, session: Session):
    if checkpoint is None or ROOT_KEY not in checkpoint:
        return checkpoint
    root = ContinuationRoot.model_validate(checkpoint[ROOT_KEY])
    if (
        root.namespace.session_id != session.id
        or any(entry.state not in {"CONSUMED", "RETIRED"} for entry in root.entries)
        or any(not entry.source_acknowledged for entry in root.target_services)
    ):
        raise ContinuationConflict(
            "Unsettled continuation responsibility cannot be imported as history."
        )
    return {key: value for key, value in checkpoint.items() if key != ROOT_KEY}

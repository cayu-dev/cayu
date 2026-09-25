"""Native capacity for exact registered cleanup, never effect authorization."""

from pydantic import Field, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts.resources import (
    ResourceAcquisitionCommand,
    ResourceAcquisitionReceipt,
    ResourceOwnerConflict,
    ResourceOwnerError,
    ResourceOwnerUnavailable,
    ResourceTransferCommand,
    ResourceTransferTemplate,
    _validate_preparation_permit,
    resource_operation_digest,
)
from cayu.collaboration._contracts import MAX_ID_BYTES, ContractValue
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration._preparation import prepare_contract

CLEANUP_EVENT_SLOTS = 2  # released, then responsibility_settled


class CleanupReservation(ContractValue):
    command: ResourceAcquisitionCommand | ResourceTransferCommand | ResourceTransferTemplate = (
        Field(discriminator="kind")
    )
    permit: PermitCommand

    @model_validator(mode="after")
    def coherent(self):
        owner = (
            self.command.source
            if isinstance(self.command, ResourceAcquisitionCommand)
            else self.command.destination
        )
        _validate_preparation_permit(owner, self.command, self.permit)
        return self

    @property
    def family(self):
        return "operations" if isinstance(self.command, ResourceAcquisitionCommand) else "transfers"

    def matches(self, command):
        if isinstance(self.command, ResourceTransferTemplate):
            return (
                isinstance(command, ResourceTransferCommand)
                and self.command.bind(command.intent.receipt) == command
            )
        return self.command == command


def reservations(journal):
    """Validate registration and deduplicate its admitted native representation."""
    values = journal.get("cleanup_reservations")
    if type(values) is not dict:
        raise ResourceOwnerUnavailable("Resource cleanup capacity is unavailable.")
    result = []
    for digest, raw in values.items():
        reservation = CleanupReservation.model_validate(raw)
        if digest != resource_operation_digest(reservation.command):
            raise ResourceOwnerUnavailable("Cleanup reservation operation conflicts.")
        record = journal[reservation.family].get(digest)
        if record is not None:
            schema = (
                ResourceAcquisitionCommand
                if reservation.family == "operations"
                else ResourceTransferCommand
            )
            if not reservation.matches(schema.model_validate_json(record["command"])):
                raise ResourceOwnerConflict("Native resource differs from its cleanup reservation.")
        result.append((digest, reservation, record is None))
    return result


def bind_cleanup_capacity(journal, registrations, *, redactor):
    """Called under the native journal transaction before planning qualification.

    Rebinding never erases older reservations: their exact operations can still
    require cleanup. Capacity is checked by the same journal publication owner.
    """
    for command, permit in registrations:
        checked = prepare_contract(
            CleanupReservation, {"command": command, "permit": permit}, redactor=redactor
        )
        digest = resource_operation_digest(checked.command)
        raw = journal["cleanup_reservations"].get(digest)
        if raw is not None and CleanupReservation.model_validate(raw) != checked:
            raise ResourceOwnerConflict("Registered cleanup capacity conflicts.")
        journal["cleanup_reservations"][digest] = checked.model_dump(mode="json")


def require_cleanup_capacity(journal, command, permit):
    raw = journal["cleanup_reservations"].get(resource_operation_digest(command))
    if raw is None:
        raise ResourceOwnerUnavailable("Planning requires retained native cleanup capacity.")
    reservation = CleanupReservation.model_validate(raw)
    if reservation.command != command or reservation.permit != permit:
        raise ResourceOwnerConflict("Planning cleanup capacity has another registration.")


def cleanup_event_slots(journal, command, permit):
    """The actual terminal write consumes only its own pre-reserved capacity."""
    digest = resource_operation_digest(command)
    raw = journal["cleanup_reservations"].get(digest)
    if raw is None:
        return None
    reservation = CleanupReservation.model_validate(raw)
    if not reservation.matches(command) or reservation.permit != permit:
        raise ResourceOwnerConflict("Cleanup differs from its retained capacity registration.")
    return CLEANUP_EVENT_SLOTS


def require_operation_capacity(journal, family, maximum):
    keys = set(journal[family])
    keys.update(
        digest for digest, reservation, _ in reservations(journal) if reservation.family == family
    )
    if len(keys) > maximum:
        raise ResourceOwnerError("Resource operation and cleanup reservation capacity exhausted.")


def terminal_projection(reservation):
    """Conservative size-only projection. Never stored or returned as evidence."""
    command = reservation.command
    if isinstance(command, ResourceTransferTemplate):
        acquisition = command.acquisition
        # Native receipts bind bounded identifiers. Quotes maximize valid JSON
        # escaping, including the second encoding inside the journal command.
        material_ids = ('"' * MAX_ID_BYTES,) * acquisition.intent.max_materials
        receipt = ResourceAcquisitionReceipt(
            command=acquisition,
            receipt_id='"' * MAX_ID_BYTES,
            stage="owned",
            operation_digest=resource_operation_digest(acquisition),
            material_ids=material_ids,
            content_commitment="sha256:" + "f" * 64,
            manifest_commitment="f" * 64,
            material_count=len(material_ids),
            total_bytes=acquisition.intent.max_total_bytes,
            allowed_operations=acquisition.intent.allowed_operations,
        )
        command = command.bind(receipt)
    return {
        "command": canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_cleanup_capacity"
        ).decode(),
        "stage": "released",
        "receipt": None,
        "material_ids": [],
        "cleanup_permit": reservation.permit.model_dump(mode="json"),
        "event_slots_remaining": CLEANUP_EVENT_SLOTS,
        "responsibility_settled": False,
    }


def include_cleanup_envelope(journal, envelope):
    for digest, reservation, unused in reservations(journal):
        if not unused:
            continue
        envelope[reservation.family][digest] = terminal_projection(reservation)
        envelope["events"].extend(
            {
                "operation" if reservation.family == "operations" else "transfer": digest,
                "stage": "responsibility_settled",
            }
            for _ in range(CLEANUP_EVENT_SLOTS)
        )
    return envelope

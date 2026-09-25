"""Exact preparation discharge under LocalArtifactResourceOwner's mutation fence."""

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts._resource_cleanup_capacity import cleanup_event_slots
from cayu.artifacts.resources import (
    RESOURCE_MAX_OPERATIONS,
    ResourceAcquisitionReceipt,
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
    ResourcePreparationReceipt,
    ResourceTransferCommand,
    ResourceTransferReceipt,
    _append_resource_event,
    _reserve_events,
    _ResourceSettlementReader,
    _validate_preparation_permit,
    resource_operation_digest,
)
from cayu.collaboration._contracts import ExactMatch


async def settle_preparation(owner, command, permit, *, source_owner=None):
    _validate_preparation_permit(owner.owner, command, permit)
    await owner._preparation_reader.authorize_preparation_cleanup(command, permit)
    transfer = isinstance(command, ResourceTransferCommand)
    family = "transfers" if transfer else "operations"
    digest = resource_operation_digest(command)
    encoded = canonical_durable_json_bytes(
        command.model_dump(mode="json"), "resource_cleanup"
    ).decode()
    if transfer:
        async with owner._journal.transaction() as journal:
            absent = digest not in journal[family]
        if absent:
            # Both long-lived native mutation fences remain held. Never nest
            # journal transactions: source and destination may share a journal.
            await require_transfer_source(owner, source_owner, command)
    async with owner._journal.transaction() as journal:
        record = journal[family].get(digest)
        authorization = journal["authorizations"].get(digest)
        if authorization is not None and (
            not isinstance(authorization, dict)
            or authorization.get("command") != encoded
            or ResourcePreparationReceipt.model_validate(authorization["receipt"]).permit != permit
        ):
            raise ResourceOwnerConflict("Cleanup differs from retained preparation authority.")
        if record is None:
            if len(journal[family]) >= RESOURCE_MAX_OPERATIONS:
                raise ResourceOwnerUnavailable("Preparation exclusion capacity exhausted.")
            # A normal bounded operation slot owns this exclusion. No material
            # or lease is invented, and every late native dispatch sees released.
            slots = cleanup_event_slots(journal, command, permit)
            record = {
                "command": encoded,
                "stage": "released",
                "receipt": None,
                "material_ids": [],
                "cleanup_permit": permit.model_dump(mode="json"),
                "event_slots_remaining": _reserve_events(journal) if slots is None else slots,
            }
            journal[family][digest] = record
            _append_resource_event(journal, family, digest, "released")
        elif not isinstance(record, dict) or record.get("command") != encoded:
            raise ResourceOwnerConflict("Cleanup differs from retained operation identity.")
        elif "cleanup_permit" in record:
            if record["cleanup_permit"] != permit.model_dump(mode="json"):
                raise ResourceOwnerConflict("Cleanup differs from retained exclusion authority.")
        elif authorization is None:
            raise ResourceOwnerUnavailable("Retained operation lacks preparation authority.")
        stage = record["stage"]
        raw_material_ids = record["material_ids"]
        if not isinstance(raw_material_ids, list) or any(
            not isinstance(item, str) for item in raw_material_ids
        ):
            raise ResourceOwnerUnavailable("Preparation material identity is malformed.")
        material_ids = tuple(raw_material_ids)
        original = record.get("accepted_receipt" if transfer else "owned_receipt")
    if stage in {"owned", "accepted", "releasing"}:
        if transfer:
            await owner._release_transfer(ResourceTransferReceipt.model_validate(original))
        else:
            await owner._release(ResourceAcquisitionReceipt.model_validate(original))
    elif stage in {"pending", "uncertain", "cleaning"}:
        if transfer:
            await owner._release_pending_transfer(digest, material_ids)
        else:
            await owner._release_pending_materials(digest, material_ids)
    elif stage != "released":
        raise ResourceOwnerUnavailable("Preparation has an unsupported cleanup state.")
    await owner._settle_responsibility(digest, family)
    result = await _ResourceSettlementReader(owner, digest, family).lookup(permit)
    if not isinstance(result, ExactMatch):
        raise ResourceOwnerUnavailable("Preparation discharge has no exact native evidence.")
    return result.receipt


async def require_transfer_source(owner, source, command):
    """Authenticate historical acquisition while both native mutation fences are held.

    This proves the receiving command, not current disclosure or acquisition
    rights. Cleanup must not renew those rights after revocation.
    """
    if (
        source is None
        or source.owner != command.source
        or owner._store.id != source._store.id
        or owner._store.root != source._store.root
        or owner._store._root_identity != source._store._root_identity
    ):
        raise ResourceOwnerUnavailable("Transfer exclusion requires its exact source owner.")
    acquisition = command.intent.receipt.command
    digest = resource_operation_digest(acquisition)
    async with source._journal.transaction() as journal:
        record = journal["operations"].get(digest)
        if (
            not isinstance(record, dict)
            or record.get("command")
            != canonical_durable_json_bytes(
                acquisition.model_dump(mode="json"), "resource_source"
            ).decode()
            or ResourceAcquisitionReceipt.model_validate(record.get("owned_receipt"))
            != command.intent.receipt
        ):
            raise ResourceOwnerConflict("Transfer cleanup source receipt is not authoritative.")

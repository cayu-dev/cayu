"""Content-bound native material references for bounded receiving preparations."""

from hashlib import sha256

from pydantic import model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.artifacts._resource_material_types import ResourceMaterialReference
from cayu.artifacts.resources import (
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
    ResourcePreparationReceipt,
    ResourceTransferCommand,
    ResourceTransferReceipt,
    ResourceTransferTemplate,
    _validate_preparation_permit,
    resource_operation_digest,
)
from cayu.collaboration._contracts import ContractValue, ExactMatch


def material_commitment(value):
    return (
        "sha256:"
        + sha256(
            canonical_bounded_durable_json_bytes(
                value.model_dump(mode="json", warnings=False),
                "resource material",
                max_bytes=1024 * 1024,
                max_nodes=8192,
                max_nesting=64,
            )
        ).hexdigest()
    )


def transfer_template(command):
    return ResourceTransferTemplate(
        operation=command.operation,
        source=command.source,
        destination=command.destination,
        initiator=command.initiator,
        acquisition=command.intent.receipt.command,
        acceptance_generation=command.intent.acceptance_generation,
    )


class ResourceMaterialReadback(ContractValue):
    reference: ResourceMaterialReference
    transfer: ResourceTransferReceipt
    preparation: ResourcePreparationReceipt

    @model_validator(mode="after")
    def coherent(self):
        if self.reference != material_reference(self.transfer, self.preparation):
            raise ValueError("Resource material readback conflicts with its exact reference.")
        return self


def material_reference(transfer, preparation):
    if (
        transfer.stage != "accepted"
        or preparation.owner != transfer.command.destination
        or preparation.operation_digest != transfer.operation_digest
    ):
        raise ResourceOwnerConflict("Resource material is not an accepted exact preparation.")
    return ResourceMaterialReference(
        owner=transfer.command.destination,
        operation=transfer.command.operation,
        template_commitment=material_commitment(transfer_template(transfer.command)),
        transfer_commitment=material_commitment(transfer),
        preparation_commitment=material_commitment(preparation),
    )


async def retained_material(owner, expected):
    """Private ownership evidence under the mutation fence, never disclosure authority.

    An authenticated child owner can prove continued adoption after the original
    acquisition grant expires. New creation/public readback must additionally
    perform current authorization in read_material below.
    """
    if expected.owner != owner.owner:
        raise ResourceOwnerConflict("Resource material belongs to another receiving owner.")
    digest = sha256(
        canonical_bounded_durable_json_bytes(
            expected.operation.model_dump(mode="json"),
            "resource identity",
            max_bytes=8192,
            max_nodes=256,
            max_nesting=16,
        )
    ).hexdigest()
    async with owner._journal.transaction() as journal:
        record = journal["transfers"].get(digest)
        if not isinstance(record, dict):
            raise ResourceOwnerUnavailable("Resource material is not retained.")
        if (
            record.get("stage") != "accepted"
            or record.get("accepted_receipt") is None
            or record.get("accepted_receipt") != record.get("receipt")
            or not any(
                event.get("transfer") == digest and event.get("stage") == "accepted"
                for event in journal["events"]
            )
        ):
            raise ResourceOwnerUnavailable("Resource material lacks durable acceptance evidence.")
        command = ResourceTransferCommand.model_validate_json(record["command"])
        authorization = journal["authorizations"].get(digest)
        if not isinstance(authorization, dict) or authorization.get("command") != record["command"]:
            raise ResourceOwnerUnavailable("Resource material lacks exact preparation evidence.")
        preparation = ResourcePreparationReceipt.model_validate(authorization["receipt"])
    if (
        command.operation != expected.operation
        or resource_operation_digest(command) != digest
        or material_commitment(transfer_template(command)) != expected.template_commitment
    ):
        raise ResourceOwnerConflict("Resource material command conflicts with its exact reference.")
    _validate_preparation_permit(owner.owner, command, preparation.permit)
    if (
        preparation.owner != owner.owner
        or preparation.operation_digest != digest
        or preparation.command_bytes_sha256 != sha256(record["command"].encode()).hexdigest()
        or preparation.lease.operation_digest != digest
        or preparation.lease.receiver != owner.owner
    ):
        raise ResourceOwnerUnavailable("Resource material preparation identity conflicts.")
    found = await owner._read_transfer_for_cleanup(command)
    if not isinstance(found, ExactMatch) or found.receipt.command != command:
        raise ResourceOwnerUnavailable("Resource material is no longer accepted.")
    actual = material_reference(found.receipt, preparation)
    if actual != expected:
        raise ResourceOwnerConflict("Resource material receipt conflicts with its exact reference.")
    return ResourceMaterialReadback(
        reference=expected, transfer=found.receipt, preparation=preparation
    )


async def read_material(owner, expected):
    retained = await retained_material(owner, expected)
    command = retained.transfer.command
    # The current lease/registration remains authoritative. Historical hashes
    # cannot renew disclosure, adoption or acquisition permission.
    preparation = await owner._require_authorization(command, retained.preparation)
    async with owner._preparation_reader.revalidation_guard(command, preparation.lease):
        if owner._preparation_reader.transfer_permit(command) != preparation.permit:
            raise ResourceOwnerUnavailable("Resource material preparation authority conflicts.")
        return retained
